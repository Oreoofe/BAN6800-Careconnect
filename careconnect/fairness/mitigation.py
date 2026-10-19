"""Bias mitigation: compare the standard cut-off with three alternatives, all evaluated the same way.

    python -m careconnect.fairness.mitigation

* before  - single cut-off chosen for recall >= 0.65 (training OOF predictions)
* A       - one cut-off per age group, each set for recall >= 0.65 on training OOF predictions
* B       - Fairlearn ThresholdOptimizer, equalized-odds constraint, fitted on training OOF scores
            (CONFIGURED for evaluation, NOT approved; exported to artifacts/fairness_policy.json)
* C       - ThresholdOptimizer with a demographic-parity constraint (for comparison only)

All rules are fitted on training data only and evaluated once on the test set.
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_score, recall_score

from careconnect import config
from careconnect.fairness.metrics import crosscheck_with_fairlearn, fairness_report
from careconnect.fairness.policy import POLICY_PATH, apply_policy, export_threshold_optimizer
from careconnect.modeling.data import xy

warnings.filterwarnings("ignore")


class ScoreOnly(BaseEstimator, ClassifierMixin):
    """Pre-fitted wrapper so ThresholdOptimizer post-processes existing scores (column 'p')."""
    classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.c_[1 - X["p"].values, X["p"].values]

    def predict(self, X):
        return (X["p"].values >= 0.5).astype(int)

    def __sklearn_is_fitted__(self):
        return True


def summary(y, yh, te):
    d = {"recall": recall_score(y, yh), "precision": precision_score(y, yh, zero_division=0), "flag_rate": float(yh.mean())}
    for key, col in [("age_band", te["age_band"]), ("Gender", te["Gender"]), ("Scholarship", te["Scholarship"])]:
        r = fairness_report(y, yh, col.astype(str).values)
        d[key] = {"dpd": r["dpd"], "eod": r["eod"], "di_ratio": r["di_ratio"], "tpr_gap": r["tpr_gap"], "fpr_gap": r["fpr_gap"]}
    d["age_groups"] = {g: {k: v[k] for k in ["flag_rate", "tpr", "fpr", "n", "no_show_rate"]}
                       for g, v in fairness_report(y, yh, te["age_band"].astype(str).values)["groups"].items()}
    return d


ATTRS = ["age_band", "Gender", "Scholarship"]
N_SEEDS = 200


def expected_summary(y, pmf, te):
    """Fairness of a randomised rule in expectation: group TPR/FPR/selection = mean flag probability.
    This is the quantity the rule is designed to equalise; any single realisation scatters around it."""
    d = {"recall": float((pmf * y).sum() / y.sum()), "precision": float((pmf * y).sum() / pmf.sum()),
         "flag_rate": float(pmf.mean())}
    for a in ATTRS:
        g = te[a].astype(str).values
        rows = {}
        for v in sorted(np.unique(g)):
            k = g == v
            rows[v] = {"n": int(k.sum()), "selection": float(pmf[k].mean()), "tpr": float(pmf[k & (y == 1)].mean()),
                       "fpr": float(pmf[k & (y == 0)].mean())}
        sel = [r["selection"] for r in rows.values()]
        tpr_gap = max(r["tpr"] for r in rows.values()) - min(r["tpr"] for r in rows.values())
        fpr_gap = max(r["fpr"] for r in rows.values()) - min(r["fpr"] for r in rows.values())
        d[a] = {"dpd": max(sel) - min(sel), "eod": max(tpr_gap, fpr_gap), "di_ratio": min(sel) / max(sel),
                "tpr_gap": tpr_gap, "fpr_gap": fpr_gap, "groups": rows}
    return d


def seed_distribution(y, pmf, te, n=N_SEEDS):
    """Equalized-odds gap of n independent realisations of the randomised rule."""
    out = {a: [] for a in ATTRS}
    for s in range(n):
        yh = (np.random.default_rng(10_000 + s).random(len(pmf)) < pmf).astype(int)
        for a in ATTRS:
            out[a].append(fairness_report(y, yh, te[a].astype(str).values)["eod"])
    return {a: {"mean": float(np.mean(v)), "p05": float(np.percentile(v, 5)), "p50": float(np.median(v)),
                "p95": float(np.percentile(v, 95)), "max": float(np.max(v)),
                "share_above_limit": float(np.mean(np.array(v) > config.FAIRNESS_LIMIT)), "n_seeds": n}
            for a, v in out.items()}


def main():
    from fairlearn.postprocessing import ThresholdOptimizer
    m = joblib.load(config.ARTIFACTS / "xgboost.joblib")
    tr = pd.read_pickle(config.ARTIFACTS / "train_frame.pkl")
    te = pd.read_pickle(config.ARTIFACTS / "test_frame.pkl")
    _, ytr = xy(tr)
    Xte, yte = xy(te)
    ytr, yte = ytr.values, yte.values
    oof = np.load(config.ARTIFACTS / "oof_xgb.npy")
    pte = m.predict_proba(Xte)[:, 1]
    T = json.load(open(config.ARTIFACTS / "operating_point.json"))["flag_threshold"]
    gtr, gte = tr["age_band"].astype(str).values, te["age_band"].astype(str).values

    res = {"before": summary(yte, (pte >= T).astype(int), te)}
    res["before"]["crosscheck_fairlearn_age"] = crosscheck_with_fairlearn(yte, (pte >= T).astype(int), gte)

    th = {}
    ts = np.linspace(0.02, 0.9, 353)
    for g in sorted(np.unique(gtr)):
        k = gtr == g
        r = np.array([recall_score(ytr[k], (oof[k] >= t).astype(int)) for t in ts])
        th[g] = float(ts[r >= config.RECALL_TARGET].max())
    yA = np.array([pte[i] >= th[g] for i, g in enumerate(gte)]).astype(int)
    res["A_group_thresholds"] = {"thresholds": th, **summary(yte, yA, te)}

    for key, constraint in [("B_equalized_odds", "equalized_odds"), ("C_demographic_parity", "demographic_parity")]:
        to = ThresholdOptimizer(estimator=ScoreOnly(), constraints=constraint, objective="balanced_accuracy_score",
                                predict_method="predict_proba", prefit=True)
        to.fit(pd.DataFrame({"p": oof.astype("float64")}), ytr, sensitive_features=gtr)
        Xp = pd.DataFrame({"p": pte.astype("float64")})
        yh = to.predict(Xp, sensitive_features=gte, random_state=config.RANDOM_STATE).astype(int)
        pmf = to._pmf_predict(Xp, sensitive_features=gte)[:, 1]
        # headline numbers = expected values (deterministic); one realisation and the spread are kept beside them
        res[key] = expected_summary(yte, pmf, te)
        res[key]["evaluation"] = "expected value over the rule's randomisation"
        res[key]["single_realisation_fairlearn_seed42"] = summary(yte, yh, te)
        res[key]["realisation_spread"] = seed_distribution(yte, pmf, te)
        res[key]["age_groups"] = {g: {"flag_rate": r["selection"], "tpr": r["tpr"], "fpr": r["fpr"], "n": r["n"]}
                                  for g, r in res[key]["age_band"]["groups"].items()}
        if key == "B_equalized_odds":
            pol = export_threshold_optimizer(to, sorted(np.unique(gtr)), T)
            lib_pmf = to._pmf_predict(Xp, sensitive_features=gte)[:, 1]
            our_pmf, our_yh = apply_policy(pte, gte, pol)
            pol["verification"] = {"max_abs_diff_vs_fairlearn_pmf": float(np.max(np.abs(lib_pmf - our_pmf))),
                                   "share_randomised": float(((our_pmf > 0) & (our_pmf < 1)).mean())}
            pol["expected_test_metrics"] = {"recall": float((our_pmf * yte).sum() / yte.sum()),
                                            "flag_rate": float(our_pmf.mean())}
            pol["status"] = ("configured for evaluation only - NOT approved; fails the original acceptance criterion C8; "
                             "any use requires an AI Ethics Committee decision")
            json.dump(pol, open(POLICY_PATH, "w"), indent=1)
            res[key]["policy_realisation_numpy_seed42"] = summary(yte, our_yh, te)
            print("policy export check, max |pmf diff| =", pol["verification"]["max_abs_diff_vs_fairlearn_pmf"])
    res["configured"] = "B_equalized_odds"   # configured for evaluation; not an approved policy
    json.dump(res, open(config.ARTIFACTS / "mitigation.json", "w"), indent=1, default=float)
    for k, v in res.items():
        if isinstance(v, dict):
            print(k, round(v["recall"], 3), round(v["flag_rate"], 3), {a: round(v[a]["eod"], 3) for a in ["age_band", "Gender", "Scholarship"]})


if __name__ == "__main__":
    main()
