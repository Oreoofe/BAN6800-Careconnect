"""Compare outreach decision policies on the same held-out test set (capstone verification cycle 2).

    python -m careconnect.fairness.policy_comparison      -> artifacts/policy_comparison.json

Why: Option B (Fairlearn equalized-odds post-processing) meets the 0.05 limit only in expectation and
randomises ~82% of decisions. This module asks whether a deterministic rule can do comparably, and what
each choice costs. Every rule is fitted on 5-fold out-of-fold TRAINING scores (or training data) and
evaluated once on the TEST set. Nothing is tuned on the test set.

Candidates
  P0  single cut-off (recall >= 0.65 on training OOF)                         deterministic
  P1  equal opportunity: one cut-off per age group, each reaching TPR >= r     deterministic
      (r = 0.65, 0.75, 0.85; r = 0.85 roughly matches Option B's recall)
  P2  minimax: per-age-group cut-offs chosen by coordinate descent to minimise
      the training-OOF equalized-odds gap, subject to overall recall >= 0.65   deterministic
  P3  Fairlearn ThresholdOptimizer, true-positive-rate parity                   randomised
  P4  Option B: ThresholdOptimizer, equalized odds (current policy)            randomised
  P5  "unaware" model: XGBoost retrained without Age and age_band, single
      cut-off (fairness through unawareness)                                   deterministic

Reported for each: recall, precision, F1, flag rate; for age band, gender and scholarship the
demographic-parity difference (DPD), equalized-odds difference (EOD = max of TPR gap and FPR gap),
TPR gap, FPR gap, disparate-impact ratio (DI = min/max selection rate), per-group n / no-shows /
selection / TPR / FPR, and a 300-sample bootstrap 95% interval for each EOD. Randomised rules are
reported in expectation plus the spread of 200 independent realisations.
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from careconnect import config
from careconnect.fairness.mitigation import ScoreOnly
from careconnect.modeling.data import preprocessor, xy

warnings.filterwarnings("ignore")
ATTRS = ["age_band", "Gender", "Scholarship"]
GRID = np.round(np.arange(0.02, 0.9001, 0.0025), 4)


# ---------------------------------------------------------------- metrics
def group_table(y, pred, groups):
    """pred may be 0/1 decisions or flag probabilities (expected values)."""
    rows = {}
    for v in sorted(np.unique(groups)):
        k = groups == v
        rows[str(v)] = {"n": int(k.sum()), "no_shows": int(y[k].sum()), "no_show_rate": float(y[k].mean()),
                        "selection": float(pred[k].mean()), "tpr": float(pred[k & (y == 1)].mean()),
                        "fpr": float(pred[k & (y == 0)].mean())}
    sel = [r["selection"] for r in rows.values()]
    tg = max(r["tpr"] for r in rows.values()) - min(r["tpr"] for r in rows.values())
    fg = max(r["fpr"] for r in rows.values()) - min(r["fpr"] for r in rows.values())
    return {"dpd": max(sel) - min(sel), "eod": max(tg, fg), "tpr_gap": tg, "fpr_gap": fg,
            "di_ratio": (min(sel) / max(sel)) if max(sel) > 0 else None, "groups": rows}


def eod_only(y, pred, groups):
    t, f = [], []
    for v in np.unique(groups):
        k = groups == v
        t.append(pred[k & (y == 1)].mean())
        f.append(pred[k & (y == 0)].mean())
    return max(max(t) - min(t), max(f) - min(f))


def overall(y, pred):
    tp = float((pred * y).sum())
    flagged = float(pred.sum())
    rec = tp / y.sum()
    prec = tp / flagged if flagged else 0.0
    return {"recall": rec, "precision": prec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "flag_rate": float(pred.mean()), "flags_per_1000": float(pred.mean() * 1000),
            "true_no_shows_flagged_per_1000": float((pred * y).mean() * 1000)}


def evaluate(y, pred, te, randomised=False, n_boot=300, n_draws=200, seed=0):
    out = {"randomised": randomised, **overall(y, pred)}
    out["share_decisions_randomised"] = float(((pred > 0) & (pred < 1)).mean())
    rng = np.random.default_rng(seed)
    for a in ATTRS:
        g = te[a].astype(str).values
        out[a] = group_table(y, pred, g)
        # bootstrap the audit itself (sampling uncertainty of the test set)
        if randomised:
            draws = (rng.random(len(pred)) < pred).astype(int)      # one realisation, then bootstrap it
        else:
            draws = pred
        bs = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            bs.append(eod_only(y[i], draws[i], g[i]))
        out[a]["eod_bootstrap_95ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        if randomised:
            real = []
            for s in range(n_draws):
                d = (np.random.default_rng(20_000 + s).random(len(pred)) < pred).astype(int)
                real.append(eod_only(y, d, g))
            out[a]["realised_eod"] = {"p05": float(np.percentile(real, 5)), "p50": float(np.median(real)),
                                      "p95": float(np.percentile(real, 95)),
                                      "share_above_0.05": float(np.mean(np.array(real) > 0.05)), "n_draws": n_draws}
    return out


# ---------------------------------------------------------------- rules
def single_cutoff(y_tr, oof, target=config.RECALL_TARGET):
    rec = np.array([recall_score(y_tr, (oof >= t).astype(int)) for t in GRID])
    return float(GRID[rec >= target].max())


def tpr_cutoffs(y_tr, oof, g_tr, target):
    th = {}
    for v in np.unique(g_tr):
        k = g_tr == v
        pos = np.sort(oof[k & (y_tr == 1)])
        # largest cut-off whose TPR in this group is >= target
        idx = int(np.floor((1 - target) * len(pos)))
        th[str(v)] = float(pos[min(idx, len(pos) - 1)])
    return th


def apply_cutoffs(scores, groups, th):
    cut = pd.Series(groups).astype(str).map(th).values.astype(float)
    return (np.asarray(scores) >= cut).astype(int)


def minimax_cutoffs(y_tr, oof, g_tr, min_recall=config.RECALL_TARGET, iters=6):
    """Coordinate descent over per-group cut-offs to minimise the training-OOF equalized-odds gap,
    subject to overall training-OOF recall >= min_recall. Deterministic; no test data."""
    groups = [str(v) for v in np.unique(g_tr)]
    th = tpr_cutoffs(y_tr, oof, g_tr, 0.75)           # start from an equal-opportunity point
    cand = GRID[(GRID > 0.05) & (GRID < 0.6)]

    def objective(t):
        yh = apply_cutoffs(oof, g_tr, t)
        rec = (yh * y_tr).sum() / y_tr.sum()
        return eod_only(y_tr, yh, g_tr) + (10 if rec < min_recall else 0), rec
    best, _ = objective(th)
    history = [best]
    for _ in range(iters):
        for gname in groups:
            for c in cand[::2]:
                trial = {**th, gname: float(c)}
                val, _ = objective(trial)
                if val < best - 1e-9:
                    best, th = val, trial
        history.append(best)
    return th, {"train_oof_eod": best, "train_oof_recall": objective(th)[1], "history": history}


def threshold_optimizer(oof, y_tr, g_tr, constraint):
    from fairlearn.postprocessing import ThresholdOptimizer
    to = ThresholdOptimizer(estimator=ScoreOnly(), constraints=constraint, objective="balanced_accuracy_score",
                            predict_method="predict_proba", prefit=True)
    to.fit(pd.DataFrame({"p": oof.astype("float64")}), y_tr, sensitive_features=g_tr)
    return to


def main():
    m = joblib.load(config.ARTIFACTS / "xgboost.joblib")
    tr = pd.read_pickle(config.ARTIFACTS / "train_frame.pkl")
    te = pd.read_pickle(config.ARTIFACTS / "test_frame.pkl")
    Xtr, ytr = xy(tr)
    Xte, yte = xy(te)
    ytr, yte = ytr.values, yte.values
    oof = np.load(config.ARTIFACTS / "oof_xgb.npy")
    pte = m.predict_proba(Xte)[:, 1]
    gtr, gte = tr["age_band"].astype(str).values, te["age_band"].astype(str).values
    res = {"model_version": config.MODEL_VERSION, "n_test": len(yte), "test_no_show_rate": float(yte.mean()),
           "fitted_on": "training out-of-fold scores (5-fold)", "evaluated_on": "held-out test set, once",
           "policies": {}}
    P = res["policies"]

    T0 = single_cutoff(ytr, oof)
    P["P0_single_cutoff"] = {"rule": f"flag if score >= {T0}", **evaluate(yte, (pte >= T0).astype(int), te)}

    for r in [0.65, 0.75, 0.85]:
        th = tpr_cutoffs(ytr, oof, gtr, r)
        P[f"P1_equal_opportunity_tpr{int(r*100)}"] = {"rule": "per-age-group cut-offs, training TPR >= %.2f" % r,
                                                       "cutoffs": th, **evaluate(yte, apply_cutoffs(pte, gte, th), te)}

    th, info = minimax_cutoffs(ytr, oof, gtr)
    P["P2_minimax_per_group"] = {"rule": "per-age-group cut-offs minimising training-OOF equalized-odds gap (recall >= 0.65)",
                                 "cutoffs": th, "training": info, **evaluate(yte, apply_cutoffs(pte, gte, th), te)}

    for key, constraint in [("P3_tpr_parity_randomised", "true_positive_rate_parity"),
                            ("P4_option_B_equalized_odds", "equalized_odds")]:
        to = threshold_optimizer(oof, ytr, gtr, constraint)
        pmf = to._pmf_predict(pd.DataFrame({"p": pte.astype("float64")}), sensitive_features=gte)[:, 1]
        P[key] = {"rule": f"Fairlearn ThresholdOptimizer ({constraint}), expected values",
                  **evaluate(yte, pmf, te, randomised=True)}

    # P5: fairness through unawareness - retrain without Age / age_band, same hyper-parameters
    feats = [f for f in config.FEATURES if f not in ("Age", "age_band")]
    clf = clone(m.named_steps["clf"])
    un = Pipeline([("pre", preprocessor(feats)), ("clf", clf)])
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    oof_u = cross_val_predict(clone(un), Xtr[feats], ytr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    un.fit(Xtr[feats], ytr)
    pu = un.predict_proba(Xte[feats])[:, 1]
    Tu = single_cutoff(ytr, oof_u)
    from sklearn.metrics import roc_auc_score
    P["P5_unaware_model_no_age"] = {"rule": f"model without Age/age_band; flag if score >= {Tu}",
                                    "test_roc_auc": float(roc_auc_score(yte, pu)),
                                    **evaluate(yte, (pu >= Tu).astype(int), te)}
    res["reference_test_roc_auc_full_model"] = float(roc_auc_score(yte, pte))

    res["note"] = ("Deterministic rules can equalise one error rate per group (TPR or FPR) but generally not both, "
                   "because the groups' ROC curves differ; Fairlearn reaches equal TPR and FPR by randomising between "
                   "two cut-offs. The bootstrap intervals show how precisely a 22,103-row test set can measure a 0.05 gap "
                   "for groups with a few hundred no-shows.")
    json.dump(res, open(config.ARTIFACTS / "policy_comparison.json", "w"), indent=1, default=float)
    print(f"{'policy':38s} {'recall':>6s} {'prec':>6s} {'F1':>6s} {'flag':>6s} {'ageEOD':>7s} {'ageTPRg':>7s} {'ageFPRg':>7s} {'genEOD':>7s} {'schEOD':>7s} {'rand%':>6s}")
    for k, v in P.items():
        print(f"{k:38s} {v['recall']:6.3f} {v['precision']:6.3f} {v['f1']:6.3f} {v['flag_rate']:6.3f} "
              f"{v['age_band']['eod']:7.3f} {v['age_band']['tpr_gap']:7.3f} {v['age_band']['fpr_gap']:7.3f} "
              f"{v['Gender']['eod']:7.3f} {v['Scholarship']['eod']:7.3f} {v['share_decisions_randomised']*100:6.1f}")


if __name__ == "__main__":
    main()
