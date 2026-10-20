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


N_DRAWS = 1000          # realisations of a randomised rule; seeds DRAW_SEED0 .. DRAW_SEED0 + N_DRAWS - 1
DRAW_SEED0 = 20_000


def evaluate(y, pred, te, randomised=False, n_boot=300, n_draws=N_DRAWS, seed=0):
    """Metrics of a decision rule on the test set.

    Deterministic rule: `pred` is 0/1 and every metric is computed from the decisions.
    Randomised rule: `pred` is the flag PROBABILITY of each appointment. Headline metrics are expected values:
    group TPR/FPR = mean flag probability over that group's no-shows/attendances, and recall, precision and F1 are
    ratios of expected counts (E[TP]/E[no-shows], E[TP]/E[flags]); they are not averages of per-draw metrics.

    Uncertainty (cycle 6 correction). The 95% interval is a percentile bootstrap that resamples test appointments
    with replacement (`n_boot` resamples, numpy default_rng(seed)) and recomputes the SAME quantity as the point
    estimate: decisions for a deterministic rule, expected values for a randomised rule. Until cycle 5 the
    randomised rule's interval bootstrapped one random realisation instead, which is an interval for that draw,
    not for the expected gap; that superseded interval is still reported, labelled, for the audit trail.
    Draw-to-draw variation of a randomised rule is reported separately from `n_draws` realisations."""
    out = {"randomised": randomised, **overall(y, pred)}
    out["share_decisions_randomised"] = float(((pred > 0) & (pred < 1)).mean())
    out["metric_definition"] = ("expected values from flag probabilities (ratios of expected counts)" if randomised
                                else "computed from 0/1 decisions")
    rng = np.random.default_rng(seed)
    one_draw = (np.random.default_rng(seed + 1).random(len(pred)) < pred).astype(int) if randomised else None
    for a in ATTRS:
        g = te[a].astype(str).values
        out[a] = group_table(y, pred, g)
        bs, bs_old = [], []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            bs.append(eod_only(y[i], pred[i], g[i]))
            if randomised:
                bs_old.append(eod_only(y[i], one_draw[i], g[i]))
        out[a]["eod_bootstrap_95ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out[a]["ci_method"] = (f"percentile bootstrap, {n_boot} resamples of test appointments (numpy seed {seed}), "
                               + ("of the expected-value gap" if randomised else "of the decision-based gap"))
        if randomised:
            out[a]["superseded_single_draw_bootstrap_95ci"] = [float(np.percentile(bs_old, 2.5)), float(np.percentile(bs_old, 97.5))]
            real = []
            for s in range(n_draws):
                d = (np.random.default_rng(DRAW_SEED0 + s).random(len(pred)) < pred).astype(int)
                real.append(eod_only(y, d, g))
            real = np.array(real)
            first = real[:200]
            out[a]["realised_eod"] = {"p05": float(np.percentile(real, 5)), "p50": float(np.median(real)),
                                      "p95": float(np.percentile(real, 95)), "mean": float(real.mean()),
                                      "share_above_0.05": float(np.mean(real > 0.05)), "n_draws": n_draws,
                                      "seeds": f"numpy default_rng({DRAW_SEED0}) .. default_rng({DRAW_SEED0 + n_draws - 1})",
                                      "monte_carlo_se_share": float(np.sqrt(np.mean(real > 0.05) * (1 - np.mean(real > 0.05)) / n_draws)),
                                      "first_200_draws": {"share_above_0.05": float(np.mean(first > 0.05)),
                                                          "p95": float(np.percentile(first, 95)),
                                                          "note": "the cycle-5 figures (200 draws, same seeds)"}}
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


# ---------------------------------------------------------------- original acceptance criteria (fixed; not edited after results)
# Sources: Module 4 report section 7 ("accepted for staging if ROC-AUC >= 0.70, Recall >= 0.65, all fairness metrics PASS
# (<5% disparity, DIR >= 0.80)") and section 1 (success criteria incl. F1 >= 0.55; protected attributes Gender and age_band);
# Module 1 Charter Principle 2 ("error rate disparity between any two groups must not exceed 5%. Automatic halt if
# threshold breached"); Module 5 (welfare scholarship used as the stand-in for goal G3, Medicaid vs private).
CRITERIA = [
    ("C1", "ROC-AUC >= 0.70", "Module 4 s.7"),
    ("C2", "Recall >= 0.65", "Module 4 s.7"),
    ("C3", "F1 >= 0.55", "Module 4 s.1 ('the acceptance threshold'; omitted from the s.7 staging list) - blocking by reviewer decision"),
    ("C4", "Equalized-odds gap <= 0.05 for age band AND gender", "Module 4 s.7 / Charter P2"),
    ("C5", "Demographic-parity gap <= 0.05 for age band AND gender", "Module 4 s.7 ('all fairness metrics')"),
    ("C6", "Disparate-impact ratio >= 0.80 for age band AND gender", "Module 4 s.7"),
    ("C7", "Equalized-odds gap <= 0.05 for scholarship (G3 stand-in)", "Module 1 G3 via Module 5 stand-in"),
    ("C8", "Limit holds reliably: upper 95% bound of every gap <= 0.05 (and, for randomised rules, the 95th "
           "percentile of realised gaps <= 0.05)", "Charter P2 ('automatic halt if threshold breached')"),
]


def acceptance(v, auc):
    a, g, sc = v["age_band"], v["Gender"], v["Scholarship"]
    rel = all(x["eod_bootstrap_95ci"][1] <= 0.05 for x in (a, g, sc))
    if v["randomised"]:
        rel = rel and all(x["realised_eod"]["p95"] <= 0.05 for x in (a, g, sc))
    res = {"C1": auc >= 0.70, "C2": v["recall"] >= 0.65, "C3": v["f1"] >= 0.55,
           "C4": a["eod"] <= 0.05 and g["eod"] <= 0.05, "C5": a["dpd"] <= 0.05 and g["dpd"] <= 0.05,
           "C6": a["di_ratio"] >= 0.80 and g["di_ratio"] >= 0.80, "C7": sc["eod"] <= 0.05, "C8": rel}
    res["meets_full_fairness_requirement"] = all(res[c] for c in ["C4", "C5", "C6", "C7", "C8"])
    res["meets_all_original_criteria"] = all(res[c] for c in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"])
    return {k: bool(x) for k, x in res.items()}


def option_b_mechanism(policy, te, p, y):
    """Why Option B flags appointments regardless of predicted risk (Fairlearn equalized-odds post-processing)."""
    g = te["age_band"].astype(str).values
    rows = {}
    for grp, r in policy["groups"].items():
        k = g == grp
        floor = r["p_ignore"] * r["prediction_constant"]
        rows[grp] = {"n_test": int(k.sum()), "p_ignore": r["p_ignore"], "prediction_constant": r["prediction_constant"],
                     "risk_unrelated_flag_probability": floor,
                     "lowest_score_in_group_flag_probability": floor,
                     "binding_group": r["p_ignore"] == 0.0}
    floors = np.array([rows[x]["risk_unrelated_flag_probability"] for x in g])
    from careconnect.fairness.policy import flag_probability
    pmf = np.array([flag_probability(s, x, policy) for s, x in zip(p, g)])
    return {"groups": rows, "x_best_common_false_positive_rate": next(iter(policy["groups"].values()))["prediction_constant"],
            "expected_risk_unrelated_flags_per_1000": float(floors.mean() * 1000),
            "expected_total_flags_per_1000": float(pmf.mean() * 1000),
            "share_of_flags_unrelated_to_risk": float(floors.sum() / pmf.sum()),
            "share_of_appointments_with_nonzero_flag_probability": float((pmf > 0).mean()),
            "explanation": (
                "Fairlearn's equalized-odds post-processing (Hardt et al., 2016) picks one operating point (common false-positive "
                "rate x_best, true-positive rate y_best) that every group can reach: the best point on the LOWEST of the groups' "
                "ROC curves. Groups whose ROC curve lies above that point are pulled down to it by mixing their own threshold rule "
                "with the ROC diagonal: with probability p_ignore the score is ignored and the appointment is flagged with "
                "probability x_best. That product is a flag probability that is the same for every appointment in the group, "
                "whatever its score. The group with the lowest curve (here: senior) needs no mixing (p_ignore = 0).")}


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
    res["criteria"] = [{"id": c, "criterion": t, "source": src} for c, t, src in CRITERIA]
    for k, v in P.items():
        v["acceptance"] = acceptance(v, v.get("test_roc_auc", res["reference_test_roc_auc_full_model"]))
    res["no_policy_reliably_meets_fairness"] = not any(v["acceptance"]["meets_full_fairness_requirement"] for v in P.values())
    res["policies_meeting_full_fairness_requirement"] = [k for k, v in P.items() if v["acceptance"]["meets_full_fairness_requirement"]]
    from careconnect.fairness.policy import load_policy
    res["option_B_mechanism"] = option_b_mechanism(load_policy(), te, pte, yte)

    res["note"] = ("Deterministic rules can equalise one error rate per group (TPR or FPR) but generally not both, "
                   "because the groups' ROC curves differ; Fairlearn reaches equal TPR and FPR by randomising between "
                   "two cut-offs. The bootstrap intervals show how precisely a 22,103-row test set can measure a 0.05 gap "
                   "for groups with a few hundred no-shows.")
    json.dump(res, open(config.ARTIFACTS / "policy_comparison.json", "w"), indent=1, default=float)
    print(f"{'policy':38s} {'recall':>6s} {'prec':>6s} {'F1':>6s} {'flag':>6s} {'ageEOD':>7s} {'ageTPRg':>7s} {'ageFPRg':>7s} {'genEOD':>7s} {'schEOD':>7s} {'rand%':>6s}")
    for k, v in P.items():
        print(f"{k:38s} {v['recall']:6.3f} {v['precision']:6.3f} {v['f1']:6.3f} {v['flag_rate']:6.3f} "
              f"{v['age_band']['eod']:7.3f} {v['age_band']['tpr_gap']:7.3f} {v['age_band']['fpr_gap']:7.3f} "
              f"{v['Gender']['eod']:7.3f} {v['Scholarship']['eod']:7.3f} {v['share_decisions_randomised']*100:6.1f} "
              + " ".join(c for c in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"] if not v["acceptance"][c]))
    print("no policy reliably meets the fairness requirement:", res["no_policy_reliably_meets_fairness"])
    print("Option B risk-unrelated share of flags:", round(res["option_B_mechanism"]["share_of_flags_unrelated_to_risk"], 4))


if __name__ == "__main__":
    main()
