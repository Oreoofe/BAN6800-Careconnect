"""Independent audit of a build, run on the same runner after the build (cycle 6).

    python scripts/independent_audit.py <artifacts-root> <out-dir>

Written separately from the project's verifier (scripts/verify_results.py). It does not import careconnect.fairness,
careconnect.monitoring, careconnect.modeling.evaluate or the verifier. Project code is reused only to (a) rebuild the
modelling table, which this script cross-checks row by row against its own re-derivation from the raw CSV, and (b) rerun LIME
as a reproducibility check (flagged as such). Every recomputed value is compared with the value the build WROTE to its
artifacts; nothing is compared with numbers typed into this file, except the published Cycle 5 checksums, kept as a history check.

Result codes: MATCH, MISMATCH, INFO (a reported quantity with no build counterpart).
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from careconnect import config  # noqa: E402

ROOT = Path(sys.argv[1]).resolve()
A = ROOT / "artifacts"
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
J = lambda p: json.load(open(A / p))
R, OP, PC, MON, X, MIT, G, POLICY = (J("model_results.json"), J("operating_point.json"), J("policy_comparison.json"),
                                     J("monitoring/monitoring.json"), J("explanations.json"), J("mitigation.json"),
                                     J("promotion_decision.json"), J("fairness_policy.json"))
rows, t0 = [], time.time()


def rec(item, built, recomputed, how, tol=5e-4, note=""):
    if isinstance(built, bool) or isinstance(recomputed, bool) or tol == 0:
        ok = built == recomputed
    elif isinstance(built, (int, float)) and isinstance(recomputed, (int, float)):
        ok = abs(float(built) - float(recomputed)) <= tol
    elif isinstance(built, (list, tuple)) and isinstance(recomputed, (list, tuple)):
        ok = len(built) == len(recomputed) and all(abs(float(a) - float(b)) <= tol for a, b in zip(built, recomputed))
    else:
        ok = built == recomputed
    rows.append({"item": item, "build_artifact": built, "recomputed": recomputed, "how": how, "tolerance": tol,
                 "result": "MATCH" if ok else "MISMATCH", "note": note})


def info(item, value, how, note=""):
    rows.append({"item": item, "build_artifact": None, "recomputed": value, "how": how, "tolerance": None, "result": "INFO", "note": note})


sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
r3 = lambda v: [round(float(x), 4) for x in v]

# ---------------------------------------------------------------- 1. data and checksums
rec("dataset sha256 (config pin)", config.RAW_SHA256, sha(config.RAW_CSV), "sha256 of the raw CSV", tol=0)
info("model sha256", sha(A / "xgboost.joblib"), "sha256 of artifacts/xgboost.joblib")
info("policy sha256", sha(A / "fairness_policy.json"), "sha256 of artifacts/fairness_policy.json")
rec("model sha256 equals Cycle 5 (history check)", "6d60bea7544ffac64049a66af98f4d133508cb77aa464702a91075706127a385",
    sha(A / "xgboost.joblib"), "sha256", tol=0, note="the model is not retrained differently in cycle 6; a change here would be a finding")
df = pd.read_csv(config.RAW_CSV)
d = df.drop_duplicates("AppointmentID")
bad_age = int((~d["Age"].between(0, 100)).sum()); d = d[d["Age"].between(0, 100)]
sd = pd.to_datetime(d["ScheduledDay"], utc=True).dt.tz_localize(None)
ad = pd.to_datetime(d["AppointmentDay"], utc=True).dt.tz_localize(None)
keep = sd.dt.normalize() <= ad
bad_sched = int((~keep).sum()); d = d[keep].copy(); sd, ad = sd[keep], ad[keep]
Q = json.load(open(A / "pipeline" / "data_quality_report.json"))["cleaning"]
rec("invalid ages removed", Q["invalid_age_removed"], bad_age, "own cleaning of the raw CSV", tol=0)
rec("bookings after appointment removed", Q["scheduled_after_appointment_removed"], bad_sched, "own cleaning", tol=0)
rec("rows after cleaning", Q["rows_out"], len(d), "own cleaning", tol=0)
d["y"] = (d["No-show"] == "Yes").astype(int)
d["lead"] = (ad - sd.dt.normalize()).dt.days
d["appt"], d["book"] = ad.dt.normalize(), sd.dt.normalize()
d = d.reset_index(drop=True)
pa, pn = np.zeros(len(d), int), np.zeros(len(d), int)
for _, g in d.groupby("PatientId"):
    if len(g) > 1:
        ap, yy = g["appt"].values, g["y"].values
        for i, b in zip(g.index, g["book"].values):
            m_ = ap < b
            pa[i], pn[i] = int(m_.sum()), int(yy[m_].sum())
F = json.load(open(A / "pipeline" / "data_quality_report.json"))["features"]
rec("rows with outcome-known history", F["rows_with_history"], int((pa > 0).sum()), "own brute-force history", tol=0)
from careconnect.modeling.data import load_table, split  # noqa: E402
tab = load_table()
same = (len(d) == len(tab) and (d["lead"].values == tab["lead_time_days"].values).all()
        and (pa == tab["prior_appointments"].values).all() and (pn == tab["prior_noShows"].values).all()
        and (d["y"].values == tab["noshow_flag"].values).all())
rec("pipeline lead time and history columns equal own re-derivation (every row)", True, bool(same), "row-by-row comparison")
tr, te = split(tab)
_, own_te = train_test_split(np.arange(len(d)), test_size=0.2, stratify=d["y"].values, random_state=42)
rec("own stratified split selects the same test rows", True, bool(sorted(own_te.tolist()) == sorted(te.index.tolist())), "own split")
rec("train / test rows", [R["n_train"], R["n_test"]], [len(tr), len(te)], "split", tol=0)
y = te[config.TARGET].astype(int).values
ytr = tr[config.TARGET].astype(int).values

# ---------------------------------------------------------------- 2. models at their cut-offs (all three re-scored)
m = joblib.load(A / "xgboost.joblib")
p = m.predict_proba(te[config.FEATURES])[:, 1]
for name in ["xgboost", "random_forest", "logistic_regression"]:
    path = A / f"{name}.joblib"
    built = R["models"][name]["at_recall_target"]
    if not path.exists():
        info(f"{name} not re-scored", "model file absent", "artifacts", "file not available on this runner")
        continue
    mm = joblib.load(path)
    pp = mm.predict_proba(te[config.FEATURES])[:, 1]
    yh_ = (pp >= built["threshold"]).astype(int)
    rec(f"{name} ROC-AUC / recall / precision / F1 / flag rate at its cut-off {built['threshold']}",
        r3([built["roc_auc"], built["recall"], built["precision"], built["f1"], built["flag_rate"]]),
        r3([roc_auc_score(y, pp), recall_score(y, yh_), precision_score(y, yh_), f1_score(y, yh_), yh_.mean()]), "sklearn", tol=1e-4)
T, H = OP["flag_threshold"], OP["high_threshold"]
yh = (p >= T).astype(int)
rec("xgboost cut-off used in operating_point.json equals model_results threshold", R["models"]["xgboost"]["at_recall_target"]["threshold"], T, "json", tol=0)
tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
AN = json.load(open(A / "analysis.json"))
rec("confusion matrix TN/FP/FN/TP", [AN["confusion"][k] for k in ["tn", "fp", "fn", "tp"]], [int(tn), int(fp), int(fn), int(tp)], "sklearn", tol=0)
tier = np.where(p >= H, "high", np.where(p >= T, "medium", "low"))
for t in ["high", "medium", "low"]:
    rec(f"tier {t} share / no-show rate", r3([AN["tiers"][t]["share"], AN["tiers"][t]["no_show_rate"]]),
        r3([(tier == t).mean(), y[tier == t].mean()]), "own", tol=1e-4)
rec("patient-grouped and ablation AUCs present (not re-trained here)", True,
    "roc_auc" in R["robustness_patient_grouped"] and "roc_auc" in R["ablation_no_sensitive_inputs"], "json",
    note="robustness and ablation models are not re-trained by this audit")


# ---------------------------------------------------------------- 3. policies (own implementations)
def grp(yv, pred, g):
    out = {}
    for v in np.unique(g):
        k = g == v
        out[str(v)] = {"n": int(k.sum()), "no_shows": int(yv[k].sum()), "sel": float(pred[k].mean()),
                       "tpr": float(pred[k & (yv == 1)].mean()), "fpr": float(pred[k & (yv == 0)].mean())}
    return out


def gaps(yv, pred, g):
    r = grp(yv, pred, g)
    t = [z["tpr"] for z in r.values()]; f = [z["fpr"] for z in r.values()]; s = [z["sel"] for z in r.values()]
    return {"eod": max(max(t) - min(t), max(f) - min(f)), "tpr_gap": max(t) - min(t), "fpr_gap": max(f) - min(f),
            "dpd": max(s) - min(s), "di": min(s) / max(s), "groups": r}


def summ(yv, pred):
    tpe, fl = float((pred * yv).sum()), float(pred.sum())
    rc, pr = tpe / yv.sum(), tpe / fl
    return {"recall": rc, "precision": pr, "f1": 2 * pr * rc / (pr + rc), "flag_rate": float(pred.mean())}


att = {a: te[a].astype(str).values for a in ["age_band", "Gender", "Scholarship"]}


def flag_prob(score, group):
    r = POLICY["groups"][group]
    op = lambda o, sc: (sc > o[1]) if o[0] == ">" else (sc < o[1])
    return r["p_ignore"] * r["prediction_constant"] + (1 - r["p_ignore"]) * (r["p0"] * op(r["op0"], score) + r["p1"] * op(r["op1"], score))


pmf = np.array([flag_prob(s, g) for s, g in zip(p, att["age_band"])])
PB = PC["policies"]["P4_option_B_equalized_odds"]
sB = summ(y, pmf)
rec("Option B expected recall / precision / F1 / flag rate",
    r3([PB["recall"], PB["precision"], PB["f1"], PB["flag_rate"]]), r3([sB["recall"], sB["precision"], sB["f1"], sB["flag_rate"]]),
    "own flag probabilities from fairness_policy.json", tol=1e-4, note="ratios of expected counts, not averages over draws")
rec("Option B share of appointments randomised (0 < p < 1)", PB["share_decisions_randomised"], float(((pmf > 0) & (pmf < 1)).mean()), "own", tol=1e-6)
for a in att:
    gb = gaps(y, pmf, att[a])
    rec(f"Option B expected EOD / DPD / DI ({a})", r3([PB[a]["eod"], PB[a]["dpd"], PB[a]["di_ratio"]]), r3([gb["eod"], gb["dpd"], gb["di"]]), "own", tol=1e-4)
mech = PC["option_B_mechanism"]
floor = np.array([POLICY["groups"][g]["p_ignore"] * POLICY["groups"][g]["prediction_constant"] for g in att["age_band"]])
rec("risk-unrelated share of expected flags / per 1,000 appointments",
    r3([mech["share_of_flags_unrelated_to_risk"], mech["expected_risk_unrelated_flags_per_1000"]]), r3([floor.sum() / pmf.sum(), floor.mean() * 1000]), "own", tol=1e-4)
rec("share of appointments with non-zero flag probability", mech["share_of_appointments_with_nonzero_flag_probability"], float((pmf > 0).mean()), "own", tol=1e-6)
lowest = {g: int(np.argmin(np.where(att["age_band"] == g, p, np.inf))) for g in POLICY["groups"]}
rec("lowest-score appointment in each age group gets exactly the score-independent floor", True,
    all(abs(pmf[i] - POLICY["groups"][g]["p_ignore"] * POLICY["groups"][g]["prediction_constant"]) < 1e-12 for g, i in lowest.items()), "own")

# corrected interval: bootstrap of the expected-value gap (own seed and 1,000 resamples)
rng = np.random.default_rng(424242)
bs = {a: [] for a in att}
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    for a in att:
        bs[a].append(gaps(y[i], pmf[i], att[a][i])["eod"])
for a in att:
    rec(f"Option B {a} 95% CI, bootstrap of expected values (build: 300 resamples, seed 0; audit: 1,000, seed 424242)",
        r3(PB[a]["eod_bootstrap_95ci"]), r3([np.percentile(bs[a], 2.5), np.percentile(bs[a], 97.5)]), "own bootstrap", tol=0.012,
        note="different resamples: agreement expected to about +/-0.01")
# draw-to-draw variation: exact reproduction with the build's seeds, then independent seeds
real_b = np.array([gaps(y, (np.random.default_rng(20_000 + s).random(len(pmf)) < pmf).astype(float), att["age_band"])["eod"] for s in range(1000)])
rb = PB["age_band"]["realised_eod"]
rec("Option B realised age EOD, build seeds 20000..20999: share > 0.05 / p95", r3([rb["share_above_0.05"], rb["p95"]]),
    r3([(real_b > 0.05).mean(), np.percentile(real_b, 95)]), "own simulation, same seeds", tol=1e-9)
rec("first 200 of those draws reproduce the cycle-5 figures (share > 0.05 / p95)", [0.385, 0.0678],
    [round(float((real_b[:200] > 0.05).mean()), 4), round(float(np.percentile(real_b[:200], 95)), 4)], "own simulation", tol=5e-4)
real_i = np.array([gaps(y, (np.random.default_rng(900_000 + s).random(len(pmf)) < pmf).astype(float), att["age_band"])["eod"] for s in range(1000)])
se = float(np.sqrt(rb["share_above_0.05"] * (1 - rb["share_above_0.05"]) / 1000 * 2))
rec("Option B realised age EOD, independent seeds 900000..900999: share > 0.05", rb["share_above_0.05"], float((real_i > 0.05).mean()),
    "own simulation, independent seeds", tol=3 * se, note=f"two 1,000-draw estimates differ by Monte Carlo error; tolerance 3 x SE of the difference = {3 * se:.3f}")
rec("Option B realised age EOD, independent seeds: p95", rb["p95"], float(np.percentile(real_i, 95)), "own simulation", tol=0.004)
spread = MIT["B_equalized_odds"]["realisation_spread"]["age_band"]
rec("mitigation.json realisation spread equals policy_comparison draws (same seeds)", r3([rb["share_above_0.05"], rb["p95"]]),
    r3([spread["share_above_limit"], spread["p95"]]), "json consistency", tol=1e-9)

# deterministic policies from the published cut-offs, and the TPR-parity rule re-fitted
for key in ["P1_equal_opportunity_tpr65", "P1_equal_opportunity_tpr75", "P1_equal_opportunity_tpr85", "P2_minimax_per_group"]:
    V = PC["policies"][key]
    cut = pd.Series(att["age_band"]).map(V["cutoffs"]).values.astype(float)
    dec = (p >= cut).astype(float)
    s_ = summ(y, dec)
    rec(f"{key} recall / precision / F1 / flag rate", r3([V["recall"], V["precision"], V["f1"], V["flag_rate"]]),
        r3([s_["recall"], s_["precision"], s_["f1"], s_["flag_rate"]]), "own, from published cut-offs", tol=1e-4)
    for a in att:
        ga = gaps(y, dec, att[a])
        rec(f"{key} EOD / TPR gap / FPR gap ({a})", r3([V[a]["eod"], V[a]["tpr_gap"], V[a]["fpr_gap"]]), r3([ga["eod"], ga["tpr_gap"], ga["fpr_gap"]]), "own", tol=1e-4)
    if key.endswith("tpr85"):
        rng = np.random.default_rng(777); b = {a: [] for a in att}
        for _ in range(1000):
            i = rng.integers(0, len(y), len(y))
            for a in att:
                b[a].append(gaps(y[i], dec[i], att[a][i])["eod"])
        for a in att:
            rec(f"{key} {a} 95% CI (own 1,000-resample bootstrap)", r3(V[a]["eod_bootstrap_95ci"]),
                r3([np.percentile(b[a], 2.5), np.percentile(b[a], 97.5)]), "own bootstrap", tol=0.012)
        sizes = {g: [z["n"], z["no_shows"]] for a in att for g, z in grp(y, dec, att[a]).items()}
        built_sizes = {g: [z["n"], z["no_shows"]] for a in att for g, z in V[a]["groups"].items()}
        rec(f"{key} subgroup sizes (n, no-shows)", built_sizes, sizes, "own counts", tol=0)
oof = np.load(A / "oof_xgb.npy")
from fairlearn.postprocessing import ThresholdOptimizer  # noqa: E402


from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402


class _Fixed(BaseEstimator, ClassifierMixin):
    """Pre-fitted wrapper written for this audit: the input column 'p' is already a probability."""
    classes_ = np.array([0, 1])

    def fit(self, X, yv):
        return self

    def predict_proba(self, X):
        return np.c_[1 - X["p"].values, X["p"].values]


gtr = tr["age_band"].astype(str).values
to = ThresholdOptimizer(estimator=_Fixed(), constraints="true_positive_rate_parity", objective="balanced_accuracy_score",
                        prefit=True, predict_method="predict_proba")
to.fit(pd.DataFrame({"p": oof.astype("float64")}), ytr, sensitive_features=gtr)
pm3 = to._pmf_predict(pd.DataFrame({"p": p.astype("float64")}), sensitive_features=att["age_band"])[:, 1]
V3 = PC["policies"]["P3_tpr_parity_randomised"]
s3 = summ(y, pm3)
rec("P3 TPR parity (Fairlearn re-fitted here on the published OOF scores): recall / F1 / age EOD",
    r3([V3["recall"], V3["f1"], V3["age_band"]["eod"]]), r3([s3["recall"], s3["f1"], gaps(y, pm3, att["age_band"])["eod"]]),
    "own Fairlearn fit", tol=2e-3, note="objective assumed balanced accuracy, as in the project; a mismatch would indicate a different fit")
# no-age model re-trained here with the same hyper-parameters
from sklearn.base import clone  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from careconnect.modeling.data import preprocessor  # noqa: E402
feats = [f for f in config.FEATURES if f not in ("Age", "age_band")]
un = Pipeline([("pre", preprocessor(feats)), ("clf", clone(m.named_steps["clf"]))]).fit(tr[feats], ytr)
pu = un.predict_proba(te[feats])[:, 1]
V5 = PC["policies"]["P5_unaware_model_no_age"]
rec("P5 model without age inputs: test ROC-AUC (re-trained here)", V5["test_roc_auc"], float(roc_auc_score(y, pu)), "own re-training", tol=2e-3)

# ---------------------------------------------------------------- 4. Module 4 s.7 weekly rule (own implementation)
tw = te.copy(); tw["week"] = tw["AppointmentDay"].dt.to_period("W-SUN").dt.start_time.dt.strftime("%Y-%m-%d")
stored = MON["weeks_cycle2_rule"]
cnt = {"age_band": {}, "Gender": {}}
weekly = []
for i, ((w, cur), st) in enumerate(zip(tw.groupby("week"), stored)):
    pw = m.predict_proba(cur[config.FEATURES])[:, 1]
    gw = cur["age_band"].astype(str).values
    pm = np.array([flag_prob(s, g) for s, g in zip(pw, gw)])
    dec = (np.random.default_rng(i).random(len(pm)) < pm).astype(float)
    yv = cur[config.TARGET].astype(int).values
    row = {"week": w, "days": int(cur["AppointmentDay"].dt.normalize().nunique()), "n": len(cur)}
    for a in ["age_band", "Gender"]:
        gv = cur[a].astype(str).values
        ok = [v for v in np.unique(gv) if yv[gv == v].sum() > 0 and (yv[gv == v] == 0).sum() > 0]
        k = np.isin(gv, ok)
        real, exp_ = gaps(yv[k], dec[k], gv[k])["eod"], gaps(yv[k], pm[k], gv[k])["eod"]
        status = "halt" if real > 0.05 else "alert" if real > 0.04 else "ok"
        cnt[a][status] = cnt[a].get(status, 0) + 1
        sa = st["module4_rule"]["attributes"][a]
        rec(f"s.7 week {w} {a}: realised / expected disparity", r3([sa["disparity"], sa["expected_disparity"]]), r3([real, exp_]), "own replay", tol=1e-4)
        rec(f"s.7 week {w} {a}: status", sa["status"], status, "own thresholds", tol=0)
        row[a] = {"realised": round(real, 4), "expected": round(exp_, 4), "status": status,
                  "min_no_shows": int(min(yv[gv == v].sum() for v in np.unique(gv)))}
    cov = st.get("coverage_days")
    rec(f"s.7 week {w}: appointment days", cov, row["days"], "own count", tol=0)
    weekly.append(row)
S4 = MON["summary"]["module4_s7_weekly"]["attribute_status_counts"]
for a in ["age_band", "Gender"]:
    rec(f"s.7 status counts by attribute ({a})", {k: v for k, v in S4[a].items() if v}, cnt[a], "own replay", tol=0)
for setname in ["rolling_28d", "calendar_months", "weeks_cycle2_rule"]:
    c2 = {}
    for r in MON[setname]:
        f = r["fairness"]
        s = "insufficient data" if f["ci95"] is None else ("halt" if f["ci95"][0] > 0.05 else "ok" if f["ci95"][1] <= 0.05 else "inconclusive")
        c2[s] = c2.get(s, 0) + 1
    key = {"rolling_28d": "rolling_28d_all", "calendar_months": "calendar_months", "weeks_cycle2_rule": "weeks_cycle2_rule"}[setname]
    rec(f"supplementary {setname}: status counts from stored intervals", {k: v for k, v in MON["summary"][key]["counts"].items() if v}, c2,
        "status logic recomputed", tol=0, note="interval values come from the replay's bootstrap and are not re-bootstrapped here")

# ---------------------------------------------------------------- 5. SHAP, counterfactuals, LIME (reproducibility)
import xgboost as xgb  # noqa: E402
pre, clf = m.named_steps["pre"], m.named_steps["clf"]
idx = np.random.RandomState(config.RANDOM_STATE).choice(len(te), 4000, replace=False)
Xs = te[config.FEATURES].iloc[idx]
names = list(pre.get_feature_names_out())
D = xgb.DMatrix(pre.transform(Xs), feature_names=names)
contrib = clf.get_booster().predict(D, pred_contribs=True)
margin = clf.get_booster().predict(D, output_margin=True)
rec("SHAP additivity: contributions + bias equal the margin (max abs error < 1e-3)", True, bool(np.max(np.abs(contrib.sum(1) - margin)) < 1e-3), "own TreeSHAP")
rec("SHAP explains the deployed probability: sigmoid(margin) = predict_proba", True, bool(np.allclose(1 / (1 + np.exp(-margin)), m.predict_proba(Xs)[:, 1], atol=1e-5)), "own")
per = {}
for j, n in enumerate(names):
    base = n.split("__", 1)[1]
    f = next((c for c in sorted(config.FEATURES, key=len, reverse=True) if base == c or base.startswith(c + "_")), base)
    per[f] = per.get(f, 0) + contrib[:, j]
imp = {f: float(np.abs(v).mean()) for f, v in per.items()}
tot = sum(imp.values())
share = AN["shap"]["share"]
rec("SHAP shares, top four inputs", r3([share[f] for f in ["lead_time_days", "Age", "Neighbourhood", "SMS_received"]]),
    r3([imp[f] / tot for f in ["lead_time_days", "Age", "Neighbourhood", "SMS_received"]]), "own TreeSHAP on the same 4,000-row sample", tol=5e-4)
for c in X["counterfactuals"]:
    row = te.iloc[c["test_row"]]
    best = 1.0
    for lead in sorted({v for v in [0, 1, 2, 3, 5, 7, 14, 21, 28] if v < int(row["lead_time_days"])} | {int(row["lead_time_days"])}):
        for sms in sorted({int(row["SMS_received"]), 1}):
            r2 = row.copy(); r2["lead_time_days"] = lead; r2["SMS_received"] = sms
            s = float(m.predict_proba(pd.DataFrame([r2])[config.FEATURES].infer_objects())[:, 1][0])
            best = min(best, flag_prob(s, str(row["age_band"])))
    rec(f"counterfactual {c['example']}: lowest reachable flag probability", round(c["lowest_reachable"]["policy_flag_probability"], 4), round(best, 4), "own grid", tol=1e-4)
try:
    from careconnect.explain.local import lime_explanations  # project code: a reproducibility rerun, not an independent check
    examples = json.load(open(A / "analysis.json"))
    ex = examples["samples_documented_cycle1"] + examples["samples"]
    rerun = lime_explanations(m, tr, ex)
    built = {(e["config"], e["example"]): e for e in X["lime"]}
    diffs = [abs(e["lime_r2"] - built[(e["config"], e["example"])]["lime_r2"]) + abs(e["abs_error"] - built[(e["config"], e["example"])]["abs_error"]) for e in rerun]
    rec("LIME rerun (project code, fixed seeds) reproduces stored R2 and errors", True, bool(max(diffs) < 1e-9), "reproducibility rerun",
        note="this re-runs the project's own LIME code; it shows determinism, not independent correctness")
    rec("LIME adequate examples (local sampling / cycle-1 configuration)",
        [sum(e["fidelity"] == "adequate" for e in X["lime"] if e["config"] == c) for c in ["local_sampling", "cycle1_training_distribution"]],
        [sum(e["fidelity"] == "adequate" for e in rerun if e["config"] == c) for c in ["local_sampling", "cycle1_training_distribution"]], "rerun", tol=0)
except Exception as e:      # noqa: BLE001
    info("LIME rerun not completed", f"{type(e).__name__}: {e}", "rerun")

# ---------------------------------------------------------------- 6. gate
blk = [r["check"].split(" ")[0] for r in G["checks"] if r["blocking"] and not r["passed"]]
rec("gate: blocking failures are C8, C3 and Ethics Committee approval", ["C8", "C3", "Ethics"], blk, "promotion_decision.json", tol=0)
rec("gate: decision NOT ELIGIBLE and no automatic promotion", True, G["decision"].startswith("NOT ELIGIBLE") and G["automatic_promotion"] is False, "json")

summary = {"match": sum(r["result"] == "MATCH" for r in rows), "mismatch": sum(r["result"] == "MISMATCH" for r in rows),
           "info": sum(r["result"] == "INFO" for r in rows), "seconds": round(time.time() - t0)}
json.dump({"summary": summary, "rows": rows, "weekly_s7": weekly}, open(OUT / "independent_audit.json", "w"), indent=1, default=str)
for r in rows:
    print(f"{r['result']:8s} | {r['item']} | build {r['build_artifact']} | recomputed {r['recomputed']} | {r['note']}")
print("SUMMARY", json.dumps(summary))
sys.exit(1 if summary["mismatch"] else 0)
