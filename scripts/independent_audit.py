"""Independent audit of the Cycle 5 evidence (commit 27943a1, build-outputs-14).

Written separately from the project's own verifier. It does NOT import careconnect.fairness, careconnect.monitoring,
careconnect.modeling.evaluate or scripts/verify_results.py. The only project code it reuses is the data pipeline
(to rebuild the modelling table and split), and it cross-checks that table against its own re-derivation from the raw CSV.

Inputs: the raw CSV (downloaded by the pipeline), the published model / policy / JSON artifacts from build-outputs-14
(directory given as argv[1]). Output: audit_report.json + printed table.
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

PUB = Path(sys.argv[1])          # build-outputs-14 checkout
A = PUB / "artifacts"
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
J = lambda p: json.load(open(A / p))
R, OP, PC, MON, X, AN, G, POLICY = (J("model_results.json"), J("operating_point.json"), J("policy_comparison.json"),
                                    J("monitoring/monitoring.json"), J("explanations.json"), J("analysis.json"),
                                    J("promotion_decision.json"), J("fairness_policy.json"))
rows = []


def rec(item, reported, recomputed, source, tol=5e-4, note=""):
    if isinstance(reported, (int, float)) and isinstance(recomputed, (int, float)) and not isinstance(reported, bool):
        ok = abs(float(reported) - float(recomputed)) <= tol
    elif isinstance(reported, tuple) and isinstance(recomputed, tuple) and len(reported) == len(recomputed) and tol > 0:
        ok = all(abs(float(a) - float(b)) <= tol for a, b in zip(reported, recomputed))
    else:
        ok = reported == recomputed
    rows.append({"item": item, "reported": reported, "recomputed": recomputed, "source": source,
                 "result": "MATCH" if ok else "MISMATCH", "note": note})


sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

# ------------------------------------------------------------------ 1. raw data and checksums
from careconnect import config  # noqa: E402
raw = config.RAW_CSV
rec("dataset sha256", "9132d3e7d0246617df9041d3764f20ad6f08e7b0d9f0997fa254fc5e52eda27d", sha(raw), "raw CSV downloaded in this run")
rec("model sha256", "6d60bea7544ffac64049a66af98f4d133508cb77aa464702a91075706127a385", sha(A / "xgboost.joblib"), "build-outputs-14")
rec("policy sha256", "e18da7cd12d13fc431a730af9f2afe6cbfa75e7de7928f19ded79a7364e11e1f", sha(A / "fairness_policy.json"), "build-outputs-14")

# ------------------------------------------------------------------ 2. independent cleaning + features from raw
df = pd.read_csv(raw)
rec("raw rows", 110527, len(df), "raw CSV")
d = df.drop_duplicates("AppointmentID")
bad_age = int((~d["Age"].between(0, 100)).sum()); d = d[d["Age"].between(0, 100)]
sd = pd.to_datetime(d["ScheduledDay"], utc=True).dt.tz_localize(None)
ad = pd.to_datetime(d["AppointmentDay"], utc=True).dt.tz_localize(None)
keep = sd.dt.normalize() <= ad
bad_sched = int((~keep).sum()); d = d[keep].copy(); sd, ad = sd[keep], ad[keep]
rec("invalid ages removed", 8, bad_age, "own cleaning of raw CSV")
rec("bookings after appointment removed", 5, bad_sched, "own cleaning of raw CSV")
rec("rows after cleaning", 110514, len(d), "own cleaning of raw CSV")
d["y"] = (d["No-show"] == "Yes").astype(int)
d["lead"] = (ad - sd.dt.normalize()).dt.days
d["appt"] = ad.dt.normalize(); d["book"] = sd.dt.normalize()
# prior history by brute force per patient (outcome known before the booking day)
pa, pn = np.zeros(len(d), int), np.zeros(len(d), int)
d = d.reset_index(drop=True)
for _, g in d.groupby("PatientId"):
    if len(g) == 1:
        continue
    ap, yy = g["appt"].values, g["y"].values
    for i, b in zip(g.index, g["book"].values):
        m = ap < b
        pa[i], pn[i] = int(m.sum()), int(yy[m].sum())
d["pa"], d["pn"] = pa, pn
rec("rows with outcome-known history", 30878, int((pa > 0).sum()), "own brute-force history from raw CSV")

# ------------------------------------------------------------------ 3. project table + split (pipeline code) cross-checked
from careconnect.modeling.data import load_table, split  # noqa: E402
tab = load_table()
rec("modelling table rows", 110514, len(tab), "pipeline output")
own = d[["lead", "pa", "pn", "y"]].reset_index(drop=True)
same = (len(own) == len(tab) and (own["lead"].values == tab["lead_time_days"].values).all()
        and (own["pa"].values == tab["prior_appointments"].values).all() and (own["pn"].values == tab["prior_noShows"].values).all()
        and (own["y"].values == tab["noshow_flag"].values).all())
rec("pipeline lead time and prior-history columns equal own re-derivation (all rows)", True, bool(same), "row-by-row comparison")
tr, te = split(tab)
_, own_te = train_test_split(np.arange(len(d)), test_size=0.2, stratify=d["y"].values, random_state=42)
rec("own stratified split (seed 42) selects the same test rows", True, bool(sorted(own_te.tolist()) == sorted(te.index.tolist())), "own sklearn split")
rec("train rows", 88411, len(tr), "split"); rec("test rows", 22103, len(te), "split")
rec("test no-show rate", 0.2019, round(float(te[config.TARGET].mean()), 4), "split", tol=1e-4)
y = te[config.TARGET].astype(int).values

# ------------------------------------------------------------------ 4. model metrics at the single cut-off
m = joblib.load(A / "xgboost.joblib")
p = m.predict_proba(te[config.FEATURES])[:, 1]
T, H = OP["flag_threshold"], OP["high_threshold"]
yh = (p >= T).astype(int)
tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
x = R["models"]["xgboost"]["at_recall_target"]
rec("cut-off", 0.25, T, "operating_point.json", tol=0)
rec("ROC-AUC", 0.7455, round(roc_auc_score(y, p), 4), "sklearn on test set", tol=1e-4)
rec("recall", 0.656, recall_score(y, yh), "sklearn"); rec("precision", 0.343, precision_score(y, yh), "sklearn")
rec("F1 single cut-off", 0.450, f1_score(y, yh), "sklearn"); rec("flag rate", 0.386, yh.mean(), "own")
rec("confusion TN/FP/FN/TP", "12031/5610/1536/2926", f"{tn}/{fp}/{fn}/{tp}", "sklearn")
tier = np.where(p >= H, "high", np.where(p >= T, "medium", "low"))
for t, (sh, rate) in {"high": (0.089, 0.464), "medium": (0.298, 0.307), "low": (0.614, 0.113)}.items():
    rec(f"tier {t} share", sh, (tier == t).mean(), "own"); rec(f"tier {t} no-show rate", rate, y[tier == t].mean(), "own")
for name, a in [("logistic_regression", (0.680, 0.652, 0.305, 0.416)), ("random_forest", (0.743, 0.649, 0.345, 0.450))]:
    v = R["models"][name]["at_recall_target"]
    rec(f"{name} AUC/recall/precision/F1 (report table vs model_results.json)", a,
        tuple(round(v[k], 3) for k in ["roc_auc", "recall", "precision", "f1"]), "model_results.json (not retrained here)", tol=0,
        note="baseline models are not published; values taken from the build's JSON, not recomputed")


# ------------------------------------------------------------------ 5. policies (own implementations)
def grp_rates(yv, pred, g):
    out = {}
    for v in np.unique(g):
        k = g == v
        out[str(v)] = {"n": int(k.sum()), "no_shows": int(yv[k].sum()), "sel": pred[k].mean(),
                       "tpr": pred[k & (yv == 1)].mean(), "fpr": pred[k & (yv == 0)].mean()}
    return out


def gaps(yv, pred, g):
    r = grp_rates(yv, pred, g)
    t = [z["tpr"] for z in r.values()]; f = [z["fpr"] for z in r.values()]; s = [z["sel"] for z in r.values()]
    return {"eod": max(max(t) - min(t), max(f) - min(f)), "tpr_gap": max(t) - min(t), "fpr_gap": max(f) - min(f),
            "dpd": max(s) - min(s), "di": min(s) / max(s), "groups": r}


def summary(yv, pred):
    tpe, fl = (pred * yv).sum(), pred.sum()
    rc, pr = tpe / yv.sum(), tpe / fl
    return {"recall": rc, "precision": pr, "f1": 2 * pr * rc / (pr + rc), "flag_rate": pred.mean()}


att = {a: te[a].astype(str).values for a in ["age_band", "Gender", "Scholarship"]}


def flag_prob(score, grp):
    r = POLICY["groups"][grp]
    inner = r["p0"] * (score > r["op0"][1] if r["op0"][0] == ">" else score < r["op0"][1]) + \
        r["p1"] * (score > r["op1"][1] if r["op1"][0] == ">" else score < r["op1"][1])
    return r["p_ignore"] * r["prediction_constant"] + (1 - r["p_ignore"]) * inner


pmf = np.array([flag_prob(s, g) for s, g in zip(p, att["age_band"])])
PB = PC["policies"]["P4_option_B_equalized_odds"]
sB = summary(y, pmf)
for k, rv in [("recall", 0.852), ("precision", 0.283), ("f1", 0.425), ("flag_rate", 0.607)]:
    rec(f"Option B expected {k}", rv, sB[k], "own flag probabilities from fairness_policy.json",
        note="expected-value metrics: precision = E[TP]/E[flags] (ratio of expectations)")
rec("Option B share of appointments with 0 < flag probability < 1 ('81.6% randomised')", 0.816, ((pmf > 0) & (pmf < 1)).mean(), "own",
    note="definition: share of APPOINTMENTS whose decision is a random draw; not a share of flags")
for a, rv in [("age_band", 0.041), ("Gender", 0.031), ("Scholarship", 0.029)]:
    rec(f"Option B expected EOD {a}", rv, gaps(y, pmf, att[a])["eod"], "own")
gB = gaps(y, pmf, att["age_band"])
rec("Option B expected age DPD", 0.045, gB["dpd"], "own"); rec("Option B expected age DI", 0.929, gB["di"], "own")
floor = np.array([POLICY["groups"][g]["p_ignore"] * POLICY["groups"][g]["prediction_constant"] for g in att["age_band"]])
rec("risk-unrelated share of Option B expected flags ('10.8%')", 0.108, floor.sum() / pmf.sum(), "own", tol=5e-4)
rec("risk-unrelated flags per 100 appointments ('6.55')", 6.55, floor.mean() * 100, "own", tol=5e-3)
rec("share of appointments with non-zero flag probability ('91.8%')", 0.918, (pmf > 0).mean(), "own")
for g, rv in [("child", 0.113), ("young_adult", 0.098), ("teen", 0.079), ("adult", 0.058), ("elderly", 0.026), ("senior", 0.0)]:
    rec(f"floor flag probability {g}", rv, POLICY["groups"][g]["p_ignore"] * POLICY["groups"][g]["prediction_constant"], "fairness_policy.json")
lo_idx = {g: np.argmin(np.where(att["age_band"] == g, p, np.inf)) for g in POLICY["groups"]}
rec("lowest-score appointment in each group gets exactly the floor", True,
    all(abs(pmf[i] - POLICY["groups"][g]["p_ignore"] * POLICY["groups"][g]["prediction_constant"]) < 1e-12 for g, i in lo_idx.items()), "own")
# realised distribution: the project's seeds, and independent seeds
for label, seeds in [("project seeds 20000+s", [20000 + s for s in range(200)]), ("independent seeds 900000+s", [900000 + s for s in range(1000)])]:
    real = np.array([gaps(y, (np.random.default_rng(s).random(len(pmf)) < pmf).astype(float), att["age_band"])["eod"] for s in seeds])
    rec(f"Option B realised age EOD: share of draws > 0.05 ({label})", 0.385, float((real > 0.05).mean()), "own simulation",
        tol=0.0005 if "project" in label else 0.06)
    rec(f"Option B realised age EOD: 95th percentile ({label})", 0.068, float(np.percentile(real, 95)), "own simulation",
        tol=0.0005 if "project" in label else 0.01)
# CI definition check: bootstrap of the EXPECTED-value gap vs the published CI (one realisation, then bootstrap)
rng = np.random.default_rng(12345)
bs = []
for _ in range(300):
    i = rng.integers(0, len(y), len(y))
    bs.append(gaps(y[i], pmf[i], att["age_band"][i])["eod"])
rec("Option B age CI as published [lo, hi] vs bootstrap of the expected-value gap", tuple(round(v, 3) for v in PB["age_band"]["eod_bootstrap_95ci"]),
    (round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)), "own bootstrap (300) of expected values",
    note="published CI bootstraps ONE random realisation, so it is not an interval around the plotted expected 0.041")

# deterministic policies from the published cut-offs
for key, label, exp in [("P1_equal_opportunity_tpr85", "EO TPR 0.85", {"recall": 0.864, "precision": 0.300, "f1": 0.446, "flag_rate": 0.581,
                                                                      "age": 0.100, "Gender": 0.043, "Scholarship": 0.054}),
                        ("P1_equal_opportunity_tpr65", "EO TPR 0.65", {"recall": 0.664, "f1": 0.439, "age": 0.064}),
                        ("P1_equal_opportunity_tpr75", "EO TPR 0.75", {"recall": 0.768, "f1": 0.446, "age": 0.083}),
                        ("P2_minimax_per_group", "Minimax", {"recall": 0.782, "f1": 0.447, "age": 0.089})]:
    cut = pd.Series(att["age_band"]).map(PC["policies"][key]["cutoffs"]).values.astype(float)
    dec = (p >= cut).astype(float)
    s_ = summary(y, dec)
    for k, rv in exp.items():
        val = s_[k] if k in s_ else gaps(y, dec, att["age_band" if k == "age" else k])["eod"]
        rec(f"{label} {k}", rv, val, "own, from published per-group cut-offs")
    if key == "P1_equal_opportunity_tpr85":
        ga = gaps(y, dec, att["age_band"])
        rec("EO TPR 0.85 age TPR gap", 0.060, ga["tpr_gap"], "own"); rec("EO TPR 0.85 age FPR gap", 0.100, ga["fpr_gap"], "own")
        sizes = {g: (z["n"], z["no_shows"]) for a in att for g, z in grp_rates(y, dec, att[a]).items()}
        rec("EO TPR 0.85 subgroup sizes (n, no-shows)", {"adult": (6021, 1213), "child": (4247, 879), "elderly": (1074, 178), "senior": (4433, 702),
                                                         "teen": (1584, 406), "young_adult": (4744, 1084), "F": (14270, 2862), "M": (7833, 1600),
                                                         "0": (19942, 3964), "1": (2161, 498)}, sizes, "own counts", tol=0)
        # independent bootstrap CIs (own seed, 1000 resamples) - should be close to the published 300-resample CIs
        rng = np.random.default_rng(777); b = {a: [] for a in att}
        for _ in range(1000):
            i = rng.integers(0, len(y), len(y))
            for a in att:
                b[a].append(gaps(y[i], dec[i], att[a][i])["eod"])
        for a, pub in [("age_band", (0.076, 0.138)), ("Gender", (0.027, 0.057)), ("Scholarship", (0.028, 0.080))]:
            rec(f"EO TPR 0.85 {a} 95% CI (independent bootstrap, 1000)", pub, (round(float(np.percentile(b[a], 2.5)), 3), round(float(np.percentile(b[a], 97.5)), 3)),
                "own bootstrap", tol=0.012, note="different resamples: expect agreement to about +/-0.01, not exact")
P0 = gaps(y, yh.astype(float), att["age_band"])
rec("single cut-off age EOD", 0.629, P0["eod"], "own")

# ------------------------------------------------------------------ 6. Module 4 s.7 weekly rule (own implementation)
tw = te.copy(); tw["week"] = tw["AppointmentDay"].dt.to_period("W-SUN").dt.start_time.dt.strftime("%Y-%m-%d")
weeks = list(tw.groupby("week"))
stored = MON["weeks_cycle2_rule"]
rec("number of replayed weeks", 7, len(weeks), "own grouping")
wk_out = []
for i, ((w, cur), st) in enumerate(zip(weeks, stored)):
    pw = m.predict_proba(cur[config.FEATURES])[:, 1]
    gw = cur["age_band"].astype(str).values
    pm = np.array([flag_prob(s, g) for s, g in zip(pw, gw)])
    dec = (np.random.default_rng(i).random(len(pm)) < pm).astype(float)     # same seed scheme as the replay
    yv = cur[config.TARGET].astype(int).values
    res = {}
    for a in ["age_band", "Gender"]:
        gv = cur[a].astype(str).values
        ok = [v for v in np.unique(gv) if yv[gv == v].sum() > 0 and (yv[gv == v] == 0).sum() > 0]
        k = np.isin(gv, ok)
        gg = gaps(yv[k], dec[k], gv[k]); ge = gaps(yv[k], pm[k], gv[k])
        res[a] = (gg["eod"], ge["eod"], "halt" if gg["eod"] > 0.05 else "alert" if gg["eod"] > 0.04 else "ok")
    overall = "halt" if "halt" in (res["age_band"][2], res["Gender"][2]) else "alert" if "alert" in (res["age_band"][2], res["Gender"][2]) else "ok"
    sa = st["module4_rule"]["attributes"]
    for a in ["age_band", "Gender"]:
        rec(f"s.7 {cur['AppointmentDay'].min().date()}..{cur['AppointmentDay'].max().date()} {a} realised disparity", round(sa[a]["disparity"], 4), round(res[a][0], 4), "own replay", tol=1e-4)
        rec(f"s.7 {cur['AppointmentDay'].min().date()}..{cur['AppointmentDay'].max().date()} {a} expected disparity", round(sa[a]["expected_disparity"], 4), round(res[a][1], 4), "own", tol=1e-4)
        rec(f"s.7 {cur['AppointmentDay'].min().date()}..{cur['AppointmentDay'].max().date()} {a} status", sa[a]["status"], res[a][2], "own thresholds", tol=0)
    rec(f"s.7 {w} overall status", st["module4_rule"]["status"], overall, "own", tol=0)
    wk_out.append({"week": w, "start": str(cur["AppointmentDay"].min().date()), "end": str(cur["AppointmentDay"].max().date()),
                   "days": int(cur["AppointmentDay"].dt.normalize().nunique()), "n": len(cur),
                   "age": res["age_band"], "gender": res["Gender"], "overall": overall,
                   "min_group_no_shows": int(min(cur[cur['age_band'] == v][config.TARGET].sum() for v in cur['age_band'].unique()))})
# supplementary analyses: recount statuses from the stored intervals
for setname, exp in [("rolling_28d", {"halt": 3, "inconclusive": 3}), ("calendar_months", {"halt": 1, "inconclusive": 1, "insufficient data": 1}),
                     ("weeks_cycle2_rule", {"halt": 5, "inconclusive": 1, "insufficient data": 1})]:
    cnt = {}
    for r in MON[setname]:
        f = r["fairness"]
        s = "insufficient data" if f["ci95"] is None else ("halt" if f["ci95"][0] > 0.05 else "ok" if f["ci95"][1] <= 0.05 else "inconclusive")
        cnt[s] = cnt.get(s, 0) + 1
    rec(f"supplementary {setname} status counts (recomputed from stored CIs)", exp, cnt, "monitoring.json intervals", tol=0,
        note="interval values themselves come from the replay's bootstrap; only the status logic is recomputed")

# ------------------------------------------------------------------ 7. SHAP additivity and shares (own TreeSHAP call)
import xgboost as xgb  # noqa: E402
pre, clf = m.named_steps["pre"], m.named_steps["clf"]
idx = np.random.RandomState(config.RANDOM_STATE).choice(len(te), 4000, replace=False)
Xs = te[config.FEATURES].iloc[idx]
Z = pre.transform(Xs)
names = list(pre.get_feature_names_out())
contrib = clf.get_booster().predict(xgb.DMatrix(Z, feature_names=names), pred_contribs=True)[:, :-1]
margin_full = clf.get_booster().predict(xgb.DMatrix(Z, feature_names=names), pred_contribs=True).sum(1)
margin = clf.get_booster().predict(xgb.DMatrix(Z, feature_names=names), output_margin=True)
rec("SHAP additivity: max |sum(contributions incl. bias) - margin| < 1e-3", True, bool(np.max(np.abs(margin_full - margin)) < 1e-3), "own TreeSHAP call")
rec("SHAP: sigmoid(margin) equals predict_proba", True, bool(np.allclose(1 / (1 + np.exp(-margin)), m.predict_proba(Xs)[:, 1], atol=1e-5)), "own")
per = {}
for j, n in enumerate(names):
    base = n.split("__", 1)[1]
    f = next((c for c in sorted(config.FEATURES, key=len, reverse=True) if base == c or base.startswith(c + "_")), base)
    per[f] = per.get(f, 0) + contrib[:, j]
imp = {f: float(np.abs(v).mean()) for f, v in per.items()}
tot = sum(imp.values())
for f, rv in [("lead_time_days", 0.555), ("Age", 0.134), ("Neighbourhood", 0.075), ("SMS_received", 0.042)]:
    rec(f"SHAP share {f}", rv, imp[f] / tot, "own TreeSHAP on the same 4,000-row test sample", tol=6e-4)
rec("SHAP share Gender + Scholarship < 4%", True, (imp["Gender"] + imp["Scholarship"]) / tot < 0.04, "own")

# ------------------------------------------------------------------ 8. counterfactual floors for the three tier examples
for c in X["counterfactuals"]:
    row = te.iloc[c["test_row"]]
    best = 1.0
    for lead in sorted({v for v in [0, 1, 2, 3, 5, 7, 14, 21, 28] if v < int(row["lead_time_days"])} | {int(row["lead_time_days"])}):
        for sms in sorted({int(row["SMS_received"]), 1}):
            r2 = row.copy(); r2["lead_time_days"] = lead; r2["SMS_received"] = sms
            s = float(m.predict_proba(pd.DataFrame([r2])[config.FEATURES].infer_objects())[:, 1][0])
            best = min(best, flag_prob(s, str(row["age_band"])))
    rec(f"counterfactual {c['example']}: lowest reachable flag probability", round(c["lowest_reachable"]["policy_flag_probability"], 3), round(best, 3), "own grid search", tol=1e-3)

# ------------------------------------------------------------------ 9. gate
blk = [r["check"].split(" ")[0] for r in G["checks"] if r["blocking"] and not r["passed"]]
rec("blocking gate failures", ["C8", "C3", "Ethics"], blk, "promotion_decision.json", tol=0)

json.dump({"rows": rows, "weekly_s7_own": wk_out}, open(OUT / "audit_report.json", "w"), indent=1, default=str)
mism = [r for r in rows if r["result"] != "MATCH"]
for r in rows:
    print(f"{r['result']:8s} | {r['item']} | reported {r['reported']} | recomputed {r['recomputed'] if not isinstance(r['recomputed'], float) else round(r['recomputed'], 4)} | {r['note']}")
for w in wk_out:
    print("WEEK", w)
print(f"SUMMARY {len(rows) - len(mism)} match, {len(mism)} mismatch")
