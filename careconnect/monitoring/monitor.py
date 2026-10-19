"""Model monitoring: data drift, prediction drift, performance and fairness by week.

    python -m careconnect.monitoring.monitor

What this run is: a REPLAY. The held-out test appointments are grouped by appointment week and
scored as if each week were a new production batch, then compared with the training data. Real
outcome labels are available for every replayed week, so performance and fairness can be tracked
too. One clearly labelled stress-test batch (lead times shifted) shows that the alert logic fires.
It is not live production monitoring: CareConnect is not deployed on hospital data.

Signals and thresholds (Module 4 monitoring hooks):
* data drift     - Population Stability Index per feature; warn >= 0.10, alert >= 0.20 (retrain trigger)
* prediction drift - PSI of the risk score
* performance    - ROC-AUC and recall per week; alert if AUC falls > 0.03 below validation
* bias drift     - equalized-odds gap by age group under the configured (not approved) policy; cycle-1 rule: warn > 0.04, halt > 0.05
Concept drift (a changed relationship between inputs and outcomes) shows up as falling performance
with little data drift; that pattern is reported separately.

Also writes an Evidently data-drift report (HTML) for the last 28 days when Evidently is installed.

Cycle 2 (after verification): the weekly rule halted every week because weekly age groups are too small
to measure a 0.05 gap. The replay now also applies a rule with minimum group sizes and a bootstrap 95%
interval, on rolling 28-day windows and calendar months (see FAIR_RULE). The cycle-1 rule and its
results are kept unchanged alongside, so the change is auditable.
"""
import json
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score

from careconnect import config
from careconnect.fairness.metrics import fairness_report
from careconnect.fairness.policy import apply_policy, load_policy

warnings.filterwarnings("ignore")
OUT = config.ARTIFACTS / "monitoring"
# appointment_dow is excluded: a partial week (e.g. one holiday-shortened week) changes the weekday mix by
# construction, which is calendar composition rather than drift.
DRIFT_FEATURES = ["Age", "lead_time_days", "prior_appointments", "prior_noShow_rate", "SMS_received", "Scholarship",
                  "Gender", "age_band", "Neighbourhood"]
MIN_POS = 30   # minimum no-shows per group before its error rates enter the weekly fairness gap
TH = {"psi_warn": 0.10, "psi_alert": 0.20, "auc_drop_alert": 0.03, "eod_warn": 0.04, "eod_halt": 0.05}


def psi(ref, cur, bins=10):
    """Population Stability Index. Numeric: reference-quantile bins; categorical: category shares."""
    ref, cur = pd.Series(ref), pd.Series(cur)
    if not pd.api.types.is_numeric_dtype(ref) or ref.nunique() <= 12:
        cats = ref.astype(str).value_counts(normalize=True)
        cs = cur.astype(str).value_counts(normalize=True).reindex(cats.index, fill_value=0)
        r, c = cats.values, cs.values
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref)
        c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def level(v, warn, alert):
    return "alert" if v >= alert else ("warn" if v >= warn else "ok")


def batch_report(name, cur, ref, model, policy, T, val_auc, ref_scores):
    X = cur[config.FEATURES]
    y = cur[config.TARGET].values
    s = model.predict_proba(X)[:, 1]
    if policy:
        _, yh = apply_policy(s, cur["age_band"].astype(str).values, policy)
    else:
        yh = (s >= T).astype(int)
    drift = {f: psi(ref[f], cur[f]) for f in DRIFT_FEATURES}
    g = cur["age_band"].astype(str).values
    pos = pd.Series(y).groupby(g).sum()
    ok = pos[pos >= MIN_POS].index           # too few no-shows in a group makes its error rates noise
    k = np.isin(g, ok)
    fr = fairness_report(y[k], yh[k], g[k])
    auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else None
    r = {"batch": name, "n": len(cur), "no_show_rate": float(y.mean()), "mean_score": float(s.mean()),
         "flag_rate": float(yh.mean()), "auc": auc, "recall": float(recall_score(y, yh)),
         "score_psi": psi(ref_scores, s), "feature_psi": drift, "max_feature_psi": max(drift.values()),
         "age_eod": fr["eod"], "age_dpd": fr["dpd"], "fairness_groups_scored": list(map(str, ok)),
         "fairness_groups_skipped_small": [str(x) for x in pos.index if x not in ok]}
    r["status"] = {
        "data_drift": level(r["max_feature_psi"], TH["psi_warn"], TH["psi_alert"]),
        "prediction_drift": level(r["score_psi"], TH["psi_warn"], TH["psi_alert"]),
        "performance": "alert" if auc is not None and auc < val_auc - TH["auc_drop_alert"] else "ok",
        "bias_drift": "halt" if r["age_eod"] > TH["eod_halt"] else ("warn" if r["age_eod"] > TH["eod_warn"] else "ok")}
    r["concept_drift_suspected"] = r["status"]["performance"] == "alert" and r["status"]["data_drift"] == "ok"
    r["actions"] = [a for a, cond in [
        ("Retraining review (PSI >= 0.20)", r["status"]["data_drift"] == "alert"),
        ("Investigate score shift", r["status"]["prediction_drift"] != "ok"),
        ("Performance review with Data Science lead", r["status"]["performance"] == "alert"),
        ("Fairness halt: suspend flagging, notify Ethics Committee within 24h", r["status"]["bias_drift"] == "halt"),
        ("Fairness warning: review at next monthly audit", r["status"]["bias_drift"] == "warn")] if cond]
    return r


def evidently_report(ref, cur, path):
    cols = [c for c in DRIFT_FEATURES]
    # plain object/str dtypes: Evidently's type inference is not tested against pandas 3 string dtypes
    ref = ref[cols].copy()
    cur = cur[cols].copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(ref[c]):
            ref[c] = ref[c].astype(object)
            cur[c] = cur[c].astype(object)
    try:                                    # Evidently >= 0.7 API
        from evidently import Report
        from evidently.presets import DataDriftPreset
        snap = Report([DataDriftPreset()]).run(cur, ref)
        snap.save_html(str(path))
        return "evidently>=0.7"
    except ImportError:
        pass
    except Exception as e:                  # report the failure instead of crashing the monitoring run
        return f"failed: {type(e).__name__}: {str(e)[:200]}"
    try:                                    # Evidently 0.4-0.6 API
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
        rep = Report(metrics=[DataDriftPreset()])
        rep.run(reference_data=ref[cols], current_data=cur[cols])
        rep.save_html(str(path))
        return "evidently<0.7"
    except Exception as e:  # pragma: no cover
        return f"skipped: {e}"


# ---------------------------------------------------------------------------------------------------
# Cycle-2 fairness rule. The cycle-1 weekly rule (above, kept unchanged in batch_report) halted in all 7
# replayed weeks: weekly age groups are small, so their error-rate gaps are mostly sampling noise.
FAIR_RULE = {"window": "rolling 28 days, evaluated weekly", "min_no_shows_per_group": 50,
             "min_attended_per_group": 50, "bootstrap": 300, "limit": 0.05,
             "statuses": {"ok": "95% interval for the gap lies wholly at or below the limit",
                          "halt": "95% interval lies wholly above the limit: suspend flagging, notify Ethics Committee",
                          "inconclusive": "interval straddles the limit: no conclusion; review at monthly audit, extend window",
                          "insufficient data": "fewer than 2 groups meet the minimum counts: no fairness conclusion is drawn"}}


def fairness_rule(y, yh, groups, seed=0, rule=FAIR_RULE):
    """Decide ok / halt / inconclusive / insufficient data. Groups below the minimum counts are listed as
    'not evaluated' and are NEVER counted as passing; insufficient evidence is not evidence of fairness."""
    y, yh, groups = np.asarray(y), np.asarray(yh), np.asarray(groups)
    counts = {}
    for v in np.unique(groups):
        k = groups == v
        counts[str(v)] = {"n": int(k.sum()), "no_shows": int(y[k].sum()), "attended": int((y[k] == 0).sum())}
    elig = [v for v, c in counts.items() if c["no_shows"] >= rule["min_no_shows_per_group"]
            and c["attended"] >= rule["min_attended_per_group"]]
    skipped = {v: c for v, c in counts.items() if v not in elig}
    out = {"groups_evaluated": elig, "groups_not_evaluated": skipped, "counts": counts}
    if len(elig) < 2:
        return {**out, "status": "insufficient data", "eod": None, "ci95": None}
    k = np.isin(groups, elig)
    yy, hh, gg = y[k], yh[k], groups[k]
    point = fairness_report(yy, hh, gg)["eod"]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(rule["bootstrap"]):
        i = rng.integers(0, len(yy), len(yy))
        try:
            bs.append(fairness_report(yy[i], hh[i], gg[i])["eod"])
        except ValueError:
            continue
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    status = "halt" if lo > rule["limit"] else ("ok" if hi <= rule["limit"] else "inconclusive")
    return {**out, "status": status, "eod": point, "ci95": [lo, hi],
            "note": ("not all groups evaluated" if skipped else "all groups evaluated")}


def explain_status(fr, exp_eod, drift_status, limit=0.05):
    """Plain reason for a window's fairness status, and what the evidence attributes it to."""
    cnt = ", ".join(f"{g} {c['no_shows']} no-shows/{c['attended']} attended" for g, c in fr["groups_not_evaluated"].items())
    if fr["status"] == "insufficient data":
        return (f"Fewer than two groups reach {FAIR_RULE['min_no_shows_per_group']} no-shows and {FAIR_RULE['min_attended_per_group']} "
                f"attended visits ({cnt}). No fairness conclusion is drawn; this is not a pass."), "insufficient sample"
    lo, hi = fr["ci95"]
    base = f"Realised gap {fr['eod']:.3f}, bootstrap 95% CI {lo:.3f}-{hi:.3f} "
    if fr["status"] == "halt":
        base += "lies wholly above 0.05."
    elif fr["status"] == "inconclusive":
        base += "straddles 0.05."
    else:
        base += "lies wholly at or below 0.05."
    if exp_eod is not None and exp_eod > limit:
        attr = "policy's expected disparity (expected gap above the limit)"
        base += f" Expected gap {exp_eod:.3f} is itself above 0.05, so this is not just the random draw."
    elif fr["eod"] > limit:
        attr = "random draw / sampling (expected gap within the limit)"
        base += f" Expected gap {exp_eod:.3f} is within 0.05: the excess comes from this draw or sampling noise."
    else:
        attr = "within limit"
    if fr["groups_not_evaluated"]:
        base += f" Not evaluated (too few cases, not counted as passing): {cnt}."
    if drift_status != "ok":
        base += f" Data-drift status in this window: {drift_status}."
    return base, attr


def window_report(name, cur, ref, model, policy, T, val_auc, ref_scores, seed, coverage_days=None, override=None):
    X = cur[config.FEATURES]
    y = cur[config.TARGET].values
    s = model.predict_proba(X)[:, 1]
    g = cur["age_band"].astype(str).values
    if policy:
        pmf, yh = apply_policy(s, g, policy, seed=seed)       # realised decisions, as production would see them
    else:
        pmf, yh = (s >= T).astype(float), (s >= T).astype(int)
    if override is not None:
        yh = override(yh, cur)
    drift = {f: psi(ref[f], cur[f]) for f in DRIFT_FEATURES}
    auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else None
    fr = fairness_rule(y, yh, g, seed=seed)
    exp_eod = None
    if fr["status"] != "insufficient data":
        k = np.isin(g, fr["groups_evaluated"])
        exp_eod = max(max(pmf[k & (g == v) & (y == 1)].mean() for v in fr["groups_evaluated"]) -
                      min(pmf[k & (g == v) & (y == 1)].mean() for v in fr["groups_evaluated"]),
                      max(pmf[k & (g == v) & (y == 0)].mean() for v in fr["groups_evaluated"]) -
                      min(pmf[k & (g == v) & (y == 0)].mean() for v in fr["groups_evaluated"]))
    r = {"batch": name, "n": len(cur), "coverage_days": coverage_days, "no_show_rate": float(y.mean()),
         "flag_rate": float(yh.mean()), "auc": auc, "recall": float(recall_score(y, yh)) if y.sum() else None,
         "score_psi": psi(ref_scores, s), "feature_psi": drift, "max_feature_psi": max(drift.values()),
         "max_psi_feature": max(drift, key=drift.get), "fairness": fr, "expected_age_eod_evaluated_groups": exp_eod}
    r["status"] = {"data_drift": level(r["max_feature_psi"], TH["psi_warn"], TH["psi_alert"]),
                   "prediction_drift": level(r["score_psi"], TH["psi_warn"], TH["psi_alert"]),
                   "performance": "alert" if auc is not None and auc < val_auc - TH["auc_drop_alert"] else "ok",
                   "bias_drift": fr["status"]}
    r["start"] = str(cur["AppointmentDay"].min().date())
    r["end"] = str(cur["AppointmentDay"].max().date())
    r["fairness"]["reason"], r["fairness"]["attribution"] = explain_status(fr, exp_eod, r["status"]["data_drift"])
    if override is not None:
        r["fairness"]["attribution"] = "synthetic stress perturbation (decisions altered after the policy)"
        r["fairness"]["reason"] += " Decisions were altered synthetically for this stress test."
    r["fairness"]["ci_method"] = f"percentile bootstrap, {FAIR_RULE['bootstrap']} resamples of the window's appointments, realised decisions"
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = joblib.load(config.ARTIFACTS / "xgboost.joblib")
    tr = pd.read_pickle(config.ARTIFACTS / "train_frame.pkl")
    te = pd.read_pickle(config.ARTIFACTS / "test_frame.pkl")
    R = json.load(open(config.ARTIFACTS / "model_results.json"))
    T = json.load(open(config.ARTIFACTS / "operating_point.json"))["flag_threshold"]
    policy = load_policy()
    val_auc = R["models"]["xgboost"]["cv_auc"]
    ref = tr.sample(20000, random_state=config.RANDOM_STATE)
    ref_scores = m.predict_proba(ref[config.FEATURES])[:, 1]
    te = te.copy()
    te["week"] = te["AppointmentDay"].dt.to_period("W-SUN").dt.start_time.dt.strftime("%Y-%m-%d")

    # 1) cycle-1 method, unchanged (weekly batches, point estimate, halt > 0.05) - preserved for comparison
    weeks = [batch_report(f"week of {w}", cur, ref, m, policy, T, val_auc, ref_scores)
             for w, cur in te.groupby("week") if len(cur) >= 500]
    # 2) cycle-2 rule applied to the same weekly batches (shows how it behaves with small groups)
    weeks_new = [window_report(f"week of {w}", cur, ref, m, policy, T, val_auc, ref_scores, seed=i)
                 for i, (w, cur) in enumerate(te.groupby("week")) if len(cur) >= 500]
    # 3) cycle-2 rule on rolling 28-day windows stepped weekly
    start, end = te["AppointmentDay"].min(), te["AppointmentDay"].max()
    ends = sorted(set(pd.date_range(start + pd.Timedelta(days=6), end, freq="7D").tolist() + [end]))
    rolling = []
    for i, e in enumerate(ends):
        cur = te[(te["AppointmentDay"] > e - pd.Timedelta(days=28)) & (te["AppointmentDay"] <= e)]
        cov = int((e - max(start, e - pd.Timedelta(days=27))).days) + 1
        rolling.append(window_report(f"28 days to {e.date()}" + ("" if cov >= 28 else f" (partial: {cov} days)"),
                                     cur, ref, m, policy, T, val_auc, ref_scores, seed=100 + i, coverage_days=cov))
    # 4) calendar months
    months = [window_report(f"calendar month {mth}", cur, ref, m, policy, T, val_auc, ref_scores, seed=200 + i,
                            coverage_days=int(cur["AppointmentDay"].dt.normalize().nunique()))
              for i, (mth, cur) in enumerate(te.groupby(te["AppointmentDay"].dt.strftime("%Y-%m")))]
    # 5) stress tests (synthetic, labelled)
    last = te[te["AppointmentDay"] > end - pd.Timedelta(days=28)]
    lead = te.sample(4000, random_state=1).copy()
    lead["lead_time_days"] = (lead["lead_time_days"] * 1.5 + 7).round().astype(int)

    def drop_older_flags(yh, cur):
        out = yh.copy()
        k = cur["age_band"].isin(["senior", "elderly"]).values & (yh == 1)
        out[k & (np.random.default_rng(7).random(len(yh)) < 0.6)] = 0
        return out
    small = pd.concat([last[last["age_band"] != "elderly"], last[last["age_band"] == "elderly"].sample(25, random_state=3)])
    stress = [
        {"name": "S1 data drift: lead times x1.5 + 7 days", "expect": "data/prediction drift alerts",
         **window_report("S1", lead, ref, m, policy, T, val_auc, ref_scores, seed=301)},
        {"name": "S2 bias drift: 60% of senior/elderly flags removed (last 28 days)", "expect": "fairness halt",
         **window_report("S2", last, ref, m, policy, T, val_auc, ref_scores, seed=302, override=drop_older_flags)},
        {"name": "S3 small group: elderly reduced to 25 appointments (last 28 days)", "expect": "elderly not evaluated",
         **window_report("S3", small, ref, m, policy, T, val_auc, ref_scores, seed=303)},
        {"name": "S4 tiny batch: 300 random appointments", "expect": "insufficient data",
         **window_report("S4", te.sample(300, random_state=4), ref, m, policy, T, val_auc, ref_scores, seed=304)}]
    for st in stress:
        st["synthetic"] = True
    ev = evidently_report(ref, last, OUT / "evidently_drift_report.html")

    def summarise(rows):
        out = {"halt": [], "inconclusive": [], "ok": [], "insufficient data": []}
        for r in rows:
            out[r["fairness"]["status"]].append(r["batch"])
        exp = [r["expected_age_eod_evaluated_groups"] for r in rows if r["expected_age_eod_evaluated_groups"] is not None]
        return {"counts": {k: len(v) for k, v in out.items()}, "windows": out,
                "windows_with_expected_gap_above_0.05": sum(e > 0.05 for e in exp),
                "windows_with_expected_gap_evaluable": len(exp),
                "expected_gap_range": [min(exp), max(exp)] if exp else None,
                "windows_with_any_group_not_evaluated": [r["batch"] for r in rows if r["fairness"]["groups_not_evaluated"]]}
    summary = {"rolling_28d_all": summarise(rolling),
               "rolling_28d_full_windows_only": summarise([r for r in rolling if (r["coverage_days"] or 0) >= 28]),
               "calendar_months": summarise(months), "weeks_cycle2_rule": summarise(weeks_new),
               "weeks_cycle1_rule_halts": sum(w["status"]["bias_drift"] == "halt" for w in weeks),
               "reading": ("'insufficient data' and groups 'not evaluated' are reported separately and are never counted as a pass; "
                           "'inconclusive' means the 95% interval straddles 0.05 and supports no conclusion either way.")}
    res = {"generated_utc": datetime.now(timezone.utc).isoformat(), "model_version": config.MODEL_VERSION,
           "mode": "replay of held-out appointments (not live production data)", "thresholds": TH,
           "fairness_rule": FAIR_RULE, "validation_auc": val_auc, "policy": policy["name"] if policy else "single cut-off",
           "weeks_cycle1_rule": weeks, "weeks_cycle2_rule": weeks_new, "rolling_28d": rolling, "calendar_months": months,
           "stress_tests": stress, "summary": summary, "evidently": ev, "evidently_window": "last 28 days of test appointments vs training sample",
           "weeks": weeks, "stress_test": stress[0]}       # cycle-1 keys kept for backward compatibility
    json.dump(res, open(OUT / "monitoring.json", "w"), indent=1, default=float)
    for label, rows in [("cycle1 weekly", weeks)]:
        for r in rows:
            print(label, r["batch"], round(r["age_eod"], 3), r["status"]["bias_drift"])
    for label, rows in [("cycle2 weekly", weeks_new), ("rolling28", rolling), ("month", months), ("stress", stress)]:
        for r in rows:
            f = r["fairness"]
            print(label, r["batch"], r["n"], f["status"], None if f["eod"] is None else round(f["eod"], 3),
                  f["ci95"] and [round(x, 3) for x in f["ci95"]], list(f["groups_not_evaluated"]), r["status"]["data_drift"])
    print("evidently:", ev)


if __name__ == "__main__":
    main()
