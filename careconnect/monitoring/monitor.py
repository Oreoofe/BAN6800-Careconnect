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
* bias drift     - equalized-odds gap by age group under the adopted policy; warn > 0.04, halt > 0.05
Concept drift (a changed relationship between inputs and outcomes) shows up as falling performance
with little data drift; that pattern is reported separately.

Also writes an Evidently data-drift report (HTML) for the latest replayed week when Evidently is installed.
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
    try:                                    # Evidently >= 0.7 API
        from evidently import Report
        from evidently.presets import DataDriftPreset
        snap = Report([DataDriftPreset()]).run(cur[cols], ref[cols])
        snap.save_html(str(path))
        return "evidently>=0.7"
    except ImportError:
        pass
    try:                                    # Evidently 0.4-0.6 API
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
        rep = Report(metrics=[DataDriftPreset()])
        rep.run(reference_data=ref[cols], current_data=cur[cols])
        rep.save_html(str(path))
        return "evidently<0.7"
    except Exception as e:  # pragma: no cover
        return f"skipped: {e}"


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
    weeks = []
    for w, cur in te.groupby("week"):
        if len(cur) < 500:
            continue
        weeks.append(batch_report(f"week of {w}", cur, ref, m, policy, T, val_auc, ref_scores))
    stress = te.sample(4000, random_state=1).copy()
    stress["lead_time_days"] = (stress["lead_time_days"] * 1.5 + 7).round().astype(int)
    stress_r = batch_report("STRESS TEST (synthetic: lead times x1.5 + 7 days)", stress, ref, m, policy, T, val_auc, ref_scores)
    stress_r["synthetic"] = True
    latest = te[te["week"] == te["week"].max()] if (te["week"] == te["week"].max()).sum() >= 500 else te
    ev = evidently_report(ref, latest, OUT / "evidently_drift_report.html")
    res = {"generated_utc": datetime.now(timezone.utc).isoformat(), "model_version": config.MODEL_VERSION,
           "mode": "replay of held-out appointments by week (not live production data)", "thresholds": TH,
           "validation_auc": val_auc, "policy": policy["name"] if policy else "single cut-off",
           "weeks": weeks, "stress_test": stress_r, "evidently": ev}
    json.dump(res, open(OUT / "monitoring.json", "w"), indent=1, default=float)
    for r in weeks + [stress_r]:
        print(r["batch"], r["n"], round(r["auc"] or 0, 3), round(r["max_feature_psi"], 3), round(r["age_eod"], 3), r["status"])
    print("evidently:", ev)


if __name__ == "__main__":
    main()
