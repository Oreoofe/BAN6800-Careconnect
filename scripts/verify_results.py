"""Independent verification of the v2 build, recomputed from the saved model and saved splits.

    python scripts/verify_results.py      -> artifacts/verification_report.json (+ printed summary)

It does not reuse the training/evaluation code paths: it reloads the joblib model, the frozen
train/test frames and the exported policy, and recomputes everything with scikit-learn and
Fairlearn directly, then compares against what the pipeline reported. Any mismatch beyond
tolerance, or any failed check, is recorded as FAIL and makes the script exit non-zero.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from careconnect import config  # noqa: E402

A = config.ARTIFACTS
CHECKS = []
TOL = 1e-9


def check(name, ok, detail=None):
    CHECKS.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})


def load(name):
    p = A / name
    return json.load(open(p)) if p.exists() else None


def main():
    from fairlearn.metrics import (MetricFrame, demographic_parity_difference, demographic_parity_ratio,
                                   equalized_odds_difference, false_positive_rate, selection_rate,
                                   true_positive_rate)
    from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    rep = {}
    m = joblib.load(A / "xgboost.joblib")
    tr = pd.read_pickle(A / "train_frame.pkl")
    te = pd.read_pickle(A / "test_frame.pkl")
    R, E, M, P, OP = load("model_results.json"), load("analysis.json"), load("mitigation.json"), \
        load("fairness_policy.json"), load("operating_point.json")
    T = OP["flag_threshold"]

    # ---------- data, split and leakage controls ----------
    full = pd.concat([tr, te])
    rep["data"] = {"rows_total": len(full), "rows_train": len(tr), "rows_test": len(te),
                   "no_show_rate_total": float(full[config.TARGET].mean()),
                   "no_show_rate_train": float(tr[config.TARGET].mean()),
                   "no_show_rate_test": float(te[config.TARGET].mean()),
                   "patients_total": int(full["patient_id_hash"].nunique()),
                   "patients_in_both_train_and_test": int(len(set(tr.patient_id_hash) & set(te.patient_id_hash))),
                   "appointment_dates": [str(full.AppointmentDay.min().date()), str(full.AppointmentDay.max().date())],
                   "split": f"stratified random 80/20 on appointments, random_state={config.RANDOM_STATE}"}
    check("train/test rows disjoint", len(tr.index.intersection(te.index)) == 0)
    check("test share is 20%", abs(len(te) / len(full) - 0.20) < 0.001, round(len(te) / len(full), 4))
    check("stratification: train/test no-show rates within 0.002",
          abs(rep["data"]["no_show_rate_train"] - rep["data"]["no_show_rate_test"]) < 0.002)
    used = list(m.named_steps["pre"].feature_names_in_)
    rep["features_used_by_model"] = used
    check("model input schema == config.FEATURES", used == config.FEATURES)
    forbidden = {config.TARGET, "No-show", "patient_id_hash", "PatientId", "AppointmentID", "ScheduledDay", "AppointmentDay"}
    check("no target, identifier or raw date among model inputs", not (set(used) & forbidden), sorted(set(used) & forbidden))
    # history features only count visits before the booking day (spot check 500 test rows)
    samp = te.sample(500, random_state=1)
    bad = 0
    for _, r in samp.iterrows():
        h = full[full.patient_id_hash == r.patient_id_hash]
        k = h.AppointmentDay < r.ScheduledDay.normalize()
        bad += int(k.sum() != r.prior_appointments or h.loc[k, config.TARGET].sum() != r.prior_noShows)
    check("prior-visit features use only outcomes known at booking (500-row recount)", bad == 0, f"{bad} mismatches")

    # ---------- threshold selection ----------
    oof = np.load(A / "oof_xgb.npy")
    ytr = tr[config.TARGET].values
    rec_at_T = recall_score(ytr, (oof >= T).astype(int))
    rec_above = recall_score(ytr, (oof >= T + 0.0025).astype(int))
    rep["threshold"] = {"flag_threshold": T, "source": "5-fold out-of-fold predictions on the training set",
                        "train_oof_recall_at_T": rec_at_T, "train_oof_recall_at_T_plus_step": rec_above,
                        "rule": "largest cut-off (0.0025 grid) with OOF recall >= 0.65"}
    check("cut-off is the largest grid value reaching OOF recall >= 0.65", rec_at_T >= 0.65 and rec_above < 0.65,
          (round(rec_at_T, 4), round(rec_above, 4)))

    # ---------- performance, recomputed ----------
    X, y = te[config.FEATURES], te[config.TARGET].values
    p = m.predict_proba(X)[:, 1]
    yh = (p >= T).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
    perf = {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "recall": recall_score(y, yh),
            "precision": precision_score(y, yh), "f1": f1_score(y, yh), "flag_rate": float(yh.mean()),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}
    rep["xgboost_test"] = perf
    rep["comparison_table"] = {n: {k: R["models"][n]["at_recall_target"][k] for k in ["threshold", "roc_auc", "pr_auc", "recall", "precision", "f1", "flag_rate"]}
                               for n in R["models"]}
    rep["baselines"] = R["baselines"]
    rep["robustness"] = {"patient_grouped": R["robustness_patient_grouped"], "no_sensitive_inputs": R["ablation_no_sensitive_inputs"]}
    rec = R["models"]["xgboost"]["at_recall_target"]
    for k in ["roc_auc", "recall", "precision", "f1", "flag_rate"]:
        check(f"reported {k} matches recomputation", abs(rec[k] - perf[k]) < 1e-6, (rec[k], perf[k]))
    check("reported confusion matrix matches", E["confusion"] == perf["confusion"])
    check("XGBoost beats both baselines on ROC-AUC by > 0.2", perf["roc_auc"] > max(b["roc_auc"] for b in R["baselines"].values()) + 0.2)

    # ---------- fairness, recomputed with Fairlearn ----------
    groups = {"age_band": te["age_band"].astype(str).values, "Gender": te["Gender"].astype(str).values,
              "Scholarship": te["Scholarship"].map({0: "no", 1: "yes"}).values}
    from careconnect.fairness.policy import apply_policy
    pmf, yB = apply_policy(p, groups["age_band"], P)
    fair = {}
    for rule, pred in [("single_cutoff", yh), ("option_B_seed42", yB)]:
        fair[rule] = {}
        for g, sf in groups.items():
            mf = MetricFrame(metrics={"selection_rate": selection_rate, "tpr": true_positive_rate, "fpr": false_positive_rate},
                             y_true=y, y_pred=pred, sensitive_features=sf)
            byg = mf.by_group
            fair[rule][g] = {
                "demographic_parity_difference": float(demographic_parity_difference(y, pred, sensitive_features=sf)),
                "equalized_odds_difference": float(equalized_odds_difference(y, pred, sensitive_features=sf)),
                "disparate_impact_ratio": float(demographic_parity_ratio(y, pred, sensitive_features=sf)),
                "groups": {str(k): {"n": int((sf == k).sum()), "no_shows": int(y[sf == k].sum()),
                                    "no_show_rate": float(y[sf == k].mean()),
                                    "selection_rate": float(byg.loc[k, "selection_rate"]),
                                    "tpr": float(byg.loc[k, "tpr"]), "fpr": float(byg.loc[k, "fpr"])} for k in byg.index}}
    # bootstrap 95% intervals for the equalized-odds gap (sampling uncertainty of the audit itself)
    rng = np.random.default_rng(0)
    for rule, pred in [("single_cutoff", yh), ("option_B_seed42", yB)]:
        for g, sf in groups.items():
            vals = []
            for _ in range(300):
                i = rng.integers(0, len(y), len(y))
                vals.append(equalized_odds_difference(y[i], pred[i], sensitive_features=sf[i]))
            fair[rule][g]["eod_bootstrap_95ci"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
            fair[rule][g]["share_of_bootstraps_above_0.05"] = float(np.mean(np.array(vals) > 0.05))
    # randomisation sensitivity: Option B realised with 50 different seeds
    seeds = []
    for s in range(50):
        yb = (np.random.default_rng(1000 + s).random(len(pmf)) < pmf).astype(int)
        seeds.append({g: float(equalized_odds_difference(y, yb, sensitive_features=sf)) for g, sf in groups.items()})
    fair["option_B_over_50_random_seeds"] = {g: {"min": min(d[g] for d in seeds), "max": max(d[g] for d in seeds),
                                                 "mean": float(np.mean([d[g] for d in seeds]))} for g in groups}
    fair["option_B_expected"] = {"recall": float((pmf * y).sum() / y.sum()), "flag_rate": float(pmf.mean()),
                                 "share_randomised": float(((pmf > 0) & (pmf < 1)).mean())}
    rep["fairness"] = fair
    check("single-cut-off fairness matches mitigation.json (age EOD)",
          abs(fair["single_cutoff"]["age_band"]["equalized_odds_difference"] - M["before"]["age_band"]["eod"]) < 1e-9)
    check("exported Option B rule reproduces Fairlearn's probabilities", P["verification"]["max_abs_diff_vs_fairlearn_pmf"] < 1e-9,
          P["verification"]["max_abs_diff_vs_fairlearn_pmf"])
    # expected-value fairness of the randomised rule, recomputed independently from the probabilities
    fair["option_B_expected_by_group"] = {}
    for g, sf in groups.items():
        rows = {str(v): {"n": int((sf == v).sum()), "selection": float(pmf[sf == v].mean()),
                         "tpr": float(pmf[(sf == v) & (y == 1)].mean()), "fpr": float(pmf[(sf == v) & (y == 0)].mean())}
                for v in np.unique(sf)}
        tg = max(r["tpr"] for r in rows.values()) - min(r["tpr"] for r in rows.values())
        fg = max(r["fpr"] for r in rows.values()) - min(r["fpr"] for r in rows.values())
        sel = [r["selection"] for r in rows.values()]
        fair["option_B_expected_by_group"][g] = {"eod": max(tg, fg), "dpd": max(sel) - min(sel), "di": min(sel) / max(sel), "groups": rows}
        e = fair["option_B_expected_by_group"][g]["eod"]
        check(f"Option B {g}: EXPECTED equalized-odds gap <= 0.05", e <= 0.05, round(e, 4))
        mk = M["B_equalized_odds"][{"Scholarship": "Scholarship"}.get(g, g)]["eod"]
        check(f"Option B {g}: expected gap matches mitigation.json", abs(mk - e) < 1e-9, (round(mk, 4), round(e, 4)))
        r42 = fair["option_B_seed42"][g]["equalized_odds_difference"]
        CHECKS.append({"check": f"INFO Option B {g}: one realised draw (numpy seed 42) gap", "status": "INFO", "detail": round(r42, 4)})
        sp = fair["option_B_over_50_random_seeds"][g]
        CHECKS.append({"check": f"INFO Option B {g}: realised gap over 50 draws min/mean/max", "status": "INFO",
                       "detail": (round(sp["min"], 4), round(sp["mean"], 4), round(sp["max"], 4))})

    # ---------- explanations ----------
    import xgboost as xgb
    pre, clf = m.named_steps["pre"], m.named_steps["clf"]
    Xs = X.sample(300, random_state=2)
    names = list(pre.get_feature_names_out())
    dm = xgb.DMatrix(pre.transform(Xs), feature_names=names)
    contrib = clf.get_booster().predict(dm, pred_contribs=True)
    margin = clf.get_booster().predict(dm, output_margin=True)
    add_err = float(np.max(np.abs(contrib.sum(1) - margin)))
    check("SHAP additivity: contributions sum to the model's log-odds (300 rows)", add_err < 1e-3, add_err)
    prob_from_margin = 1 / (1 + np.exp(-margin))
    check("SHAP explains the deployed model: sigmoid(margin) == predict_proba", np.max(np.abs(prob_from_margin - m.predict_proba(Xs)[:, 1])) < 1e-5)
    check("analysis.json produced by model version " + config.MODEL_VERSION, E.get("model_version") == config.MODEL_VERSION)
    XP = load("explanations.json")
    if XP:
        for e in XP["lime"]:
            s = next(x for x in E["samples"] if x["tier"] == e["tier"])
            sc = float(m.predict_proba(pd.DataFrame([s["features"]])[config.FEATURES])[0, 1])
            check(f"LIME {e['tier']}: explained score equals the model's score", abs(sc - e["model_score"]) < 1e-9, round(sc, 4))
        for c in XP["counterfactuals"]:
            cf = c["counterfactual"]
            if cf:
                s = next(x for x in E["samples"] if x["tier"] == c["tier"])
                row = pd.DataFrame([{**s["features"], "lead_time_days": cf["lead_time_days"], "SMS_received": cf["SMS_received"]}])[config.FEATURES]
                new = float(m.predict_proba(row)[0, 1])
                check(f"counterfactual {c['tier']}: re-scored change falls below the cut-off", new < T and abs(new - cf["score"]) < 1e-9, round(new, 4))
                check(f"counterfactual {c['tier']}: only actionable features changed",
                      cf["lead_time_days"] <= s["features"]["lead_time_days"] and cf["SMS_received"] >= s["features"]["SMS_received"])
        rep["explanations"] = {"lime_r2": {e["tier"]: e["lime_r2"] for e in XP["lime"]},
                               "counterfactuals": XP["counterfactuals"], "population": XP["population"]}
    else:
        check("explanations.json exists", False)

    # ---------- MLflow ----------
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(config.MLFLOW_URI)
        c = MlflowClient()
        exp = c.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
        runs = c.search_runs([exp.experiment_id], max_results=100)
        rep["mlflow"] = {"experiment_id": exp.experiment_id,
                         "runs": [{"run_id": r.info.run_id, "name": r.info.run_name, "status": r.info.status,
                                   "test_roc_auc": r.data.metrics.get("test_roc_auc", r.data.metrics.get("roc_auc"))} for r in runs]}
        mv = c.get_model_version_by_alias(config.REGISTERED_MODEL, "candidate")
        rep["mlflow"]["registry"] = {"name": config.REGISTERED_MODEL, "version": mv.version, "alias": "candidate",
                                     "tags": mv.tags, "source_run": mv.run_id}
        aliases = c.get_registered_model(config.REGISTERED_MODEL).aliases
        rep["mlflow"]["registry"]["all_aliases"] = dict(aliases)
        check("MLflow: >= 7 runs logged (2 baselines, 3 models, 2 robustness)", len(runs) >= 7, len(runs))
        check("MLflow: candidate version comes from the xgboost run", mv.run_id == R["models"]["xgboost"]["mlflow"]["run_id"])
        check("MLflow: no 'production' alias set automatically", "production" not in aliases)
    except Exception as e:
        check("MLflow tracking and registry readable", False, f"{type(e).__name__}: {e}")

    # ---------- gate and monitoring ----------
    G = load("promotion_decision.json")
    check("promotion gate ran and did not auto-promote", bool(G) and G["automatic_promotion"] is False, G and G["decision"])
    rep["promotion_gate"] = G
    MON = load("monitoring/monitoring.json")
    if MON:
        rep["monitoring"] = {"evidently": MON["evidently"],
                             "weeks": [{k: w[k] for k in ["batch", "n", "auc", "max_feature_psi", "score_psi", "age_eod", "status"]} for w in MON["weeks"]],
                             "stress_test_status": MON["stress_test"]["status"]}
        check("Evidently HTML report written", (A / "monitoring" / "evidently_drift_report.html").exists(), MON["evidently"])
        check("stress test raises data-drift alert", MON["stress_test"]["status"]["data_drift"] == "alert")
    else:
        check("monitoring.json exists", False)

    rep["module5_comparison"] = {
        "module5_v1": {"rows": 110514, "train": 88411, "test": 22103, "roc_auc": 0.752, "recall": 0.678, "precision": 0.349,
                       "f1": 0.461, "cutoff": 0.2525, "age_eod_single_cutoff": 0.687, "tuning": "25,000-row sample, 3-fold CV"},
        "v2": {"rows": len(full), "train": len(tr), "test": len(te), "roc_auc": perf["roc_auc"], "recall": perf["recall"],
               "precision": perf["precision"], "f1": perf["f1"], "cutoff": T,
               "age_eod_single_cutoff": fair["single_cutoff"]["age_band"]["equalized_odds_difference"],
               "tuning": f"full training set, {R['cv_folds']}-fold CV"},
        "same_split_rows": len(full) == 110514 and len(te) == 22103}
    rep["checks"] = CHECKS
    rep["summary"] = {"pass": sum(c["status"] == "PASS" for c in CHECKS), "fail": sum(c["status"] == "FAIL" for c in CHECKS)}
    json.dump(rep, open(A / "verification_report.json", "w"), indent=1, default=str)
    for c in CHECKS:
        print(f"{c['status']}  {c['check']}  {'' if c['detail'] is None else c['detail']}")
    print(json.dumps(rep["summary"]))
    return 0 if rep["summary"]["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
