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
    POL = load("fairness_policy.json")
    from careconnect.fairness.policy import flag_probability
    if XP:
        allex = {x["tier"]: x for x in E["samples"] + E["samples_documented_cycle1"]}
        for x in E["samples"]:
            check(f"example '{x['tier']}' is genuinely in its score tier", x["tier"] == x["score_tier"], round(x["prob"], 4))
        for e in XP["lime"]:
            s = allex[e["example"]]
            sc = float(m.predict_proba(pd.DataFrame([s["features"]])[config.FEATURES])[0, 1])
            check(f"LIME [{e['config']}] {e['example']}: explains the v2 model's score", abs(sc - e["model_score"]) < 1e-9, round(sc, 4))
            CHECKS.append({"check": f"INFO LIME [{e['config']}] {e['example']}: R2 / |error| / top-3 overlap / fidelity",
                           "status": "INFO", "detail": (round(e["lime_r2"], 3), round(e["abs_error"], 3), e["top3_overlap"], e["fidelity"])})
        for c in XP["counterfactuals"]:
            s = allex[c["example"]]
            sc0 = float(m.predict_proba(pd.DataFrame([s["features"]])[config.FEATURES])[0, 1])
            pf0 = flag_probability(sc0, str(s["features"]["age_band"]), POL)
            check(f"counterfactual {c['example']}: original policy flag probability re-computed", abs(pf0 - c["original_policy_flag_probability"]) < 1e-12, round(pf0, 4))
            cf = c["counterfactual"]
            if cf:
                row = pd.DataFrame([{**s["features"], "lead_time_days": cf["lead_time_days"], "SMS_received": cf["SMS_received"]}])[config.FEATURES]
                new = float(m.predict_proba(row)[0, 1])
                pf = flag_probability(new, str(s["features"]["age_band"]), POL)
                check(f"counterfactual {c['example']}: re-scored change gives policy flag probability 0", pf == 0.0 and abs(new - cf["score"]) < 1e-9, round(new, 4))
                check(f"counterfactual {c['example']}: feasible (earlier slot >= 0 days, SMS only added)",
                      0 <= cf["lead_time_days"] <= s["features"]["lead_time_days"] and cf["SMS_received"] >= s["features"]["SMS_received"])
            else:
                CHECKS.append({"check": f"INFO counterfactual {c['example']}: none found", "status": "INFO",
                               "detail": ("not flagged now" if not c["applicable"] else
                                          f"lowest reachable flag probability {c['lowest_reachable']['policy_flag_probability']:.3f}")})
        rep["explanations"] = {"lime": [{k: e[k] for k in ["config", "example", "model_score", "lime_local_prediction", "lime_r2",
                                                            "abs_error", "lime_top3", "shap_top3", "top3_overlap", "fidelity"]} for e in XP["lime"]],
                               "counterfactuals": XP["counterfactuals"], "population": XP["population"]}
    else:
        check("explanations.json exists", False)

    # ---------- policy comparison: recompute the deterministic rules from their stored cut-offs ----------
    PC = load("policy_comparison.json")
    if PC:
        for k, v in PC["policies"].items():
            if "cutoffs" in v:
                cut = pd.Series(groups["age_band"]).map(v["cutoffs"]).values.astype(float)
                yk = (p >= cut).astype(int)
                rec = float(yk[y == 1].mean())
                eodk = float(equalized_odds_difference(y, yk, sensitive_features=groups["age_band"]))
                check(f"policy {k}: recall and age EOD reproduce from stored cut-offs",
                      abs(rec - v["recall"]) < 1e-9 and abs(eodk - v["age_band"]["eod"]) < 1e-9, (round(rec, 4), round(eodk, 4)))
        b = PC["policies"]["P4_option_B_equalized_odds"]
        check("policy comparison Option B equals mitigation.json Option B (age EOD, expected)",
              abs(b["age_band"]["eod"] - M["B_equalized_odds"]["age_band"]["eod"]) < 1e-9)
        rep["policy_comparison"] = {k: {x: v[x] for x in ["recall", "precision", "f1", "flag_rate", "share_decisions_randomised"]}
                                    | {"age_eod": v["age_band"]["eod"], "age_ci": v["age_band"]["eod_bootstrap_95ci"]}
                                    for k, v in PC["policies"].items()}
    else:
        check("policy_comparison.json exists", False)

    # ---------- acceptance matrix, recomputed from the stored metrics (criteria as originally written) ----------
    if PC:
        for k, v in PC["policies"].items():
            a, g_, sc = v["age_band"], v["Gender"], v["Scholarship"]
            auc = v.get("test_roc_auc", PC["reference_test_roc_auc_full_model"])
            rel = all(x["eod_bootstrap_95ci"][1] <= 0.05 for x in (a, g_, sc)) and (
                not v["randomised"] or all(x["realised_eod"]["p95"] <= 0.05 for x in (a, g_, sc)))
            mine = {"C1": auc >= 0.70, "C2": v["recall"] >= 0.65, "C3": v["f1"] >= 0.55,
                    "C4": a["eod"] <= 0.05 and g_["eod"] <= 0.05, "C5": a["dpd"] <= 0.05 and g_["dpd"] <= 0.05,
                    "C6": a["di_ratio"] >= 0.8 and g_["di_ratio"] >= 0.8, "C7": sc["eod"] <= 0.05, "C8": rel}
            check(f"acceptance matrix for {k} reproduces", all(mine[c] == v["acceptance"][c] for c in mine),
                  "".join(c if mine[c] else "-" for c in sorted(mine)))
        none = not any(v["acceptance"]["meets_full_fairness_requirement"] for v in PC["policies"].values())
        check("statement 'no evaluated policy reliably meets the 0.05 equalized-odds limit across all required groups' is supported",
              none == PC["no_policy_reliably_meets_fairness"] and none)
        rep["acceptance_matrix"] = {k: v["acceptance"] for k, v in PC["policies"].items()}
        # Option B mechanism: lowest-scoring appointment in each mixed group still has a non-zero flag probability
        mech = PC["option_B_mechanism"]
        for grp, r in POL["groups"].items():
            k = groups["age_band"] == grp
            pmin = float(min(p[k]))
            pf = flag_probability(pmin, grp, POL)
            expect = r["p_ignore"] * r["prediction_constant"]
            check(f"Option B {grp}: lowest-score appointment's flag probability = p_ignore x x_best", abs(pf - expect) < 1e-9,
                  (round(pmin, 4), round(pf, 4)))
        floors = np.array([POL["groups"][x]["p_ignore"] * POL["groups"][x]["prediction_constant"] for x in groups["age_band"]])
        check("Option B share of flags unrelated to risk reproduces", abs(floors.sum() / pmf.sum() - mech["share_of_flags_unrelated_to_risk"]) < 1e-9,
              round(floors.sum() / pmf.sum(), 4))
        rep["option_B_mechanism"] = mech

    # ---------- documentation states the measured findings ----------
    must = {"docs/governance.md": ["unrelated to predicted risk", "every monitored time window", "equal opportunity", "no evaluated policy"],
            "docs/algorithmic_impact_assessment.md": ["unrelated to predicted risk", "every monitored time window", "equal opportunity"],
            "docs/public_trust_statement.md": ["regardless of", "not yet met"],
            "docs/model_card.md": ["unrelated to predicted risk", "every monitored time window", "equal opportunity", "No evaluated policy"],
            "docs/committee_decision_pack.md": ["unrelated to predicted risk", "every monitored time window", "equal opportunity", "No evaluated policy"]}
    for f, phrases in must.items():
        path = config.ROOT / f
        txt = path.read_text() if path.exists() else ""
        miss = [x for x in phrases if x.lower() not in txt.lower()]
        check(f"{f} discloses the measured findings", not miss, miss or None)

    # ---------- build identity ----------
    import hashlib
    import os
    sha = os.environ.get("GITHUB_SHA", "")[:7]
    rep["build"] = {"commit": sha or None, "model_results_git_sha": R.get("git_sha"),
                    "model_sha256": hashlib.sha256((A / "xgboost.joblib").read_bytes()).hexdigest(),
                    "policy_sha256": hashlib.sha256((A / "fairness_policy.json").read_bytes()).hexdigest()}
    if sha:
        check("model trained from the commit being built (GITHUB_SHA)", R.get("git_sha") == sha, (R.get("git_sha"), sha))
        audit_shas = {json.loads(line)["git_sha"] for line in open(config.ROOT / "logs" / "audit.log")}
        check("every pipeline audit event stamped with the commit being built", audit_shas == {sha}, sorted(audit_shas))

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
        check("Evidently HTML report written", (A / "monitoring" / "evidently_drift_report.html").exists(), MON["evidently"])
        st = {t["name"][:2]: t for t in MON["stress_tests"]}
        check("S1 stress test (lead-time shift) raises data-drift alert", st["S1"]["status"]["data_drift"] == "alert")
        check("S2 stress test (older patients' flags removed) raises fairness halt", st["S2"]["fairness"]["status"] == "halt",
              st["S2"]["fairness"]["status"])
        check("S3 stress test (25 elderly appointments) leaves elderly unevaluated",
              "elderly" in st["S3"]["fairness"]["groups_not_evaluated"])
        check("S4 stress test (300 appointments) returns 'insufficient data'", st["S4"]["fairness"]["status"] == "insufficient data",
              st["S4"]["fairness"]["status"])
        # independent recompute of the latest full rolling window's realised gap
        from careconnect.fairness.policy import apply_policy as _ap
        last = MON["rolling_28d"][-1]
        end_ = te["AppointmentDay"].max()
        win = te[te["AppointmentDay"] > end_ - pd.Timedelta(days=28)]
        pw = m.predict_proba(win[config.FEATURES])[:, 1]
        _, yw = _ap(pw, win["age_band"].astype(str).values, POL, seed=100 + len(MON["rolling_28d"]) - 1)
        gw = win["age_band"].astype(str).values
        k = np.isin(gw, last["fairness"]["groups_evaluated"])
        e2 = float(equalized_odds_difference(win[config.TARGET].values[k], yw[k], sensitive_features=gw[k]))
        # status logic consistent with the stored interval for every evaluated window
        bad = []
        for setname in ["rolling_28d", "weeks_cycle2_rule", "calendar_months", "stress_tests"]:
            for r in MON[setname]:
                f = r["fairness"]
                if f["status"] == "insufficient data":
                    ok_ = f["eod"] is None and len(f["groups_evaluated"]) < 2
                else:
                    lo, hi = f["ci95"]
                    want = "halt" if lo > 0.05 else ("ok" if hi <= 0.05 else "inconclusive")
                    ok_ = f["status"] == want and len(f["groups_evaluated"]) >= 2
                ok_ = ok_ and not (set(f["groups_evaluated"]) & set(f["groups_not_evaluated"]))
                if not ok_:
                    bad.append(r["batch"])
        check("monitoring: every window's status follows its interval; insufficient data never scored", not bad, bad or None)
        S = MON["summary"]
        check("monitoring: expected Option B age gap > 0.05 in every evaluable window (rolling + months)",
              S["rolling_28d_all"]["windows_with_expected_gap_above_0.05"] == S["rolling_28d_all"]["windows_with_expected_gap_evaluable"]
              and S["calendar_months"]["windows_with_expected_gap_above_0.05"] == S["calendar_months"]["windows_with_expected_gap_evaluable"],
              (S["rolling_28d_all"]["expected_gap_range"], S["calendar_months"]["expected_gap_range"]))
        check("latest rolling window: realised age gap reproduces", abs(e2 - (last["fairness"]["eod"] or -1)) < 1e-9, round(e2, 4))
        rep["monitoring"] = {
            "rule": MON["fairness_rule"], "evidently": MON["evidently"],
            "rolling_28d": [{"window": r["batch"], "n": r["n"], "auc": r["auc"], "max_psi": r["max_feature_psi"], "max_psi_feature": r["max_psi_feature"],
                             "fairness_status": r["fairness"]["status"], "eod": r["fairness"]["eod"], "ci95": r["fairness"]["ci95"],
                             "not_evaluated": list(r["fairness"]["groups_not_evaluated"])} for r in MON["rolling_28d"]],
            "weeks_cycle1": [{"week": w["batch"], "eod": w["age_eod"], "status": w["status"]["bias_drift"]} for w in MON["weeks_cycle1_rule"]],
            "weeks_cycle2": [{"week": w["batch"], "status": w["fairness"]["status"], "eod": w["fairness"]["eod"], "ci95": w["fairness"]["ci95"],
                              "not_evaluated": list(w["fairness"]["groups_not_evaluated"])} for w in MON["weeks_cycle2_rule"]],
            "months": [{"month": r["batch"], "n": r["n"], "status": r["fairness"]["status"], "eod": r["fairness"]["eod"], "ci95": r["fairness"]["ci95"]}
                       for r in MON["calendar_months"]],
            "stress": [{"name": t["name"], "expect": t["expect"], "fairness": t["fairness"]["status"], "status": t["status"]} for t in MON["stress_tests"]],
            "summary": MON["summary"],
            "expected_vs_realised": [{"window": r["batch"], "realised": r["fairness"]["eod"], "expected": r["expected_age_eod_evaluated_groups"],
                                      "status": r["fairness"]["status"]} for r in MON["rolling_28d"] + MON["calendar_months"]]}
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
