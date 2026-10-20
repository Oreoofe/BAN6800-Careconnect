"""Full build / retraining run with a promotion gate.

    python -m careconnect.run_all            # pipeline -> train -> evaluate -> mitigate -> explain -> monitor -> docs -> gate

A finished training job never promotes itself. The gate below checks performance, fairness and
data quality against the Module 4 acceptance criteria and writes artifacts/promotion_decision.json.
Even when every automated check passes, the decision is "eligible for Ethics Committee review":
the MLflow alias `production` is set only by a named approver (see docs/governance.md).
"""
import json
from datetime import datetime, timezone

from careconnect import config


def gate() -> dict:
    """Promotion gate against the ORIGINAL acceptance criteria (Module 4 s.7 staging list, Charter P2 halt rule),
    applied to the configured policy. Criteria are not relaxed after seeing results; F1 (Module 4 s.1 success
    criterion, not in the s.7 staging list) is reported as an unmet original criterion."""
    R = json.load(open(config.ARTIFACTS / "model_results.json"))
    Q = json.load(open(config.ARTIFACTS / "pipeline" / "data_quality_report.json"))
    B = json.load(open(config.ARTIFACTS / "pipeline" / "bias_report.json"))
    PC = json.load(open(config.ARTIFACTS / "policy_comparison.json"))
    x = R["models"]["xgboost"]["at_recall_target"]
    configured = "P4_option_B_equalized_odds"
    acc = PC["policies"][configured]["acceptance"]
    crit = {c["id"]: c["criterion"] for c in PC["criteria"]}
    checks = [("data contract passed", Q["validation"]["gate_passed"], True),
              ("representation-bias gate passed", B["overall_pass"], True),
              ("patient-grouped AUC within 0.02 of random-split AUC", round(R["robustness_patient_grouped"]["roc_auc"], 4),
               abs(R["robustness_patient_grouped"]["roc_auc"] - x["roc_auc"]) <= 0.02)]
    rows = [{"check": c, "value": v, "passed": bool(p), "blocking": True} for c, v, p in checks]
    for cid in ["C1", "C2", "C4", "C5", "C6", "C7", "C8"]:            # Module 4 s.7 staging list + Charter P2
        rows.append({"check": f"{cid} {crit[cid]} (configured policy: Option B)", "value": acc[cid], "passed": acc[cid], "blocking": True})
    # C3: Module 4 s.1 calls ROC-AUC, recall AND F1 "the acceptance threshold"; the s.7 staging list omits F1. The capstone brief
    # sets no numeric threshold. Reviewer decision (cycle 4): treat F1 >= 0.55 as BLOCKING unless the rubric clearly says otherwise.
    rows.append({"check": f"C3 {crit['C3']}", "value": round(x["f1"], 4), "passed": acc["C3"], "blocking": True,
                 "source_note": "Module 4 s.1 'The acceptance threshold is: ROC-AUC >= 0.70, Recall >= 0.65, F1 >= 0.55'; "
                                "omitted from the Module 4 s.7 staging list; treated as blocking by reviewer decision"})
    # Module 4 s.7 staging condition: "Ethics Committee reviews and approves the Model Card" - a human decision, never automated.
    rows.append({"check": "Ethics Committee has reviewed and approved the Model Card (Module 4 s.7)", "value": "not recorded",
                 "passed": False, "blocking": True})
    blocking = [r["check"] for r in rows if r["blocking"] and not r["passed"]]
    decision = ("eligible for Ethics Committee review" if not blocking else
                "NOT ELIGIBLE under the original acceptance criteria; evidence package prepared for an Ethics Committee decision")
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(), "model_version": config.MODEL_VERSION,
           "configured_policy": configured, "configured_policy_status": "evaluation only - not approved for operational use",
           "checks": rows, "blocking_failures": blocking,
           "original_criteria_not_met": [r["check"] for r in rows if not r["passed"]],
           "no_evaluated_policy_reliably_meets_fairness": PC["no_policy_reliably_meets_fairness"],
           "decision": decision, "automatic_promotion": False,
           "recommendation": "Do not deploy until a better model or sufficient additional evidence is available",
           "deployment_status": "not deployed (no public API; live dashboard serves v1 from main)"}
    json.dump(out, open(config.ARTIFACTS / "promotion_decision.json", "w"), indent=1)
    return out


def main():
    from careconnect.pipeline.run import run
    from careconnect.modeling import train, evaluate
    from careconnect.fairness import mitigation, policy_comparison
    from careconnect.explain import local
    from careconnect.monitoring import monitor
    from careconnect.governance import committee_pack, model_card
    run()
    train.main()
    evaluate.main()
    mitigation.main()
    policy_comparison.main()
    local.main()
    monitor.main()
    g = gate()
    model_card.main()
    committee_pack.main()
    print(json.dumps(g, indent=1))


if __name__ == "__main__":
    main()
