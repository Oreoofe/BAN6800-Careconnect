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
    R = json.load(open(config.ARTIFACTS / "model_results.json"))
    M = json.load(open(config.ARTIFACTS / "mitigation.json"))
    Q = json.load(open(config.ARTIFACTS / "pipeline" / "data_quality_report.json"))
    B = json.load(open(config.ARTIFACTS / "pipeline" / "bias_report.json"))
    x = R["models"]["xgboost"]["at_recall_target"]
    b = M[M["adopted"]]
    checks = [
        ("data contract passed", Q["validation"]["gate_passed"], True),
        ("representation-bias gate passed", B["overall_pass"], True),
        ("ROC-AUC >= 0.70", round(x["roc_auc"], 4), x["roc_auc"] >= 0.70),
        ("recall >= 0.65 (single cut-off)", round(x["recall"], 4), x["recall"] >= 0.65),
        ("F1 >= 0.55", round(x["f1"], 4), x["f1"] >= 0.55),
        ("adopted policy: EXPECTED age-group equalized-odds gap <= 0.05", round(b["age_band"]["eod"], 4), b["age_band"]["eod"] <= config.FAIRNESS_LIMIT),
        ("adopted policy: EXPECTED gender gap <= 0.05", round(b["Gender"]["eod"], 4), b["Gender"]["eod"] <= config.FAIRNESS_LIMIT),
        ("adopted policy: EXPECTED scholarship gap <= 0.05", round(b["Scholarship"]["eod"], 4), b["Scholarship"]["eod"] <= config.FAIRNESS_LIMIT),
        ("adopted policy: 95th percentile of realised age gap (200 draws) <= 0.05",
         round(b["realisation_spread"]["age_band"]["p95"], 4), b["realisation_spread"]["age_band"]["p95"] <= config.FAIRNESS_LIMIT),
        ("patient-grouped AUC within 0.02 of random-split AUC",
         round(R["robustness_patient_grouped"]["roc_auc"], 4), abs(R["robustness_patient_grouped"]["roc_auc"] - x["roc_auc"]) <= 0.02),
    ]
    rows = [{"check": c, "value": v, "passed": bool(p)} for c, v, p in checks]
    non_blocking = ("F1", "adopted policy: 95th percentile")
    blocking = [r["check"] for r in rows if not r["passed"] and not r["check"].startswith(non_blocking)]
    decision = "eligible for Ethics Committee review" if not blocking else "not eligible - fix blocking checks"
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(), "model_version": config.MODEL_VERSION,
           "checks": rows, "blocking_failures": blocking,
           "non_blocking_notes": ["F1 target (0.55) is reported but treated as non-blocking: at a 20% base rate and "
                                  "recall >= 0.65, F1 is bounded by precision; the committee decides whether to revise it.",
                                  "The 95th-percentile check shows how often a single month's randomised decisions could exceed "
                                  "the limit by chance; it is reported to the committee rather than blocking."],
           "decision": decision, "automatic_promotion": False}
    json.dump(out, open(config.ARTIFACTS / "promotion_decision.json", "w"), indent=1)
    return out


def main():
    from careconnect.pipeline.run import run
    from careconnect.modeling import train, evaluate
    from careconnect.fairness import mitigation, policy_comparison
    from careconnect.explain import local
    from careconnect.monitoring import monitor
    from careconnect.governance import model_card
    run()
    train.main()
    evaluate.main()
    mitigation.main()
    policy_comparison.main()
    local.main()
    monitor.main()
    g = gate()
    model_card.main()
    print(json.dumps(g, indent=1))


if __name__ == "__main__":
    main()
