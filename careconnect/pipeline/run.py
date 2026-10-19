"""Run the Module 3 pipeline end to end:
EXTRACT -> VALIDATE -> CLEAN -> ANONYMIZE -> FEATURES -> BIAS_CHECK -> PUBLISH.

    python -m careconnect.pipeline.run                 # real data (downloads + verifies checksum)
    python -m careconnect.pipeline.run --sample        # synthetic smoke test (tools/make_sample_data.py)

Outputs: data/processed/careconnect_modeling.parquet (git-ignored, pseudonymised) and
artifacts/pipeline/{data_quality_report,bias_report,lineage}.json (aggregate only, committed).
"""
import argparse
import json
from datetime import datetime, timezone

from careconnect import config
from careconnect.pipeline.anonymize import anonymize
from careconnect.pipeline.audit import audit, config_hash, git_sha, write_table
from careconnect.pipeline.bias_check import run_bias_checks
from careconnect.pipeline.clean import clean
from careconnect.pipeline.extract import extract
from careconnect.pipeline.features import engineer_features
from careconnect.pipeline.validate import validate

OUT = config.ARTIFACTS / "pipeline"


def run(sample: bool = False, enforce_bias_gate: bool = True) -> dict:
    raw = extract(config.SAMPLE_CSV if sample else None, verify=not sample)
    lineage = [{"stage": "EXTRACT", "rows": len(raw)}]
    vq = validate(raw, expect_rows=None if sample else (110_000, 111_000))
    lineage.append({"stage": "VALIDATE", "rows": len(raw), "checks_passed": vq["passed"], "checks": vq["n_checks"]})
    df, cr = clean(raw)
    lineage.append({"stage": "CLEAN", "rows": len(df)})
    df, ar = anonymize(df)
    lineage.append({"stage": "ANONYMIZE", "rows": len(df)})
    df, fr = engineer_features(df)
    lineage.append({"stage": "FEATURE_ENGINEERING", "rows": len(df), "columns": len(df.columns)})
    bias = run_bias_checks(df)
    lineage.append({"stage": "BIAS_CHECK", "rows": len(df), "overall_pass": bias["overall_pass"]})
    if enforce_bias_gate and not bias["overall_pass"]:
        audit("PUBLISH", published=False, reason="bias gate failed")
        raise RuntimeError("Representation-bias gate failed; modelling table not published.")

    keep = config.FEATURES + [config.TARGET, "patient_id_hash", "ScheduledDay", "AppointmentDay"]
    table = df[keep]
    write_table(table, config.MODELING_TABLE)
    lineage.append({"stage": "PUBLISH", "rows": len(table), "columns": len(table.columns),
                    "path": str(config.MODELING_TABLE.relative_to(config.ROOT))})
    audit("PUBLISH", published=True, rows=len(table), columns=len(table.columns))

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"generated_utc": datetime.now(timezone.utc).isoformat(), "git_sha": git_sha(),
            "config_hash": config_hash(), "sample": sample}
    json.dump({**meta, "validation": vq, "cleaning": cr, "anonymization": ar, "features": fr},
              open(OUT / "data_quality_report.json", "w"), indent=1, default=str)
    json.dump({**meta, **bias}, open(OUT / "bias_report.json", "w"), indent=1, default=str)
    json.dump({**meta, "stages": lineage}, open(OUT / "lineage.json", "w"), indent=1, default=str)
    return {"rows": len(table), "cleaning": cr, "anonymization": ar, "bias_pass": bias["overall_pass"], "lineage": lineage}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="run on the synthetic smoke-test file")
    a = ap.parse_args()
    print(json.dumps(run(sample=a.sample, enforce_bias_gate=not a.sample), indent=1, default=str))
