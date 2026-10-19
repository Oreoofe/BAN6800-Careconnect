"""VALIDATE: executable data contract (the Module 3 quality gate).

Module 3 specified a 24-expectation Great Expectations suite. That suite was not preserved, so
the same contract is implemented here in plain pandas: each check is named, counted and logged,
and any *blocking* failure stops the pipeline before cleaning. Known source defects that the
cleaning stage is designed to remove (bad ages, inverted dates, Handcap > 4) are recorded as
warnings with their counts, not as failures.
"""
import pandas as pd

from careconnect import config
from careconnect.pipeline.audit import audit


def run_checks(df: pd.DataFrame, expect_rows: tuple[int, int] | None = (110_000, 111_000)) -> dict:
    checks = []

    def chk(name, passed, blocking=True, detail=None):
        checks.append({"check": name, "passed": bool(passed), "blocking": blocking, "detail": detail})

    # Table shape
    if expect_rows:
        chk("row_count_in_range", expect_rows[0] <= len(df) <= expect_rows[1], detail=len(df))
    chk("column_count_is_14", len(df.columns) == 14, detail=len(df.columns))
    chk("columns_match_contract", list(df.columns) == config.RAW_COLUMNS,
        detail={"missing": [c for c in config.RAW_COLUMNS if c not in df.columns],
                "unexpected": [c for c in df.columns if c not in config.RAW_COLUMNS]})
    if not checks[-1]["passed"]:            # schema broken: later checks cannot run safely
        return {"n_checks": len(checks), "passed": sum(c["passed"] for c in checks),
                "blocking_failures": [c["check"] for c in checks if not c["passed"]], "warnings": [],
                "checks": checks, "gate_passed": False}
    # Identifier integrity
    chk("appointment_id_not_null", df["AppointmentID"].notna().all())
    chk("appointment_id_unique", df["AppointmentID"].is_unique, blocking=False,
        detail=int(df["AppointmentID"].duplicated().sum()))
    chk("patient_id_not_null", df["PatientId"].notna().all())
    # Domain validity
    chk("gender_in_F_M", df["Gender"].isin(["F", "M"]).all())
    chk("target_in_yes_no", df["No-show"].isin(["Yes", "No"]).all())
    for c in ["Scholarship", "Hipertension", "Diabetes", "Alcoholism", "SMS_received"]:
        chk(f"{c}_binary", df[c].isin([0, 1]).all())
    chk("handcap_0_to_4", df["Handcap"].between(0, 4).all(), blocking=False,
        detail=int((~df["Handcap"].between(0, 4)).sum()))
    # Range and format
    chk("age_within_0_100", df["Age"].between(config.AGE_MIN, config.AGE_MAX).all(), blocking=False,
        detail=int((~df["Age"].between(config.AGE_MIN, config.AGE_MAX)).sum()))
    sched = pd.to_datetime(df["ScheduledDay"], errors="coerce", utc=True)
    appt = pd.to_datetime(df["AppointmentDay"], errors="coerce", utc=True)
    chk("scheduled_day_parses", sched.notna().all())
    chk("appointment_day_parses", appt.notna().all())
    chk("scheduled_not_after_appointment_day", (sched.dt.normalize() <= appt).all(), blocking=False,
        detail=int((sched.dt.normalize() > appt).sum()))
    # Completeness
    for c in ["PatientId", "Age", "Neighbourhood", "No-show"]:
        chk(f"{c}_no_nulls", df[c].notna().all())
    # Plausibility of the label rate (guards against a swapped or truncated file)
    rate = (df["No-show"] == "Yes").mean()
    chk("no_show_rate_10_to_35pct", 0.10 <= rate <= 0.35, detail=round(float(rate), 4))

    blocking_fail = [c["check"] for c in checks if c["blocking"] and not c["passed"]]
    result = {"n_checks": len(checks), "passed": sum(c["passed"] for c in checks),
              "blocking_failures": blocking_fail, "warnings": [c for c in checks if not c["blocking"] and not c["passed"]],
              "checks": checks, "gate_passed": not blocking_fail}
    return result


def validate(df: pd.DataFrame, **kw) -> dict:
    result = run_checks(df, **kw)
    audit("VALIDATE", n_checks=result["n_checks"], passed=result["passed"],
          blocking_failures=result["blocking_failures"], warnings=[w["check"] for w in result["warnings"]])
    if not result["gate_passed"]:
        raise RuntimeError(f"Data contract failed: {result['blocking_failures']}. Pipeline halted before cleaning.")
    return result
