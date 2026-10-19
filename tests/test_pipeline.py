"""Pipeline tests on the synthetic sample (no network, no real data needed)."""
import pytest

from careconnect import config
from careconnect.pipeline.anonymize import anonymize, pseudonym
from careconnect.pipeline.bias_check import run_bias_checks
from careconnect.pipeline.clean import clean
from careconnect.pipeline.extract import extract
from careconnect.pipeline.features import engineer_features
from careconnect.pipeline.validate import run_checks, validate
from tools.make_sample_data import make_sample


@pytest.fixture(scope="module")
def processed():
    make_sample()
    raw = extract(config.SAMPLE_CSV, verify=False)
    df, cr = clean(raw)
    df, _ = anonymize(df)
    df, _ = engineer_features(df)
    return raw, df, cr


def test_no_direct_identifiers(processed):
    _, df, _ = processed
    assert "PatientId" not in df.columns and "AppointmentID" not in df.columns
    assert "No-show" not in df.columns              # raw label text must never reach modelling


def test_pseudonym_is_deterministic_and_salted():
    assert pseudonym(12345) == pseudonym(12345)
    assert pseudonym(12345) != pseudonym(12346)
    assert pseudonym(12345, salt="a") != pseudonym(12345, salt="b")


def test_cleaning_rules(processed):
    raw, df, cr = processed
    assert df["Age"].between(config.AGE_MIN, config.AGE_MAX).all()
    assert (df["ScheduledDay"].dt.normalize() <= df["AppointmentDay"]).all()
    assert df["Handcap"].between(0, 4).all()
    assert set(df["noshow_flag"].unique()) <= {0, 1}
    assert cr["duplicate_appointment_ids_removed"] > 0      # planted defects are caught
    assert cr["invalid_age_removed"] > 0


def test_history_uses_only_known_outcomes(processed):
    _, df, _ = processed
    for _, g in df.groupby("patient_id_hash"):
        for _, r in g.iterrows():
            known = g["AppointmentDay"] < r["ScheduledDay"].normalize()
            assert r["prior_appointments"] == known.sum()
            assert r["prior_noShows"] == g.loc[known, "noshow_flag"].sum()


def test_validation_gate_halts_on_bad_schema(processed):
    raw, _, _ = processed
    bad = raw.drop(columns=["Gender"])
    with pytest.raises(RuntimeError):
        validate(bad, expect_rows=None)
    assert run_checks(raw, expect_rows=None)["gate_passed"]


def test_bias_gate_reports_small_subgroups(processed):
    _, df, _ = processed
    rep = run_bias_checks(df)
    assert rep["overall_pass"] is False          # 3,000-row sample cannot meet the 500-per-group minimum
    rep_small = run_bias_checks(df, min_n=10)
    assert {c["check"] for c in rep_small["checks"]} >= {"gender_representation", "label_rate_parity_gender"}


def test_no_data_files_tracked():
    import subprocess
    try:
        files = subprocess.check_output(["git", "ls-files"], cwd=config.ROOT).decode().split()
    except Exception:
        pytest.skip("not a git checkout")
    leaked = [f for f in files if f.endswith((".csv", ".parquet", ".pkl")) or f.startswith(("data/raw", "data/processed"))]
    assert not leaked, f"data files committed: {leaked}"
