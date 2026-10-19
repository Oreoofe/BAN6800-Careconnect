"""Generate a synthetic raw CSV that mirrors the Kaggle no-show schema and plants
every data-quality issue documented in the Module 2 Data Dictionary (Appendix A.1),
so the pipeline can be smoke-tested before the real dataset is available.
Run:  python -m tools.make_sample_data  ->  data/raw/sample_no_show.csv
"""
import numpy as np
import pandas as pd
from careconnect import config

def make_sample(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    hoods = (["DOWNTOWN", "NORTH SIDE", "SOUTH SIDE"] * 60 +
             [f"HOOD_{i:02d}" for i in range(30)])          # heavy tail -> rare levels
    g = rng.choice(["F", "M"], n, p=[0.65, 0.35])
    df = pd.DataFrame({
        "PatientId": rng.choice(np.arange(1, n // 3), n),
        "AppointmentID": rng.integers(500000, 600000, n),
        "Gender": g,
        "ScheduledDay": pd.to_datetime("2016-04-01") + pd.to_timedelta(rng.integers(0, 40, n), unit="D"),
        "AppointmentDay": pd.to_datetime("2016-05-01") + pd.to_timedelta(rng.integers(0, 40, n), unit="D"),
        "Age": rng.integers(-1, 102, n),                    # includes negative + extreme
        "Neighbourhood": rng.choice(hoods, n),
        "Scholarship": rng.integers(0, 2, n),
        "Hipertension": rng.integers(0, 2, n),
        "Dipertension": 0,
        "Diabetes": rng.integers(0, 2, n),
        "Alcoholism": rng.integers(0, 2, n),
        "Handcap": rng.choice([0, 0, 0, 1, 1, 2, 3, 4, 5], n),  # 5 = invalid
        "SMS_received": rng.integers(0, 2, n),
        "No-show": rng.choice(["No", "Yes"], n, p=[0.8, 0.2]),
    })
    df = df.drop(columns=["Dipertension"])                   # trap column removed
    # plant ~2% duplicated AppointmentIDs (RAID I-2)
    dup_idx = rng.choice(df.index, max(1, n // 50), replace=False)
    # (fixed in the capstone rebuild: the Module 3 version assigned each ID to itself, so no duplicates were
    #  actually planted; the 45 duplicates seen in the Module 3 smoke test were random ID collisions)
    donors = rng.choice(df.index.difference(dup_idx), len(dup_idx), replace=False)
    df.loc[dup_idx, "AppointmentID"] = df.loc[donors, "AppointmentID"].values
    # plant ~1% inverted dates (RAID I-3)
    inv_idx = rng.choice(df.index, max(1, n // 100), replace=False)
    df.loc[inv_idx, "ScheduledDay"] = (pd.to_datetime(df.loc[inv_idx, "AppointmentDay"])
                                       + pd.Timedelta(days=3)).astype(str)
    out = config.SAMPLE_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Sample written: {out} ({len(df)} rows)")
    return out

if __name__ == "__main__":
    make_sample()
