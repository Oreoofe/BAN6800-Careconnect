"""FEATURE ENGINEERING, with the prior-history leak corrected.

Modules 3-4 counted a patient's earlier appointments by *scheduling* order. An appointment can be
booked earlier than the current one but take place later, so its outcome is not yet known when the
current appointment is scored. On the real data that affected 29,727 of 110,514 rows.

Here a prior appointment counts only if its appointment day is strictly before the day the current
appointment was booked, i.e. its outcome was knowable at scoring time.
"""
import numpy as np
import pandas as pd

from careconnect import config
from careconnect.pipeline.audit import audit


def age_band(age: int) -> str:
    for name, lo, hi in config.AGE_BANDS:
        if lo <= age <= hi:
            return name
    raise ValueError(f"age out of range: {age}")


def prior_history(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised outcome-known history: for each row, prior visits by the same patient that
    happened before this row's booking day, and how many of those were missed."""
    pid = pd.factorize(df["patient_id_hash"])[0].astype(np.int64)
    appt_day = df["AppointmentDay"].values.astype("datetime64[D]").astype(np.int64)
    book_day = df["ScheduledDay"].dt.normalize().values.astype("datetime64[D]").astype(np.int64)
    big = np.int64(100_000)                                   # > number of days spanned
    key_appt = pid * big + appt_day
    order = np.argsort(key_appt, kind="stable")
    sorted_keys = key_appt[order]
    y_sorted = df["noshow_flag"].values[order].astype(np.int64)
    cum_y = np.concatenate([[0], np.cumsum(y_sorted)])
    # index of first appointment of each patient in the sorted array
    block_start = np.searchsorted(sorted_keys, pid * big, side="left")
    pos = np.searchsorted(sorted_keys, pid * big + book_day, side="left")   # strictly earlier appointment days
    out = pd.DataFrame(index=df.index)
    out["prior_appointments"] = (pos - block_start).astype(np.int64)
    out["prior_noShows"] = (cum_y[pos] - cum_y[block_start]).astype(np.int64)
    out["prior_noShow_rate"] = np.where(out["prior_appointments"] > 0,
                                        out["prior_noShows"] / out["prior_appointments"].clip(lower=1), 0.0)
    return out


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["lead_time_days"] = (df["AppointmentDay"] - df["ScheduledDay"].dt.normalize()).dt.days.astype(int)
    df["appointment_dow"] = df["AppointmentDay"].dt.dayofweek.astype(int)
    df["appointment_month"] = df["AppointmentDay"].dt.month.astype(int)
    df["age_band"] = df["Age"].map(age_band)
    df = pd.concat([df, prior_history(df)], axis=1)
    rep = {"engineered": ["lead_time_days", "appointment_dow", "appointment_month", "age_band",
                          "prior_appointments", "prior_noShows", "prior_noShow_rate"],
           "history_rule": "prior visit counted only if AppointmentDay < booking day of current row",
           "rows_with_history": int((df["prior_appointments"] > 0).sum()), "columns_out": len(df.columns)}
    audit("FEATURE_ENGINEERING", **rep)
    return df, rep
