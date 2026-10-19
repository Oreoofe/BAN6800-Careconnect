"""CLEAN: remove records that are invalid by the data dictionary rules (Module 2, Appendix A.1).

Every rule removes or corrects rows for a documented reason, and the counts are logged.
Rows are removed rather than capped: capping an impossible age would invent clinical signal.
"""
import pandas as pd

from careconnect import config
from careconnect.pipeline.audit import audit


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rep = {"rows_in": len(df)}
    df = df.copy()
    n = len(df)
    df = df.drop_duplicates(subset="AppointmentID", keep="first")                       # RAID I-2
    rep["duplicate_appointment_ids_removed"] = n - len(df)

    n = len(df)
    df = df[df["Age"].between(config.AGE_MIN, config.AGE_MAX)]                           # registration errors
    rep["invalid_age_removed"] = n - len(df)

    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"], utc=True).dt.tz_localize(None)
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], utc=True).dt.tz_localize(None)
    n = len(df)
    df = df[df["ScheduledDay"].dt.normalize() <= df["AppointmentDay"]]                   # RAID I-3
    rep["scheduled_after_appointment_removed"] = n - len(df)

    bad = ~df["Handcap"].between(0, 4)                                                    # RAID I-1
    rep["handcap_out_of_range_set_to_0"] = int(bad.sum())
    df.loc[bad, "Handcap"] = 0

    df["noshow_flag"] = (df["No-show"] == "Yes").astype("int8")                           # 1 = no-show
    df = df.drop(columns=["No-show"])     # raw label text dropped at source (Module 5 found it leaking into training)
    df["Gender"] = df["Gender"].astype(str)
    df["Neighbourhood"] = df["Neighbourhood"].astype(str).str.strip().str.upper()
    rep["rows_out"] = len(df)
    audit("CLEAN", **rep)
    return df.reset_index(drop=True), rep
