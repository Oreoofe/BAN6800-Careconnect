"""ANONYMIZE: remove direct identifiers before anything reaches the modelling zone.

* AppointmentID is dropped.
* PatientId is replaced by a salted HMAC-SHA256 pseudonym (needed only to build per-patient
  history and patient-grouped evaluation; it never leaves the processed zone).
* Neighbourhoods with fewer than K_ANON_MIN records are grouped as OTHER (k-anonymity on a
  quasi-identifier, Sweeney 2002).
"""
import hashlib
import hmac

import pandas as pd

from careconnect import config
from careconnect.pipeline.audit import audit


def pseudonym(value, salt: str = config.SALT) -> str:
    return hmac.new(salt.encode(), str(int(value)).encode(), hashlib.sha256).hexdigest()[:16]


def anonymize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    n_hoods_in = df["Neighbourhood"].nunique()
    df["patient_id_hash"] = df["PatientId"].map(pseudonym)
    df = df.drop(columns=["PatientId", "AppointmentID"])
    counts = df["Neighbourhood"].value_counts()
    rare = counts[counts < config.K_ANON_MIN].index
    df.loc[df["Neighbourhood"].isin(rare), "Neighbourhood"] = "OTHER"
    rep = {"columns_removed": ["PatientId", "AppointmentID"], "pseudonym": "HMAC-SHA256 (salted, truncated 64-bit)",
           "neighbourhoods_in": int(n_hoods_in), "neighbourhoods_grouped_as_other": int(len(rare)),
           "neighbourhoods_out": int(df["Neighbourhood"].nunique()), "rows_out": len(df)}
    audit("ANONYMIZE", **rep)
    return df, rep
