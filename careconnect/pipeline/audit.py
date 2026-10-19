"""Append-only JSONL audit log and table I/O helpers shared by every pipeline stage."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone

import pandas as pd

from careconnect import config


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def config_hash() -> str:
    keys = ["AGE_MIN", "AGE_MAX", "K_ANON_MIN", "BIAS_MIN_N", "RANDOM_STATE", "TEST_SIZE", "RECALL_TARGET"]
    return hashlib.sha256(json.dumps({k: getattr(config, k) for k in keys}, sort_keys=True).encode()).hexdigest()[:12]


def audit(stage: str, **details) -> dict:
    """Append one immutable event. No patient-level values are ever written here."""
    config.LOGS.mkdir(parents=True, exist_ok=True)
    event = {"ts": datetime.now(timezone.utc).isoformat(), "stage": stage, "actor": "careconnect-pipeline",
             "git_sha": git_sha(), "config_hash": config_hash(), **details}
    with open(config.LOGS / "audit.log", "a") as f:
        f.write(json.dumps(event, default=str) + "\n")
    return event


def write_table(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError:                      # pyarrow missing: fall back to pickle with the same stem
        df.to_pickle(path.with_suffix(".pkl"))


def read_table(path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.read_pickle(path.with_suffix(".pkl"))
