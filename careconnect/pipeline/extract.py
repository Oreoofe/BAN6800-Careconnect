"""EXTRACT: obtain the raw CSV, verify its checksum, load it with an explicit schema."""
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

from careconnect import config
from careconnect.pipeline.audit import audit


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_raw(path: Path = config.RAW_CSV, url: str = config.RAW_URL) -> Path:
    """Download the public dataset if it is not already present, then verify its checksum."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    digest = sha256(path)
    if digest != config.RAW_SHA256:
        raise RuntimeError(f"Checksum mismatch for {path.name}: {digest} (expected {config.RAW_SHA256}). "
                           "The file differs from the version this project was validated on.")
    return path


def extract(path: Path | None = None, verify: bool = True) -> pd.DataFrame:
    path = Path(path) if path else (fetch_raw() if verify else config.RAW_CSV)
    df = pd.read_csv(path, dtype={"PatientId": "float64", "AppointmentID": "int64", "Gender": "string",
                                  "Neighbourhood": "string", "No-show": "string"})
    audit("EXTRACT", source=path.name, rows_out=len(df), cols=len(df.columns),
          sha256=sha256(path) if path.exists() else None)
    return df
