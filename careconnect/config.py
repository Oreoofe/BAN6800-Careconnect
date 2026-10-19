"""Central configuration for the CareConnect capstone build.

Rebuilt for the BAN 6800 capstone (October 2026). The original Module 3 and Module 4
source packages were not preserved in any repository, so this package re-implements the
pipeline and modelling steps described in those modules, with one documented correction:
prior-attendance features now use only appointments whose outcome was known at booking time.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
LOGS = ROOT / "logs"
DOCS = ROOT / "docs"

RAW_CSV = DATA_RAW / "noshowappointments-kagglev2-may-2016.csv"
SAMPLE_CSV = DATA_RAW / "sample_no_show.csv"           # synthetic smoke-test file (tools/make_sample_data.py)
MODELING_TABLE = DATA_PROCESSED / "careconnect_modeling.parquet"

# Public copy of the Kaggle "Medical Appointment No Shows" file, used for automated runs.
# The checksum pins the exact bytes, so a changed or tampered copy stops the pipeline.
RAW_URL = ("https://raw.githubusercontent.com/ksatola/Medical-Appointments-No-Shows/master/"
           "noshowappointments-kagglev2-may-2016.csv")
RAW_SHA256 = "9132d3e7d0246617df9041d3764f20ad6f08e7b0d9f0997fa254fc5e52eda27d"

RAW_COLUMNS = ["PatientId", "AppointmentID", "Gender", "ScheduledDay", "AppointmentDay", "Age",
               "Neighbourhood", "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handcap",
               "SMS_received", "No-show"]

# Pseudonymisation key. Set CARECONNECT_SALT in production (secret store); the default is for local runs only.
SALT = os.environ.get("CARECONNECT_SALT", "local-dev-salt-not-for-production")

AGE_MIN, AGE_MAX = 0, 100
AGE_BANDS = [("child", 0, 12), ("teen", 13, 18), ("young_adult", 19, 35),
             ("adult", 36, 55), ("senior", 56, 75), ("elderly", 76, 100)]
K_ANON_MIN = 30          # neighbourhoods with fewer records are grouped as OTHER
BIAS_MIN_N = 500         # minimum records per audited subgroup

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = int(os.environ.get("CARECONNECT_CV_FOLDS", "5"))
RECALL_TARGET = 0.65     # Module 4 acceptance target used to set the outreach cut-off
HIGH_TIER = 0.40         # score at or above this = high-risk tier
FAIRNESS_LIMIT = 0.05    # Ethical Charter Principle 2 (as clarified: error-rate gap)
DI_MIN = 0.80            # four-fifths rule for the disparate-impact ratio

TARGET = "noshow_flag"
NUMERIC = ["Age", "lead_time_days", "prior_appointments", "prior_noShows", "prior_noShow_rate",
           "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handcap", "SMS_received",
           "appointment_dow", "appointment_month"]
CATEGORICAL = ["Gender", "Neighbourhood", "age_band"]
FEATURES = ["Gender", "Age", "Neighbourhood", "Scholarship", "Hipertension", "Diabetes", "Alcoholism",
            "Handcap", "SMS_received", "lead_time_days", "appointment_dow", "appointment_month",
            "age_band", "prior_appointments", "prior_noShows", "prior_noShow_rate"]
# Features removed in the data-minimisation ablation (protected or sensitive health attributes)
SENSITIVE_INPUTS = ["Gender", "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handcap", "age_band"]

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlruns' / 'mlflow.db').as_posix()}")
MLFLOW_EXPERIMENT = "careconnect_capstone"
REGISTERED_MODEL = "careconnect_noshow_xgb"
MODEL_VERSION = "2.0.0"   # SemVer; 1.x = Module 4/5 model (contained the prior-history leak)
