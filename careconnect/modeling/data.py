"""Load the modelling table and make the fixed train/test split used by every later step."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from careconnect import config
from careconnect.pipeline.audit import read_table


def load_table() -> pd.DataFrame:
    df = read_table(config.MODELING_TABLE)
    for c in config.CATEGORICAL + ["patient_id_hash"]:
        df[c] = df[c].astype(str)
    return df


def split(df: pd.DataFrame):
    """Stratified 80/20 split on appointments (seed 42), as in Modules 4-5. Returns full frames so
    that grouping keys and dates stay available for robustness checks and monitoring."""
    tr, te = train_test_split(df, test_size=config.TEST_SIZE, stratify=df[config.TARGET],
                              random_state=config.RANDOM_STATE)
    return tr, te


def xy(frame: pd.DataFrame, features=None):
    features = features or config.FEATURES
    return frame[features].copy(), frame[config.TARGET].astype(int).copy()


def preprocessor(features=None) -> ColumnTransformer:
    features = features or config.FEATURES
    num = [c for c in config.NUMERIC if c in features]
    cat = [c for c in config.CATEGORICAL if c in features]
    return ColumnTransformer([("num", StandardScaler(), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
