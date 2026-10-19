"""BIAS CHECK: pre-training representation and label-rate audit (Charter Principle 2, goal G3).

Gating checks (any FAIL blocks publication of the modelling table) are the five Module 3 checks.
Informational checks report label-rate differences by age band and scholarship: these are not
data defects, but they explain why equal flag rates across age groups are not achievable by an
accurate model, which matters for how the fairness limit is defined.
"""
import pandas as pd

from careconnect import config
from careconnect.pipeline.audit import audit


def run_bias_checks(df: pd.DataFrame, min_n: int = config.BIAS_MIN_N) -> dict:
    checks = []

    def add(name, passed, value, threshold, gating=True):
        checks.append({"check": name, "passed": bool(passed), "value": value, "threshold": threshold, "gating": gating})

    g = df["Gender"].value_counts()
    add("gender_representation", (g / len(df)).max() <= 0.75 and g.min() >= min_n,
        {k: int(v) for k, v in g.items()}, f"no group >75%; each >= {min_n}")
    a = df["age_band"].value_counts()
    add("age_band_representation", len(a) == len(config.AGE_BANDS) and a.min() >= min_n,
        {k: int(v) for k, v in a.items()}, f"all {len(config.AGE_BANDS)} bands populated; each >= {min_n}")
    top = df["Neighbourhood"].value_counts(normalize=True)
    add("neighbourhood_concentration", top.max() <= 0.20, {"top": top.index[0], "share": round(float(top.max()), 4)},
        "no neighbourhood > 20% of records")
    s = df["Scholarship"].value_counts()
    add("socioeconomic_representation", s.min() >= min_n, {str(k): int(v) for k, v in s.items()}, f"each level >= {min_n}")
    rates = df.groupby("Gender")["noshow_flag"].mean()
    ratio = float(rates.min() / rates.max())
    add("label_rate_parity_gender", 0.80 <= ratio <= 1.25, round(ratio, 4), "ratio within 0.80-1.25")

    # Informational: base-rate differences that any accurate model will reflect
    for col in ["age_band", "Scholarship"]:
        r = df.groupby(col)["noshow_flag"].mean()
        add(f"label_rate_by_{col}", True, {str(k): round(float(v), 4) for k, v in r.items()},
            "informational (not a gate)", gating=False)
        checks[-1]["min_max_ratio"] = round(float(r.min() / r.max()), 4)

    overall = all(c["passed"] for c in checks if c["gating"])
    rep = {"overall_pass": overall, "checks": checks, "n": len(df)}
    audit("BIAS_CHECK", overall_pass=overall, failed=[c["check"] for c in checks if c["gating"] and not c["passed"]])
    return rep
