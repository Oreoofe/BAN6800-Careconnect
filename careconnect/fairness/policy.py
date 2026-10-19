"""The outreach policy actually applied to scores (Option B: equalized-odds post-processing).

Fairlearn's ThresholdOptimizer learns, for each age group, a rule of the form
    flag with probability  p_ignore * c + (1 - p_ignore) * (p0 * [score op0 t0] + p1 * [score op1 t1])
The fitted rule is exported to artifacts/fairness_policy.json so that the API and dashboard apply
exactly the same decision without depending on Fairlearn, and so the rule can be read and audited.

Where the rule is randomised, the random draw is derived from a hash of the request identifier,
so every decision is reproducible from the audit log.
"""
import hashlib
import json

import numpy as np

from careconnect import config

POLICY_PATH = config.ARTIFACTS / "fairness_policy.json"


def _op(score, operator, threshold):
    return (score > threshold) if operator == ">" else (score < threshold)


def flag_probability(score, group: str, policy: dict) -> float:
    rule = policy["groups"].get(group)
    if rule is None:                     # unseen group: fall back to the single cut-off
        return float(score >= policy["fallback_threshold"])
    inner = rule["p0"] * _op(score, *rule["op0"]) + rule["p1"] * _op(score, *rule["op1"])
    return float(rule.get("p_ignore", 0.0) * rule.get("prediction_constant", 0.0) + (1 - rule.get("p_ignore", 0.0)) * inner)


def uniform_from(request_id: str) -> float:
    h = hashlib.sha256(str(request_id).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def decide(score, group: str, request_id: str, policy: dict) -> dict:
    pf = flag_probability(score, group, policy)
    u = uniform_from(request_id)
    return {"flag_probability": pf, "flagged": bool(u < pf), "randomised": 0.0 < pf < 1.0, "draw": u}


def export_threshold_optimizer(to, groups, fallback_threshold) -> dict:
    """Convert a fitted fairlearn ThresholdOptimizer into the plain-JSON rule used above."""
    d = to.interpolated_thresholder_.interpolation_dict
    pol = {"name": "Option B - equalized odds (Fairlearn ThresholdOptimizer)", "sensitive_feature": "age_band",
           "fallback_threshold": fallback_threshold, "groups": {}}
    for g in groups:
        r = d[g]
        pol["groups"][str(g)] = {
            "p0": float(r["p0"]), "op0": [r["operation0"].operator, float(r["operation0"].threshold)],
            "p1": float(r["p1"]), "op1": [r["operation1"].operator, float(r["operation1"].threshold)],
            "p_ignore": float(r.get("p_ignore", 0.0)), "prediction_constant": float(r.get("prediction_constant", 0.0))}
    return pol


def load_policy(path=POLICY_PATH) -> dict | None:
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return None


def apply_policy(scores, groups, policy, seed=config.RANDOM_STATE):
    """Vectorised version for evaluation: expected flag probabilities and one seeded realisation."""
    pf = np.array([flag_probability(s, g, policy) for s, g in zip(scores, groups)])
    draws = np.random.default_rng(seed).random(len(pf))
    return pf, (draws < pf).astype(int)
