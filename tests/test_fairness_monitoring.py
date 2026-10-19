"""Fairness metric, policy and drift tests (toy data; Fairlearn cross-check runs when installed)."""
import numpy as np
import pytest

from careconnect.fairness.metrics import fairness_report
from careconnect.fairness.policy import decide, flag_probability, uniform_from
from careconnect.monitoring.monitor import psi

Y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
YH = np.array([1, 0, 1, 1, 0, 0, 0, 0])
G = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])


def test_metric_definitions():
    r = fairness_report(Y, YH, G)
    assert r["groups"]["a"]["flag_rate"] == 0.75 and r["groups"]["b"]["flag_rate"] == 0.0
    assert r["dpd"] == pytest.approx(0.75)
    assert r["tpr_gap"] == pytest.approx(1.0) and r["fpr_gap"] == pytest.approx(0.5)
    assert r["eod"] == pytest.approx(1.0)
    assert r["di_ratio"] == 0.0 and not r["pass_di"]


def test_matches_fairlearn():
    pytest.importorskip("fairlearn")
    from careconnect.fairness.metrics import crosscheck_with_fairlearn
    rng = np.random.default_rng(0)
    y, yh, g = rng.integers(0, 2, 2000), rng.integers(0, 2, 2000), rng.choice(list("abc"), 2000)
    assert crosscheck_with_fairlearn(y, yh, g)["match"]


POLICY = {"fallback_threshold": 0.25, "groups": {
    "x": {"p0": 1.0, "op0": [">", 0.3], "p1": 0.0, "op1": [">", 0.3]},
    "y": {"p0": 0.4, "op0": [">", 0.1], "p1": 0.6, "op1": [">", 0.2]}}}


def test_policy_rules():
    assert flag_probability(0.35, "x", POLICY) == 1.0
    assert flag_probability(0.25, "x", POLICY) == 0.0
    assert flag_probability(0.15, "y", POLICY) == pytest.approx(0.4)
    assert flag_probability(0.5, "unknown", POLICY) == 1.0      # fallback cut-off


def test_randomised_decisions_are_reproducible():
    a, b = decide(0.15, "y", "req-1", POLICY), decide(0.15, "y", "req-1", POLICY)
    assert a == b and a["randomised"]
    draws = [uniform_from(f"r{i}") for i in range(5000)]
    assert 0.47 < np.mean(draws) < 0.53


def test_psi():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 20000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.02
    assert psi(ref, rng.normal(1, 1, 5000)) > 0.2
    assert psi(np.array(["a"] * 50 + ["b"] * 50), np.array(["a"] * 90 + ["b"] * 10)) > 0.2
