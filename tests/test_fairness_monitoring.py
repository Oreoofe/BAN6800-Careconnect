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


# ---- cycle 2: monitoring rule and deterministic policies (toy data, behaviour only) ----
def _toy(n, gap, shares, seed=1):
    rng = np.random.default_rng(seed)
    g = rng.choice(list(shares), n, p=list(shares.values()))
    y = (rng.random(n) < 0.2).astype(int)
    tpr = np.where(g == list(shares)[0], 0.85, 0.85 - gap)
    yh = np.where(y == 1, rng.random(n) < tpr, rng.random(n) < 0.5).astype(int)
    return y, yh, g


def test_fairness_rule_outcomes():
    from careconnect.monitoring.monitor import fairness_rule
    assert fairness_rule(*_toy(20000, 0.0, {"a": .5, "b": .5}))["status"] == "ok"
    assert fairness_rule(*_toy(20000, 0.20, {"a": .5, "b": .5}))["status"] == "halt"
    r = fairness_rule(*_toy(300, 0.0, {"a": .5, "b": .5}))
    assert r["status"] == "insufficient data" and r["eod"] is None      # no conclusion either way


def test_small_group_is_not_counted_as_passing():
    from careconnect.monitoring.monitor import fairness_rule
    y, yh, g = _toy(6000, 0.0, {"a": .49, "b": .49, "c": .02})
    r = fairness_rule(y, yh, g)
    assert "c" in r["groups_not_evaluated"] and "c" not in r["groups_evaluated"]


def test_equal_opportunity_cutoffs_equalise_tpr():
    from careconnect.fairness.policy_comparison import apply_cutoffs, tpr_cutoffs
    rng = np.random.default_rng(0)
    g = rng.choice(["x", "y"], 20000)
    y = (rng.random(20000) < 0.2).astype(int)
    s = np.clip(rng.normal(np.where(y == 1, .35, .2) + np.where(g == "y", .08, 0), .1), 0, 1)
    yh = apply_cutoffs(s, g, tpr_cutoffs(y, s, g, 0.75))
    tprs = [yh[(g == v) & (y == 1)].mean() for v in ["x", "y"]]
    assert all(abs(t - 0.75) < 0.01 for t in tprs)


# ---- cycle 5: original Module 4 s.7 weekly rule (governing): alert > 0.04, halt > 0.05, no minimum group size ----
def _m4_case(fnr_b):
    """Group a has FNR 0 and FPR 0; group b has FNR = fnr_b and FPR 0 (100 no-shows per group)."""
    y = np.array([1] * 100 + [0] * 100 + [1] * 100 + [0] * 100)
    g = np.array(["a"] * 200 + ["b"] * 200)
    yh = y.copy()
    yh[200:200 + int(round(fnr_b * 100))] = 0
    return y, yh, g


def test_module4_rule_thresholds():
    from careconnect.monitoring.monitor import module4_rule
    for fnr_b, want in [(0.03, "ok"), (0.04, "ok"), (0.045 + 0.005, "alert"), (0.05, "alert"), (0.06, "halt")]:
        y, yh, g = _m4_case(fnr_b)
        r = module4_rule(y, yh, {"age_band": g, "Gender": np.array(["F", "M"] * 200)})
        assert r["attributes"]["age_band"]["status"] == want, (fnr_b, r["attributes"]["age_band"])
    # boundaries are strict: exactly 0.04 is ok, exactly 0.05 is an alert, not a halt
    y, yh, g = _m4_case(0.05)
    assert module4_rule(y, yh, {"age_band": g})["status"] == "alert"


def test_module4_rule_any_governing_attribute_halts_and_undefined_is_not_a_pass():
    from careconnect.monitoring.monitor import module4_rule
    y, yh, g = _m4_case(0.0)
    sex = np.array(["F"] * 200 + ["M"] * 200)
    halt = module4_rule(y, yh, {"age_band": np.array(["x"] * 400), "Gender": sex})          # one age group only
    assert halt["attributes"]["age_band"]["status"] == "not evaluable"
    assert halt["status"] == "not evaluable"                                              # never "ok"
    yh2 = yh.copy()
    yh2[200:210] = 0                                                                        # Gender gap 0.10
    assert module4_rule(y, yh2, {"age_band": g, "Gender": sex})["status"] == "halt"
    # a group with no no-shows has an undefined FNR and is excluded from the comparison, not treated as fair
    y3 = np.array([0] * 50 + [1, 0] * 50)
    r = module4_rule(y3, y3.copy(), {"age_band": np.array(["c"] * 50 + ["d"] * 100)})
    assert r["attributes"]["age_band"]["groups_not_comparable"] == ["c"] and r["status"] == "not evaluable"


def test_module4_rule_attribution_does_not_blame_chance_when_expected_gap_breaches():
    from careconnect.monitoring.monitor import explain_module4, module4_rule
    y, yh, g = _m4_case(0.08)
    sex = np.array(["F", "M"] * 200)
    r = module4_rule(y, yh, {"age_band": g, "Gender": sex}, pmf=yh.astype(float))   # expected == realised (deterministic)
    _, attr = explain_module4(r, "ok")
    assert "expected disparity" in attr and "random" not in attr
    pmf = np.where(y == 1, 1.0, 0.0)                                                   # expected gap 0; breach is the draw
    r2 = module4_rule(y, yh, {"age_band": g, "Gender": sex}, pmf=pmf)
    _, attr2 = explain_module4(r2, "alert", "SMS_received", 2.7)
    assert "random draw" in attr2 and "SMS_received" in attr2
