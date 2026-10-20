"""Render every dashboard page headlessly (Streamlit AppTest) and check displayed numbers against artifacts."""
import json

import pytest

from careconnect import config

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
pytest.importorskip("xgboost")
pytestmark = pytest.mark.skipif(not (config.ARTIFACTS / "analysis.json").exists(), reason="artifacts not built")
APP = str(config.ROOT / "src" / "app.py")


def _app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def _text(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info] + \
            [t.value for t in at.title] + [s.value for s in at.success] + [w.value for w in at.warning]
    for df in at.dataframe:
        parts.append(df.value.to_string())
    return "\n".join(map(str, parts))


def test_every_page_renders():
    at = _app()
    pages = list(at.sidebar.radio[0].options)
    assert len(pages) == 7
    for page in pages:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"{page}: {at.exception}"


def test_overview_numbers_match_artifacts():
    A = json.load(open(config.ARTIFACTS / "analysis.json"))
    op = A["operating"]["xgboost"]["at_recall_target"]
    txt = _text(_app())
    assert f"{op['roc_auc']:.2f}" in txt
    assert f"{op['recall']:.0%}" in txt
    assert f"{A['descriptive']['n']:,}" in txt


def test_fairness_page_matches_mitigation_artifact():
    M = json.load(open(config.ARTIFACTS / "mitigation.json"))
    at = _app()
    page = next(p for p in at.sidebar.radio[0].options if p.startswith("5"))
    at.sidebar.radio[0].set_value(page).run()
    txt = _text(at)
    b = M[M.get("configured", M.get("adopted", "B_equalized_odds"))]
    assert f"{b['age_band']['eod']:.3f}" in txt
    assert f"{b['flag_rate']:.0%}" in txt


def test_monitoring_page_shows_replay_label():
    if not (config.ARTIFACTS / "monitoring" / "monitoring.json").exists():
        pytest.skip("monitoring not built")
    at = _app()
    page = next(p for p in at.sidebar.radio[0].options if p.startswith("7"))
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception
    assert "replay" in _text(at).lower()


def test_build_identity_and_new_sections():
    import hashlib
    at = _app()
    side = "\n".join(str(c.value) for c in at.sidebar.caption)
    assert hashlib.sha256((config.ARTIFACTS / "xgboost.joblib").read_bytes()).hexdigest()[:12] in side
    pages = list(at.sidebar.radio[0].options)
    if (config.ARTIFACTS / "policy_comparison.json").exists():
        at.sidebar.radio[0].set_value(next(p for p in pages if p.startswith("5"))).run()
        assert "Comparison of decision policies" in _text(at)
    if (config.ARTIFACTS / "explanations.json").exists():
        at.sidebar.radio[0].set_value(next(p for p in pages if p.startswith("3"))).run()
        assert not at.exception and "how far to trust it" in _text(at)
    at.sidebar.radio[0].set_value(next(p for p in pages if p.startswith("4"))).run()
    assert not at.exception and "Model score tier" in _text(at)


def test_committee_disclosures_rendered():
    pc = config.ARTIFACTS / "policy_comparison.json"
    mon = config.ARTIFACTS / "monitoring" / "monitoring.json"
    if not pc.exists() or "criteria" not in json.load(open(pc)):
        pytest.skip("cycle-3 artifacts not built")
    at = _app()
    pages = list(at.sidebar.radio[0].options)
    at.sidebar.radio[0].set_value(next(p for p in pages if p.startswith("5"))).run()
    t = _text(at) + "\n".join(str(e.value) for e in at.error)
    assert not at.exception
    assert "No evaluated policy reliably met" in t and "unrelated to predicted risk" in t and "Equal opportunity" in t
    if mon.exists() and "summary" in json.load(open(mon)):
        at.sidebar.radio[0].set_value(next(p for p in pages if p.startswith("7"))).run()
        assert not at.exception and "every monitored time window" in _text(at)


def test_recommendation_and_deployment_status_rendered():
    g = config.ARTIFACTS / "promotion_decision.json"
    if not g.exists() or "recommendation" not in json.load(open(g)):
        pytest.skip("cycle-4 gate output not built")
    at = _app()
    t = _text(at) + "\n".join(str(e.value) for e in at.error) + "\n".join(str(c.value) for c in at.sidebar.caption)
    assert not at.exception
    assert "Do not deploy until a better model" in t and "No public API exists" in t and "v1" in t


def test_monitoring_page_shows_governing_module4_rule():
    mon = config.ARTIFACTS / "monitoring" / "monitoring.json"
    if not mon.exists() or "module4_rule" not in json.load(open(mon)):
        pytest.skip("cycle-5 monitoring not built")
    M = json.load(open(mon))
    at = _app()
    at.sidebar.radio[0].set_value(next(p for p in at.sidebar.radio[0].options if p.startswith("7"))).run()
    t = _text(at) + "\n".join(str(e.value) for e in at.error)
    assert not at.exception
    assert "Governing rule: original Module 4 §7" in t and "not the governing rule" in t
    assert f"Governing status over the replay: {M['summary']['module4_s7_weekly']['governing_status'].upper()}" in t
