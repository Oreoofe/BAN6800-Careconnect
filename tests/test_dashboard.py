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
    b = M[M.get("adopted", "B_equalized_odds")]
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
