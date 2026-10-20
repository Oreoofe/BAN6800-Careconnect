"""API contract tests. Skipped automatically when the trained model or FastAPI is not available."""
import importlib
import json

import pytest

from careconnect import config

pytest.importorskip("fastapi")
pytest.importorskip("xgboost")
pytestmark = pytest.mark.skipif(not (config.ARTIFACTS / "xgboost.joblib").exists(), reason="model artifact not built")

GOOD = dict(Gender="F", Age=26, Neighbourhood="JARDIM DA PENHA", Scholarship=0, Hipertension=0, Diabetes=0, Alcoholism=0,
            Handcap=0, SMS_received=1, lead_time_days=35, appointment_dow=2, appointment_month=5,
            prior_appointments=4, prior_noShows=3)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("EXPLANATION_STORE", str(tmp_path / "store.jsonl"))
    monkeypatch.delenv("CARECONNECT_API_KEY", raising=False)
    import src.api.main as m
    importlib.reload(m)
    with TestClient(m.app) as c:
        yield c, tmp_path / "store.jsonl", m


def test_health_and_info(client):
    c, _, _ = client
    assert c.get("/health").json()["model_loaded"] is True
    info = c.get("/model_info").json()
    assert info["model_version"] == config.MODEL_VERSION and "Deny care" in info["non_intended_use"]
    import hashlib
    assert info["model_sha256"] == hashlib.sha256((config.ARTIFACTS / "xgboost.joblib").read_bytes()).hexdigest()


def test_predict_and_explanation_store(client):
    c, store, _ = client
    r = c.post("/predict", json={"features": GOOD, "request_id": "t-1"})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["no_show_probability"] <= 1 and len(body["reason_codes"]) == 3
    assert body["score_tier"] in {"low", "medium", "high"} and "tier" in body["decision_explanation"]
    assert (0 < body["flag_probability"] < 1) == body["decision_randomised"]
    rec = json.loads(store.read_text().splitlines()[-1])
    assert rec["request_id"] == "t-1" and rec["model_version"] == config.MODEL_VERSION
    assert "Age" not in rec and "Neighbourhood" not in rec          # no raw features stored
    again = c.post("/predict", json={"features": GOOD, "request_id": "t-1"}).json()
    assert again["flagged_for_outreach"] == body["flagged_for_outreach"]   # reproducible decision


@pytest.mark.parametrize("bad", [{"Age": -5}, {"Age": 140}, {"Gender": "X"}, {"lead_time_days": -1},
                                 {"prior_noShows": 9}, {"SMS_received": 3}])
def test_invalid_inputs_rejected(client, bad):
    c, _, _ = client
    assert c.post("/predict", json={"features": {**GOOD, **bad}}).status_code == 422


def test_api_key_enforced(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("CARECONNECT_API_KEY", "test-key")
    monkeypatch.setenv("EXPLANATION_STORE", str(tmp_path / "s.jsonl"))
    import src.api.main as m
    importlib.reload(m)
    with TestClient(m.app) as c:
        assert c.post("/predict", json={"features": GOOD}).status_code == 401
        assert c.post("/predict", json={"features": GOOD}, headers={"X-API-Key": "test-key"}).status_code == 200
