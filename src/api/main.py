"""CareConnect scoring API (FastAPI) - v2.

Run locally:   uvicorn src.api.main:app --port 8000        (docs at /docs)
Docker:        docker compose up api

Endpoints
* GET  /health      liveness + whether model and policy are loaded
* GET  /model_info  version, operating point, configured (not approved) fairness policy, intended / prohibited use
* POST /predict     score one appointment; returns risk, outreach decision under the configured (not approved) policy,
                    plain-language reason codes, and the explanation-store record id
* GET  /metrics     Prometheus-format counters (requests, flags, errors, latency)

Safeguards: typed and range-checked inputs (422 on invalid data), optional API key
(X-API-Key; set CARECONNECT_API_KEY), per-key rate limit, no raw patient features in logs.
Every prediction is written to an append-only explanation store (logs/explanations.jsonl) with model
version, policy, score, decision, random draw and reasons, keyed by request id, for audit.
"""
import hashlib
import json
import os
import sys
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from careconnect import config  # noqa: E402
from careconnect.fairness.policy import decide  # noqa: E402
from careconnect.pipeline.features import age_band  # noqa: E402

MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "artifacts" / "xgboost.joblib"))
OP_PATH = Path(os.environ.get("OP_PATH", ROOT / "artifacts" / "operating_point.json"))
POLICY_PATH = Path(os.environ.get("POLICY_PATH", ROOT / "artifacts" / "fairness_policy.json"))
STORE = Path(os.environ.get("EXPLANATION_STORE", ROOT / "logs" / "explanations.jsonl"))
API_KEY = os.environ.get("CARECONNECT_API_KEY")          # unset = open (local development only)
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MIN", "120"))

PLAIN = {"lead_time_days": "Days between booking and appointment", "Age": "Patient age",
         "prior_noShow_rate": "Share of past appointments missed", "prior_noShows": "Number of past missed appointments",
         "prior_appointments": "Number of past appointments", "SMS_received": "Reminder SMS status",
         "Neighbourhood": "Neighbourhood", "age_band": "Age group", "Scholarship": "Welfare scholarship status",
         "appointment_month": "Month of appointment", "appointment_dow": "Day of week", "Gender": "Gender",
         "Alcoholism": "Alcoholism flag", "Diabetes": "Diabetes flag", "Hipertension": "Hypertension flag",
         "Handcap": "Disability level"}
STATE = {"model": None, "op": {"flag_threshold": 0.25, "high_threshold": 0.40}, "policy": None, "hashes": {}}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
COUNTERS = defaultdict(float)
HITS: dict = defaultdict(deque)


@asynccontextmanager
async def lifespan(app):
    if MODEL_PATH.exists():
        STATE["model"] = joblib.load(MODEL_PATH)
        STATE["hashes"]["model_sha256"] = _sha256(MODEL_PATH)
    if OP_PATH.exists():
        STATE["op"] = json.load(open(OP_PATH))
    if POLICY_PATH.exists():
        STATE["policy"] = json.load(open(POLICY_PATH))
        STATE["hashes"]["policy_sha256"] = _sha256(POLICY_PATH)
    yield


app = FastAPI(title="CareConnect Predictive Engagement API", version=config.MODEL_VERSION, lifespan=lifespan,
              description="No-show risk scoring with reason codes. Decision support only: a person decides every action.")


class Appointment(BaseModel):
    Gender: Literal["F", "M"]
    Age: int = Field(ge=0, le=100)
    Neighbourhood: str = Field(min_length=1, max_length=60)
    Scholarship: Literal[0, 1]
    Hipertension: Literal[0, 1] = 0
    Diabetes: Literal[0, 1] = 0
    Alcoholism: Literal[0, 1] = 0
    Handcap: int = Field(0, ge=0, le=4)
    SMS_received: Literal[0, 1]
    lead_time_days: int = Field(ge=0, le=365)
    appointment_dow: int = Field(ge=0, le=6)
    appointment_month: int = Field(ge=1, le=12)
    prior_appointments: int = Field(ge=0, le=500)
    prior_noShows: int = Field(ge=0, le=500)

    @model_validator(mode="after")
    def history_consistent(self):
        if self.prior_noShows > self.prior_appointments:
            raise ValueError("prior_noShows cannot exceed prior_appointments")
        return self

    def frame(self) -> pd.DataFrame:
        d = self.model_dump()
        d["age_band"] = age_band(d["Age"])                     # always derived server-side
        d["prior_noShow_rate"] = d["prior_noShows"] / d["prior_appointments"] if d["prior_appointments"] else 0.0
        return pd.DataFrame([d])[config.FEATURES]


class PredictionRequest(BaseModel):
    features: Appointment
    request_id: Optional[str] = Field(None, max_length=64, description="Caller's id; generated if absent")
    return_reasons: bool = True


class PredictionResponse(BaseModel):
    request_id: str
    no_show_probability: float
    score_tier: Literal["low", "medium", "high"] = Field(
        description="Model risk-score band only (low < flag cut-off <= medium < 0.40 <= high). It is NOT the outreach decision.")
    flagged_for_outreach: bool = Field(description="Decision of the configured (not approved) fairness policy for this request")
    flag_probability: float = Field(description="Policy's probability of flagging this score in this age group")
    decision_randomised: bool = Field(description="True when 0 < flag_probability < 1: the decision came from a logged random draw")
    decision_explanation: str
    policy: str
    reason_codes: List[str] = []
    model_version: str
    note: str = "Decision support only. A human coordinator decides what, if anything, to offer."


def auth(request: Request, x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        COUNTERS["auth_failures"] += 1
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")
    who = x_api_key or (request.client.host if request.client else "anon")
    q, now = HITS[who], time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        COUNTERS["rate_limited"] += 1
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    q.append(now)


def reasons(df: pd.DataFrame, k: int = 3) -> List[str]:
    import xgboost as xgb
    model = STATE["model"]
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    names = list(pre.get_feature_names_out())
    c = clf.get_booster().predict(xgb.DMatrix(pre.transform(df), feature_names=names), pred_contribs=True)[0][:-1]
    agg = {}
    for n, v in zip(names, c):
        b = n.split("__", 1)[1]
        g = next((x for x in config.CATEGORICAL if b.startswith(x + "_")), b)
        agg[g] = agg.get(g, 0.0) + float(v)
    top = sorted(agg.items(), key=lambda x: -abs(x[1]))[:k]
    return [f"{PLAIN.get(f, f)}: {'raises' if v > 0 else 'lowers'} risk" for f, v in top]


def store(record: dict):
    STORE.parent.mkdir(parents=True, exist_ok=True)
    with open(STORE, "a") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": STATE["model"] is not None, "policy_loaded": STATE["policy"] is not None,
            "model_version": config.MODEL_VERSION}


@app.get("/model_info")
def model_info():
    pol = STATE["policy"]
    return {"model_name": config.REGISTERED_MODEL, "model_version": config.MODEL_VERSION,
            "model_type": "XGBoost classifier in an sklearn pipeline", "target": config.TARGET,
            "operating_point": STATE["op"], "fairness_policy": pol["name"] if pol else "single cut-off",
            "policy_status": pol.get("status") if pol else None, **STATE["hashes"],
            "intended_use": "Identify appointments at higher risk of a missed visit so a person can offer support",
            "non_intended_use": "Deny care, penalise patients, or automate scheduling without human review",
            "fairness_constraint": "Equalized-odds gap <= 0.05 across age groups, gender and scholarship (Charter Principle 2)"}


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(auth)])
def predict(req: PredictionRequest):
    t0 = time.time()
    if STATE["model"] is None:
        COUNTERS["errors"] += 1
        raise HTTPException(status_code=503, detail="Model not loaded. Check MODEL_PATH.")
    df = req.features.frame()
    p = float(STATE["model"].predict_proba(df)[0, 1])
    rid = req.request_id or uuid.uuid4().hex
    op, pol = STATE["op"], STATE["policy"]
    if pol:
        d = decide(p, df["age_band"].iloc[0], rid, pol)
    else:
        d = {"flag_probability": float(p >= op["flag_threshold"]), "flagged": p >= op["flag_threshold"], "randomised": False, "draw": None}
    tier = "high" if p >= op["high_threshold"] else ("medium" if p >= op["flag_threshold"] else "low")
    if d["randomised"]:
        why = (f"Score tier '{tier}'. In this age group the policy flags this score with probability "
               f"{d['flag_probability']:.0%}; the logged draw {'fell inside' if d['flagged'] else 'fell outside'} that probability.")
    else:
        why = f"Score tier '{tier}'. The policy {'always' if d['flagged'] else 'never'} flags this score in this age group."
    rc = reasons(df) if req.return_reasons else []
    canon = json.dumps(req.features.model_dump(), sort_keys=True)
    store({"ts": datetime.now(timezone.utc).isoformat(), "request_id": rid, "model_version": config.MODEL_VERSION,
           "policy": pol["name"] if pol else "single cut-off", "score": round(p, 6), "score_tier": tier,
           "model_sha256": STATE["hashes"].get("model_sha256"), "policy_sha256": STATE["hashes"].get("policy_sha256"),
           "flag_probability": d["flag_probability"], "flagged": d["flagged"], "random_draw": d["draw"],
           "reason_codes": rc, "input_sha256": hashlib.sha256(canon.encode()).hexdigest(),
           "age_band": df["age_band"].iloc[0]})
    COUNTERS["predictions"] += 1
    COUNTERS["flags"] += int(d["flagged"])
    COUNTERS["latency_seconds_sum"] += time.time() - t0
    return PredictionResponse(request_id=rid, no_show_probability=round(p, 4), score_tier=tier,
                              flagged_for_outreach=d["flagged"], flag_probability=round(d["flag_probability"], 4),
                              decision_randomised=d["randomised"], decision_explanation=why,
                              policy=pol["name"] if pol else "single cut-off",
                              reason_codes=rc, model_version=config.MODEL_VERSION)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    lines = [f"careconnect_{k}_total {v}" if not k.endswith("_sum") else f"careconnect_{k} {v}" for k, v in COUNTERS.items()]
    return "\n".join(lines) + "\n"
