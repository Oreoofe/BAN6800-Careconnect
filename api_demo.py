"""API demo: start the FastAPI service, call every endpoint, print and save the results.

Usage:  python api_demo.py            (after artifacts/xgboost.joblib exists)
Output: docs/api_demo_output.txt
"""
import json
import subprocess
import sys
import time

import requests

PROC = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.main:app", "--port", "8000", "--log-level", "warning"])
BASE = "http://127.0.0.1:8000"
lines = []


def show(label, obj):
    s = f"{label}\n{json.dumps(obj, indent=2) if not isinstance(obj, str) else obj}\n"
    print(s)
    lines.append(s)


try:
    for _ in range(60):
        try:
            if requests.get(f"{BASE}/health", timeout=1).ok:
                break
        except Exception:
            time.sleep(0.5)
    show("GET /health", requests.get(f"{BASE}/health").json())
    show("GET /model_info", requests.get(f"{BASE}/model_info").json())
    base = dict(Gender="F", Neighbourhood="JARDIM DA PENHA", Scholarship=0, Hipertension=0, Diabetes=0, Alcoholism=0,
                Handcap=0, appointment_dow=2, appointment_month=5)
    patients = {
        "A - long lead time, past misses": dict(base, Age=26, lead_time_days=35, SMS_received=1, prior_appointments=4, prior_noShows=3),
        "B - mid lead time, one past miss": dict(base, Age=45, lead_time_days=10, SMS_received=1, prior_appointments=2, prior_noShows=1),
        "C - same-day booking, reliable": dict(base, Age=62, lead_time_days=0, SMS_received=0, prior_appointments=5, prior_noShows=0),
    }
    for i, (name, f) in enumerate(patients.items()):
        r = requests.post(f"{BASE}/predict", json={"features": f, "request_id": f"demo-{i+1}"})
        show(f"POST /predict [{name}]", r.json())
    r = requests.post(f"{BASE}/predict", json={"features": {"Age": 40}})
    show("POST /predict with missing fields -> HTTP %d" % r.status_code, [e["loc"][-1] for e in r.json()["detail"]][:6])
    r = requests.post(f"{BASE}/predict", json={"features": dict(base, Age=-5, lead_time_days=10, SMS_received=1, prior_appointments=1, prior_noShows=3)})
    show("POST /predict with impossible values (age -5, 3 misses of 1 visit) -> HTTP %d" % r.status_code, [e["msg"] for e in r.json()["detail"]])
    show("GET /metrics", requests.get(f"{BASE}/metrics").text)
finally:
    PROC.terminate()
open("docs/api_demo_output.txt", "w").write("\n".join(lines))
