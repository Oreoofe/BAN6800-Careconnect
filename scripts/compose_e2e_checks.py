"""End-to-end checks against the running Docker Compose services (standard library only; run on the CI host by the compose-e2e job in build.yml).

    python3 scripts/compose_e2e_checks.py <stage>     stage: pipeline | mlflow | monitoring | api

Each stage appends its results to compose-e2e/report.json and exits non-zero if any of its checks fail, so
every failure is visible in the workflow. Nothing here starts or deploys anything; the workflow does that.
"""
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "compose-e2e"
REPORT = OUT / "report.json"


def get(url, data=None, timeout=10):
    req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def wait(url, tries=60, pause=3):
    last = None
    for _ in range(tries):
        try:
            return get(url)
        except Exception as e:          # noqa: BLE001 - report the last error if the service never comes up
            last = e
            time.sleep(pause)
    raise RuntimeError(f"{url} not reachable: {last}")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(stage):
    OUT.mkdir(exist_ok=True)
    rep = json.loads(REPORT.read_text()) if REPORT.exists() else {"stages": {}}
    checks = []

    def check(name, ok, detail=None):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    A = ROOT / "artifacts"
    if stage == "pipeline":
        V = json.loads((A / "verification_report.json").read_text())
        R = json.loads((A / "model_results.json").read_text())
        G = json.loads((A / "promotion_decision.json").read_text())
        check("pipeline container produced the model, policy, monitoring and documents",
              all((A / f).exists() for f in ["xgboost.joblib", "fairness_policy.json", "monitoring/monitoring.json"])
              and all((ROOT / "docs" / f).exists() for f in ["model_card.md", "committee_decision_pack.md"]))
        check("independent verification inside the container: 0 FAIL", V["summary"]["fail"] == 0, V["summary"])
        check("container build identity recorded (model git_sha == commit)", R.get("git_sha") == V["build"]["commit"],
              (R.get("git_sha"), V["build"]["commit"]))
        check("promotion gate in container: NOT ELIGIBLE, no automatic promotion",
              G["decision"].startswith("NOT ELIGIBLE") and G["automatic_promotion"] is False, G["blocking_failures"])
        rep["checksums"] = {"model": sha(A / "xgboost.joblib"), "policy": sha(A / "fairness_policy.json"),
                            "committee_pack": sha(ROOT / "docs" / "committee_decision_pack.md")}
        rep["verification_summary"] = V["summary"]
    elif stage == "mlflow":
        code, body = wait("http://localhost:5000/health")
        check("MLflow server (compose service 'mlflow') answers /health", code == 200, body.decode()[:40])
        _, b = get("http://localhost:5000/api/2.0/mlflow/experiments/get-by-name?experiment_name=careconnect_capstone")
        exp = json.loads(b)["experiment"]["experiment_id"]
        _, b = get("http://localhost:5000/api/2.0/mlflow/runs/search", {"experiment_ids": [exp], "max_results": 100})
        runs = json.loads(b).get("runs", [])
        names = sorted(r["info"].get("run_name", "") for r in runs)
        check("MLflow REST API lists >= 7 runs from the pipeline container", len(runs) >= 7, names)
        _, b = get("http://localhost:5000/api/2.0/mlflow/registered-models/get?name=careconnect_noshow_xgb")
        rm = json.loads(b)["registered_model"]
        aliases = {a["alias"]: a["version"] for a in rm.get("aliases", [])}
        check("registered model has alias 'candidate' and no 'production' alias", "candidate" in aliases and "production" not in aliases, aliases)
        rep["mlflow"] = {"experiment_id": exp, "runs": names, "aliases": aliases}
    elif stage == "monitoring":
        code, body = wait("http://localhost:8090/monitoring.json")
        M = json.loads(body)
        check("monitoring service (compose service 'monitoring') serves monitoring.json", code == 200)
        check("served monitoring.json uses the governing Module 4 s.7 weekly rule", M.get("governing_rule") == "module4_s7_weekly",
              M.get("summary", {}).get("module4_s7_weekly", {}).get("counts"))
        try:
            code2, b2 = get("http://localhost:8090/evidently_drift_report.html")
            check("monitoring service serves the Evidently drift report", code2 == 200 and len(b2) > 10000, len(b2))
        except urllib.error.HTTPError as e:
            check("monitoring service serves the Evidently drift report", False, str(e))
        rep["monitoring"] = {"governing_status": M["summary"]["module4_s7_weekly"]["governing_status"],
                             "counts": M["summary"]["module4_s7_weekly"]["counts"]}
    elif stage == "api":
        code, body = wait("http://localhost:8000/health")
        check("API container healthy", code == 200, json.loads(body))
        _, b = get("http://localhost:8000/model_info")
        info = json.loads(b)
        check("API container serves the model and policy produced by the pipeline container",
              info.get("model_sha256") == sha(A / "xgboost.joblib") and info.get("policy_sha256") == sha(A / "fairness_policy.json"),
              (info.get("model_sha256", "")[:12], info.get("policy_sha256", "")[:12]))
        code, _ = wait("http://localhost:8501/_stcore/health")
        check("dashboard container healthy", code == 200)
    else:
        raise SystemExit(f"unknown stage {stage}")
    rep["stages"][stage] = checks
    REPORT.write_text(json.dumps(rep, indent=1))
    for c in checks:
        print(c["status"], c["check"], "" if c["detail"] is None else c["detail"])
    return 0 if all(c["status"] == "PASS" for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
