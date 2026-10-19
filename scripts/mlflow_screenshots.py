"""Capture real screenshots of the MLflow UI (experiment runs + model registry) for the report.

    python scripts/mlflow_screenshots.py            # needs: pip install playwright && playwright install chromium

Starts `mlflow ui` on the project's tracking database, opens the experiment and registered-model
pages in headless Chromium, and saves PNGs to docs/evidence/. Nothing is mocked: if the server or a
page fails to load, the script exits with an error instead of producing an image.
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from careconnect import config  # noqa: E402

OUT = ROOT / "docs" / "evidence"
PORT = 5055


def wait(url, secs=90):
    for _ in range(secs):
        try:
            if urllib.request.urlopen(url, timeout=2).status == 200:
                return
        except Exception:
            time.sleep(1)
    raise SystemExit(f"MLflow UI did not start at {url}")


def main():
    import mlflow
    from playwright.sync_api import sync_playwright
    OUT.mkdir(parents=True, exist_ok=True)
    srv = subprocess.Popen([sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", config.MLFLOW_URI,
                            "--host", "127.0.0.1", "--port", str(PORT)], cwd=ROOT)
    base = f"http://127.0.0.1:{PORT}"
    try:
        wait(f"{base}/health")
        mlflow.set_tracking_uri(config.MLFLOW_URI)
        exp = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
        pages = {"mlflow_experiment_runs.png": f"{base}/#/experiments/{exp.experiment_id}",
                 "mlflow_model_registry.png": f"{base}/#/models/{config.REGISTERED_MODEL}",
                 "mlflow_registered_models.png": f"{base}/#/models"}
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1600, "height": 900})
            for name, url in pages.items():
                pg.goto(url, wait_until="networkidle")
                time.sleep(4)
                pg.screenshot(path=str(OUT / name), full_page=True)
                print("saved", OUT / name)
            b.close()
    finally:
        srv.terminate()


if __name__ == "__main__":
    main()
