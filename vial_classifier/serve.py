# SERVE THE MODEL LOCALLY

# Serving runs the MLflow scoring server (the pyfunc model) as a subprocess on an internal port, with a small 
# FastAPI wrapper in front of it:

# - GET  `/health`      -> `{"status": "ok"}` once the scoring server is up, `"loading"` (503) while it loads, 
# `"error"` (503) if it died
# - POST `/invocations` -> forwarded unchanged to the scoring server
# - POST `/infer`       -> alias of `/invocations`
# - GET  `/version`     -> the model version and training timestamp of the served model
# - GET  `/drift`       -> drift detection (KS tests) over the requests logged by the scoring server

from vial_classifier.config import (
    model_name,
    experiment_name,
    artifact_path,
    model_version,
    artifact_dir,
    mlflow_state_dir,
    drift_baseline_stats_path,
    inference_log_path,
    model_results_path,
    drift_default_days,
    drift_ks_pvalue_threshold,
    export_dir,
    model_path,
    port,
    upstream_port,
    upstream_host
)

# Where the endpoint lives in production
PROD_ENDPOINT = "https://mlops-itu-jtk.ondigitalocean.app"# what does this do? It is never used...

print("Configuration loaded")

import os
import json
import time
from pathlib import Path

if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")

os.makedirs("artifacts", exist_ok=True)
os.makedirs("mlflow-state", exist_ok=True)

import pandas as pd
import numpy as np
from pprint import pprint
import warnings

warnings.filterwarnings('ignore')

import atexit
import subprocess
import sys
import urllib.error
import urllib.request

# FastAPI/uvicorn are installed as MLflow dependencies (MLflow's own scoring
# server uses them)
import uvicorn
from fastapi import FastAPI, Query, Request, Response

def _start_scoring_server():
    env = os.environ.copy()
    # The scoring server execs `uvicorn` through PATH; make sure the
    # interpreter's bin directory is on it
    interpreter_bin = os.path.dirname(os.path.abspath(sys.executable))
    env["PATH"] = os.pathsep.join([interpreter_bin, env.get("PATH", "")])
    proc = subprocess.Popen([
        sys.executable, "-m", "mlflow", "models", "serve",
        "-m", model_path,
        "-p", str(upstream_port),
        "--workers", "1",
        "--host", upstream_host,
        "--no-conda",
    ], env=env)
    atexit.register(lambda: proc.poll() is None and proc.terminate())
    return proc

scoring_proc = _start_scoring_server()
print("Scoring server starting on port", upstream_port)



# Drift detection functions
# Two-sample KS tests over the logged confidences and the predicted-class distribution, compared against the baseline statistics 
# written by the training notebook.

from datetime import datetime, timedelta, timezone
from scipy.stats import ks_2samp

def load_inference_logs(log_path=inference_log_path):
    """Load the inference log JSON file (may not exist yet)."""
    if not Path(log_path).exists():
        print(f"Warning: Inference log not found at {log_path}")
        return []
    try:
        with open(log_path, 'r') as f:
            logs = json.load(f)
            if not isinstance(logs, list):
                return []
            return logs
    except Exception as e:
        print(f"Warning: Failed to load inference log: {e}")
        return []

def filter_logs_by_time(logs, days=drift_default_days):
    """Only keep log entries from the last N days."""
    if not logs:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    filtered = []
    for log in logs:
        try:
            timestamp_str = log.get('timestamp', '')
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
                if timestamp >= cutoff:
                    filtered.append(log)
        except (ValueError, TypeError):
            filtered.append(log)
    return filtered

def load_baseline_stats():
    """Load the baseline statistics written during training."""
    if not Path(drift_baseline_stats_path).exists():
        print(f"Warning: Baseline stats not found at {drift_baseline_stats_path}")
        return None
    try:
        with open(drift_baseline_stats_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load baseline stats: {e}")
        return None

def calculate_confidence_metrics(confidences):
    confidences_array = np.array(confidences)
    return {
        'mean': float(np.mean(confidences_array)) if len(confidences_array) > 0 else 0.0,
        'std': float(np.std(confidences_array)) if len(confidences_array) > 0 else 0.0,
        'min': float(np.min(confidences_array)) if len(confidences_array) > 0 else 0.0,
        'max': float(np.max(confidences_array)) if len(confidences_array) > 0 else 0.0,
    }

def check_confidence_drift(current_confidences, baseline_stats):
    """KS test of the current confidence scores against the baseline."""
    if not baseline_stats or 'confidence_statistics' not in baseline_stats:
        return {'ks_statistic': 0.0, 'ks_pvalue': 1.0, 'drift_detected': False, 'threshold_ks': drift_ks_pvalue_threshold}

    baseline = baseline_stats['confidence_statistics']
    current_confidences_array = np.array(current_confidences)

    if len(current_confidences_array) < 2:
        return {'ks_statistic': 0.0, 'ks_pvalue': 1.0, 'drift_detected': False, 'threshold_ks': drift_ks_pvalue_threshold}

    baseline_mean = baseline['mean']
    baseline_std = baseline['std']
    current_mean = float(np.mean(current_confidences_array))

    baseline_samples = np.random.normal(baseline_mean, baseline_std, size=len(current_confidences_array))
    ks_statistic, ks_pvalue = ks_2samp(baseline_samples, current_confidences_array)

    mean_diff = abs(current_mean - baseline_mean)
    ks_drift = ks_pvalue < drift_ks_pvalue_threshold
    mean_drift = mean_diff > 0.1
    drift_detected = bool(ks_drift or mean_drift)

    return {
        'ks_statistic': float(ks_statistic),
        'ks_pvalue': float(ks_pvalue),
        'mean_diff': mean_diff,
        'current_mean': current_mean,
        'baseline_mean': baseline_mean,
        'drift_detected': drift_detected,
        'threshold_ks': drift_ks_pvalue_threshold,
        'threshold_mean': 0.1,
    }

def check_class_distribution_drift(predictions, baseline_stats):
    """KS test of the predicted class distribution against the baseline."""
    if not baseline_stats or 'class_distribution' not in baseline_stats:
        return {'ks_statistic': 0.0, 'ks_pvalue': 1.0, 'drift_detected': False, 'threshold_ks': drift_ks_pvalue_threshold}

    baseline = baseline_stats['class_distribution']
    baseline_total = baseline['total']
    current_total = len(predictions)

    if current_total == 0:
        return {'ks_statistic': 0.0, 'ks_pvalue': 1.0, 'drift_detected': False, 'threshold_ks': drift_ks_pvalue_threshold}

    baseline_prop_0 = baseline['class_0_count'] / baseline_total
    baseline_prop_1 = baseline['class_1_count'] / baseline_total
    baseline_dist = np.array([baseline_prop_0, baseline_prop_1])

    baseline_samples = np.random.choice([0, 1], size=baseline_total, p=baseline_dist)
    ks_statistic, ks_pvalue = ks_2samp(baseline_samples, predictions)

    ks_drift = ks_pvalue < drift_ks_pvalue_threshold
    drift_detected = bool(ks_drift)

    return {
        'ks_statistic': float(ks_statistic),
        'ks_pvalue': float(ks_pvalue),
        'drift_detected': drift_detected,
        'threshold_ks': drift_ks_pvalue_threshold,
    }

def detect_drift(logs, days=drift_default_days, baseline_stats=None):
    """Main drift detection: confidences + class distribution."""
    filtered_logs = filter_logs_by_time(logs, days=days)

    if not filtered_logs:
        print(f"Warning: No inference logs found in the last {days} days")
        return {
            'overall_drift_detected': False,
            'num_samples': 0,
            'confidence_drift': None,
            'class_distribution_drift': None,
            'error': 'No logs found',
        }

    confidences = []
    predictions = []
    for log in filtered_logs:
        if 'confidence' in log and log['confidence'] is not None:
            confidences.append(float(log['confidence']))
        if 'prediction' in log and log['prediction'] is not None:
            predictions.append(int(log['prediction']))

    num_samples = len(filtered_logs)

    confidence_results = check_confidence_drift(confidences, baseline_stats or {})
    class_dist_results = check_class_distribution_drift(predictions, baseline_stats or {})

    overall_drift = (
        confidence_results.get('drift_detected', False) or
        class_dist_results.get('drift_detected', False)
    )

    return {
        'overall_drift_detected': overall_drift,
        'num_samples': num_samples,
        'confidence_drift': confidence_results,
        'class_distribution_drift': class_dist_results,
        'confidence_stats': calculate_confidence_metrics(confidences),
        'class_distribution': {
            'class_0_count': predictions.count(0),
            'class_1_count': predictions.count(1),
            'total': len(predictions),
        },
    }

print("Drift detection functions defined")

app = FastAPI(docs_url=None, openapi_url=None, redoc_url=None)

def _upstream_health():
    """Return the scoring server's /health status code, None if unreachable."""
    try:
        with urllib.request.urlopen(
            f"http://{upstream_host}:{upstream_port}/health",
            timeout=1.0,
        ) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return None

@app.get("/health")
def health():
    if scoring_proc is not None and scoring_proc.poll() is not None:
        return Response(content='{"status": "error"}', status_code=503, media_type="application/json")
    if _upstream_health() == 200:
        return Response(content='{"status": "ok"}', media_type="application/json")
    return Response(content='{"status": "loading"}', status_code=503, media_type="application/json")

@app.get("/version")
def version():
    """Model version of the served model (watched by the monitoring platform)."""
    version_file = Path(model_path) / "version.json"
    if version_file.exists():
        with open(version_file) as f:
            return json.load(f)
    return {"model_version": "unknown", "trained_at": None}

@app.post("/invocations")
@app.post("/infer")
async def invocations(request: Request):
    body = await request.body()
    req = urllib.request.Request(
        f"http://{upstream_host}:{upstream_port}/invocations",
        data=body,
        headers={"Content-Type": request.headers.get("content-type", "application/json")},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return Response(
                content=resp.read(),
                status_code=resp.status,
                media_type=resp.headers.get("content-type", "application/json"),
            )
    except urllib.error.HTTPError as e:
        return Response(
            content=e.read(),
            status_code=e.code,
            media_type=e.headers.get("content-type", "application/json"),
        )
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return Response(
            content=f'{{"status": "error", "detail": "{e}"}}',
            status_code=502,
            media_type="application/json",
        )

@app.get("/drift")
def drift(days: int = Query(default=drift_default_days, ge=1)):
    """Run drift detection over the requests logged by the scoring server."""
    logs = load_inference_logs(inference_log_path)
    baseline = load_baseline_stats()
    return detect_drift(logs, days=days, baseline_stats=baseline)

print("API defined")

import threading

# uvicorn.run would block the notebook, so the API runs in a background thread
api_config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
api_server = uvicorn.Server(api_config)

api_thread = threading.Thread(target=api_server.run, daemon=True)
api_thread.start()
print(f"API server starting on http://localhost:{port} (background thread)")

import requests

# Wait for the scoring server to come up (loading torch takes a while)
for attempt in range(120):
    try:
        r = requests.get(f"http://localhost:{port}/health", timeout=2)
        print(f"[{attempt}] /health -> {r.status_code} {r.text.strip()}")
        if r.status_code == 200:
            print("Model server is ready!")
            break
    except requests.RequestException:
        print(f"[{attempt}] waiting for the API server...")
    time.sleep(2)



# # Test the endpoint
# Send one of the test images to the endpoint as base64 
# (the portable contract - the server can also take file paths, but those only work when the file is on the server's machine).

import base64

test_metadata = pd.read_csv("data/test/metadata.csv")
sample_file = test_metadata.iloc[0]["filename"]
sample_label = int(test_metadata.iloc[0]["class"])
print(f"Sample image: {sample_file} (label={sample_label})")

with open(Path("data/test") / sample_file, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

payload = {"instances": [img_b64]}

r = requests.post(f"http://localhost:{port}/infer", json=payload, timeout=60)
print("POST /infer ->", r.status_code)
pprint(r.json())

print("GET /version ->", requests.get(f"http://localhost:{port}/version").json())
print("GET /health  ->", requests.get(f"http://localhost:{port}/health").json())



# # DRIFT
# The serving wrapper logged our request(s) to `artifacts/inference_log.json`. 
# The drift check compares the logged confidences and predicted-class distribution against the baseline statistics from 
# training with two-sample KS tests. 
# The `/drift` endpoint of the service runs exactly this over the requests the container has seen.


logs = load_inference_logs(inference_log_path)

print(f"Logged requests: {len(logs)}")
pprint(logs[-3:])

baseline = load_baseline_stats()
drift_results = detect_drift(logs, days=drift_default_days, baseline_stats=baseline)
print("\nDrift check results:")
pprint(drift_results)

if drift_results["overall_drift_detected"]:
    print("\n!! Drift detected - time to retrain and redeploy")
else:
    print("\nNo drift detected")

# The endpoint does the same check over HTTP
print("\nGET /drift ->")
pprint(requests.get(f"http://localhost:{port}/drift").json())



# # DEPLOY
# Everything above ran on the Databricks job cluster. Getting the model into production was a manual hand-off: 
# build the Docker image with the exported model baked in, push it to the GitHub Container Registry and let the DigitalOcean 
# App Platform deploy it from the app spec (`infra/digitalocean/app.yaml`):

# ```
# ghcr.io/<owner>/model-serve:<tag>   <- built and pushed in CI
# infra/digitalocean/app.yaml         <- App Platform spec (note the __OWNER__ and __TAG__ placeholders)
# ```

# Your job is to automate exactly that hand-off: turn the two notebooks into a proper pipeline (modular Python, Dagger) that trains,
# exports and serves the model, and a CI/CD setup that builds the image and deploys it - including the retrain-and-redeploy 
# loop when the drift check fires.


# This is what the platform team used to run by hand - now it belongs in CI/CD.
# Nothing here is executed by the notebook.

# !docker build -t ghcr.io/OWNER/model-serve:TAG -f docker/Dockerfile.model-serve .
# !docker push ghcr.io/OWNER/model-serve:TAG
#
# # render the placeholders in the app spec and hand it to DigitalOcean
# !doctl apps spec validate infra/digitalocean/app.yaml
# !doctl apps create --spec infra/digitalocean/app.yaml   # first time
# !doctl apps update APP_ID --spec infra/digitalocean/app.yaml  # after that


# # Cleanup

# Stop the local servers. (In production the FastAPI app runs as the container's main process; the scoring server 
# stays a subprocess of it.)

api_server.should_exit = True
if scoring_proc is not None and scoring_proc.poll() is None:
    scoring_proc.terminate()
    scoring_proc.wait(timeout=10)
    print("Scoring server stopped")
print("API server stopped")


