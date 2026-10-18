"""
JSON health endpoint wrapper around the MLflow scoring server.

MLflow's built-in /health answers with a bare newline declared as
application/json (the SageMaker ping contract: 200 = model loaded), which
JSON clients reject. This wrapper fronts the scoring server so the
platform-facing endpoints are:

- GET  /health       -> {"status": "ok"} once the scoring server is up,
                        {"status": "loading"} (HTTP 503) while the model
                        loads, {"status": "error"} (HTTP 503) if the
                        scoring server process died.
- POST /invocations  -> forwarded unchanged to the scoring server.
- POST /infer        -> alias of /invocations (same forwarding).
- GET  /drift        -> runs drift detection (KS tests) over the requests
                        logged by the scoring server in this container and
                        returns the results as JSON. The baseline
                        statistics written during training are baked into
                        the image. Optional ?days=N limits the window
                        (default: 7 days).

The scoring server itself runs unchanged as a subprocess on an internal
port; the wrapper adds no inference logic, only the health response and
the reverse proxy. Runs with `python -m src.deployment.serve`.

Background for readers who only know `mlflow models serve`: that command
also starts a web server under the hood (it launches a uvicorn process
running an app built with FastAPI - the same stack used here). This file
just does the same thing explicitly so we control the response bodies:

- FastAPI is a library for defining HTTP endpoints (like the /health
  function below) as plain Python functions.
- uvicorn is the web server that listens on a port and calls those
  functions for each incoming request.
- MLflow's own server is a FastAPI/uvicorn app whose /health handler is
  fixed to return a bare newline; we cannot change its response, so we
  put our own app in front and forward /invocations to it.
"""

import atexit
import os
import subprocess
import sys
import urllib.error
import urllib.request

# FastAPI/uvicorn are installed as MLflow dependencies (MLflow's own
# scoring server uses them), so they are available without adding
# anything to pyproject.toml.
import uvicorn  # the web server: listens on PORT, calls the app per request
from fastapi import FastAPI, Query, Request, Response  # the HTTP app: maps paths to functions

from ..config import DRIFT_DEFAULT_DAYS, INFERENCE_LOG_PATH
from .check_drift import detect_drift, load_baseline_stats, load_inference_logs

UPSTREAM_HOST = "127.0.0.1"
UPSTREAM_PORT = int(os.environ.get("UPSTREAM_PORT", "5002"))
PORT = int(os.environ.get("PORT", "5001"))
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/model")
POLL_TIMEOUT = 1.0

app = FastAPI(docs_url=None, openapi_url=None, redoc_url=None)
_scoring_proc: subprocess.Popen | None = None


def _upstream_health() -> int | None:
    """Return the scoring server's /health status code, None if unreachable."""
    try:
        with urllib.request.urlopen(
            f"http://{UPSTREAM_HOST}:{UPSTREAM_PORT}/health",
            timeout=POLL_TIMEOUT,
        ) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


@app.get("/health")
def health():
    if _scoring_proc is not None and _scoring_proc.poll() is not None:
        return Response(
            content='{"status": "error"}',
            status_code=503,
            media_type="application/json",
        )
    if _upstream_health() == 200:
        return Response(
            content='{"status": "ok"}', media_type="application/json"
        )
    return Response(
        content='{"status": "loading"}',
        status_code=503,
        media_type="application/json",
    )


@app.post("/invocations")
@app.post("/infer")
async def invocations(request: Request):
    body = await request.body()
    req = urllib.request.Request(
        f"http://{UPSTREAM_HOST}:{UPSTREAM_PORT}/invocations",
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
def drift(days: int = Query(default=DRIFT_DEFAULT_DAYS, ge=1)):
    """
    Run drift detection over the requests logged by the scoring server.

    Reuses the check_drift CLI's detection logic (KS tests on confidence
    scores and class distribution); only the reporting differs: results
    are returned as JSON instead of printed and logged to MLflow (there
    is no tracking server in the deployment - the CLI covers that).
    """
    logs = load_inference_logs(INFERENCE_LOG_PATH)
    baseline = load_baseline_stats()
    return detect_drift(logs, days=days, baseline_stats=baseline)


def _start_scoring_server() -> subprocess.Popen:
    env = os.environ.copy()
    # The scoring server execs `uvicorn` through PATH; make sure the
    # interpreter's bin directory (which holds uvicorn) is on it.
    interpreter_bin = os.path.dirname(os.path.abspath(sys.executable))
    env["PATH"] = os.pathsep.join([interpreter_bin, env.get("PATH", "")])
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "mlflow", "models", "serve",
            "-m", MODEL_PATH,
            "-p", str(UPSTREAM_PORT),
            "--workers", "1",
            "--host", UPSTREAM_HOST,
            "--no-conda",
        ],
        env=env,
    )
    # uvicorn owns SIGTERM handling once it runs, so rely on atexit to
    # stop the scoring server when the wrapper process exits.
    atexit.register(lambda: proc.poll() is None and proc.terminate())
    return proc


if __name__ == "__main__":
    _scoring_proc = _start_scoring_server()
    # uvicorn.run starts the web server: it listens on 0.0.0.0:PORT and
    # dispatches each request to the matching FastAPI function above.
    # (This is the equivalent of what `mlflow models serve` does
    # internally, just with our app instead of MLflow's.)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
