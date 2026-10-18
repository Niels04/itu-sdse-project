# Model Deployment & Serving Guide

How the trained vial-classification model becomes a running inference
endpoint — locally, in Docker, and on DigitalOcean.

- The whole train → deploy flow is drawn in
  [`training-deploy.excalidraw`](./training-deploy.excalidraw) (open it in
  [excalidraw.com](https://excalidraw.com) or a VS Code Excalidraw plugin)
- Code and module reference: [`src/README.md`](../src/README.md)
- MLflow tracking is local and serverless (`mlflow-state/`); serving never
  contacts MLflow at all.

## Overview

```
python -m src.pipeline          train, register, promote (local MLflow state)
             |
python -m src.deployment.export  download the Production model to models/model
             |
     serving (one of):
       - ./scripts/start_model_server.sh          (mlflow models serve)
       - python -m src.deployment.serve           (FastAPI wrapper, /drift)
       - docker/Dockerfile.model-serve             (baked-in model)
             |
       http://...:5001/invocations   POST base64 image -> prediction
```

The pipeline registers every model and moves it to Staging; it promotes a
new model to Production only if its test F1 beats the current Production
model. Export always resolves the **Production alias**, so a run the gate
rejected is never served.

## Serve locally

```bash
uv run python -m src.pipeline
uv run python -m src.deployment.export
./scripts/start_model_server.sh        # mlflow models serve on :5001
```

`src.deployment.serve` is the alternative frontend used in the Docker
image (and the DigitalOcean deployment): it wraps the same scoring server
with a JSON `/health`, an `/infer` alias, and a `/drift` endpoint:

```bash
uv run python -m src.deployment.serve   # PORT=5001, MODEL_PATH=models/model
```

## Inference API

| Endpoint        | Method | Description |
|-----------------|--------|-------------|
| `/health`       | GET    | `{"status": "ok"}` once the model is loaded |
| `/invocations`  | POST   | inference (MLflow pyfunc contract) |
| `/infer`        | POST   | alias of `/invocations` |
| `/drift`        | GET    | KS drift detection over logged requests (`?days=N`) |

Request format — a list of images under `instances`, each either
**base64-encoded bytes** (the portable contract) or a file path that exists
on the server:

```bash
B64=$(base64 < data/images/raw/Clean_001.jpg | tr -d '\n')
curl -s http://localhost:5001/invocations \
    -H "Content-Type: application/json" \
    -d "{\"instances\": [\"$B64\"]}"
# -> {"predictions": [0], "confidences": [0.12], "raw_outputs": [...], ...}
```

Every prediction is appended to `artifacts/inference_log.json` by the
pyfunc wrapper — the input for drift detection:

```bash
curl -s http://localhost:5001/drift                     # served deployment
uv run python -m src.deployment.check_drift             # local model server
uv run python -m src.deployment.check_drift --days 30  # wider window
```

The CLI also logs the drift metrics to the model's MLflow run; exit code is
non-zero when drift is detected.

## Docker

The training and serving containers share no services: training logs to the
local MLflow state directory (a named volume), serving only reads the
exported model directory baked into its image.

```bash
# Train + export (state persists in the mlflow-state volume; the export
# lands in ./models and ./artifacts on the host via bind mounts)
docker compose -f docker/docker-compose.yml run --rm train

# Build the serving image with the freshly exported model, then run it
docker compose -f docker/docker-compose.yml build model-serve
docker compose -f docker/docker-compose.yml up -d
# -> http://localhost:5001
```

Standalone equivalents:

```bash
docker build -t vial-train -f docker/Dockerfile.train .
docker build -t vial-serve -f docker/Dockerfile.model-serve .
docker run --rm -v "$PWD/models:/app/models" -v "$PWD/artifacts:/app/artifacts" \
    -v vial-mlflow-state:/app/mlflow-state vial-train
docker run -d -p 5001:5001 vial-serve
```

## CI/CD (GitHub Actions)

Both workflows are started manually from the Actions tab; nothing runs on
push. They can be iterated on without retraining or redeploying.

- **`train.yml`** — installs the project, restores the MLflow state
  uploaded by the previous successful run (workflow artifact; the run
  history and Production model accumulate across runs), runs
  `python -m src.pipeline` and `python -m src.deployment.export` against
  the local SQLite state, uploads the state plus the exported model and
  drift baseline as artifacts, and builds/pushes the serving image to GHCR.
  With the `deploy` input set, it dispatches the deploy workflow with the
  tag it just pushed.
- **`deploy.yml`** — renders `infra/digitalocean/app.yaml` (image owner and
  tag), creates or updates the App Platform app, waits for the deployment
  to become ACTIVE, and smoke-tests `/invocations`.

## DigitalOcean App Platform

`infra/digitalocean/app.yaml` deploys the pre-built `model-serve` image
from GHCR as a single service on port 5001 with `/invocations`, `/infer`,
`/health`, and `/drift` routed. There is no MLflow component in the
deployment: inference logging goes to a local file in the container and is
discarded on redeploy; drift is served by `/drift` against the baseline
statistics baked into the image.

Manual deploy (requires `doctl`):

```bash
./scripts/deploy_do_app.sh [tag]        # default tag: latest
```

The image owner defaults to the GHCR owner (all lowercase); override with
the `OWNER` environment variable. GHCR packages must be made public once
after the first push (package settings → Change visibility).
