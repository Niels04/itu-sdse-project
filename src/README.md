# Source Package Documentation

This directory contains the MLOps pipeline for glass vial image
classification (clean vs. defective) with a PyTorch CNN: data fetching,
training, model registry, serving, and drift monitoring.

- Root [`README.md`](../README.md) — project overview and the course task
- [`docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md) — serving: local, Docker, DigitalOcean

## MLflow tracking without a server

Everything MLflow-related — runs, metrics, parameters, the model registry —
logs to the local state directory `mlflow-state/` (gitignored, created
automatically): a SQLite database (`mlflow-state/mlflow.db`) plus artifacts
(`mlflow-state/artifacts/`). **No MLflow server needs to be running** for
training, registration, export, serving, or drift checks.

The defaults come from `src/config.py`:

- `MLFLOW_TRACKING_URI` — `sqlite:///./mlflow-state/mlflow.db`
- `MLFLOW_ARTIFACT_LOCATION` — `./mlflow-state/artifacts`

Both are overridable via environment variables of the same names (set
`MLFLOW_TRACKING_URI` to an `http://` URL to log to a real server instead).
Run everything from the project root: MLflow stores artifact locations as
absolute paths, so run history follows the directory it was created in.

To browse the local state with the MLflow UI:

```bash
uv run mlflow ui --backend-store-uri sqlite:///./mlflow-state/mlflow.db
```

## Running the pipeline

```bash
uv sync                                # install dependencies
uv run python -m src.pipeline          # fetch data -> train -> register -> promote
uv run python -m src.deployment.export # export servable model to models/model
uv run python -m src.deployment.check_drift   # drift check over logged inferences
```

What `python -m src.pipeline` does, end to end:

1. Fetch the dataset from the Hetzner object store into `data/images/raw/`
   and `data/test/` (first run only; skip downloads that already exist)
2. Preprocess: build DataLoaders with the ImageNet-normalized 64x64 transform
3. Train the CNN with early stopping and threshold tuning
4. Write drift baseline statistics to `artifacts/drift_baseline_stats.json`
5. Log the `SimpleLoggingWrapper` pyfunc model to MLflow
6. Register it, transition to Staging, and promote to Production only if
   its test F1 beats the current Production model

## Module reference

```
src/
├── config.py                     # all constants and paths
├── pipeline.py                   # orchestration entry point
├── utils.py                      # shared helpers (incl. MLflow setup)
├── data/
│   ├── fetch.py                  # dataset download from Hetzner S3
│   ├── preprocess.py             # GlassVialDataset, transforms, splits
│   ├── features.py               # image statistics artifacts
│   └── organize_dataset.py       # one-off raw-data inspection utility
├── models/
│   ├── train.py                  # SimpleCNN + training loop + baseline stats
│   ├── evaluate.py               # metrics, confusion matrix, ROC
│   └── registry.py               # model registration and versioning
└── deployment/
    ├── deploy.py                 # stage transitions and aliases
    ├── logging_wrapper.py        # pyfunc wrapper that logs inferences
    ├── export.py                 # export the Production model to models/model
    ├── serve.py                  # FastAPI wrapper around the scoring server
    ├── check_drift.py            # drift detection CLI
    ├── drift_detector.py         # standalone drift detector class (unused)
    └── evaluate_test_set.py      # offline evaluation of data/test/
```

### `config.py`

Single source of truth for paths and parameters. Notable values:

| Constant | Default | Used by |
|----------|---------|---------|
| `MLFLOW_TRACKING_URI` | `sqlite:///./mlflow-state/mlflow.db` | all MLflow modules |
| `MLFLOW_ARTIFACT_LOCATION` | `./mlflow-state/artifacts` | experiment creation |
| `EXPERIMENT_NAME` | `vial_image_classification` | training |
| `MODEL_NAME` | `cnn_model` | model registry |
| `IMAGE_SIZE` | `(64, 64)` | preprocessing, inference |
| `THRESHOLD` | `0.5` | classification (tuned per model) |
| `DRIFT_KS_PVALUE_THRESHOLD` | `0.05` | drift detection |
| `DRIFT_DEFAULT_DAYS` | `7` | drift window |
| `INFERENCE_LOG_PATH` | `artifacts/inference_log.json` | serving, drift CLI |
| `DRIFT_BASELINE_STATS_PATH` | `artifacts/drift_baseline_stats.json` | drift CLI, `/drift` |

Run `uv run python -m src.config` to print the full configuration.

### `utils.py`

Shared helpers plus `setup_mlflow_tracking(experiment_name=None)`, which
every MLflow-using entry point calls first: it creates the state directory,
sets the tracking URI from config, and creates the experiment (with the
explicit artifact location) if it does not exist.

### `pipeline.py`

Orchestration: `run_full_pipeline()` runs the data pipeline, logs the test
dataset as an artifact, trains, registers, and applies the promotion gate
(`compare_with_production` loads the current Production model and keeps it
if the new model's F1 is not better). Run with `python -m src.pipeline`.

### `data/`

- **`fetch.py`** — downloads the ICPR 2024 vial dataset from the public
  Hetzner object store (unsigned S3): metadata CSVs first, then the images
  they list, into `data/images/raw/` and `data/test/`. Idempotent — existing
  files are skipped. `fetch_and_prepare_data()` is the entry point the
  pipeline calls.
- **`preprocess.py`** — `GlassVialDataset` (torch `Dataset` over the raw
  images + metadata), train/val/test transforms (resize to 64x64, ImageNet
  normalization), stratified 70/15/15 splitting, and DataLoader creation.
- **`features.py`** — computes per-channel image statistics and writes
  `artifacts/image_statistics.json`.
- **`organize_dataset.py`** — standalone utility to inspect the raw dataset
  layout and generate a metadata CSV; not part of the pipeline.

### `models/`

- **`train.py`** — `SimpleCNN` (2 conv layers, wide FC head, single logit)
  and the training loop: BCEWithLogitsLoss, early stopping on validation
  macro-F1, ReduceLROnPlateau, threshold tuning on the validation set, and
  test evaluation. Also writes the drift baseline statistics and logs the
  `SimpleLoggingWrapper` pyfunc model to MLflow.
- **`evaluate.py`** — accuracy, F1, confusion matrix, classification report,
  ROC curve; used by training and by `evaluate_test_set.py`.
- **`registry.py`** — registers the pyfunc model from the training run under
  `MODEL_NAME`, waits for the version to become READY.

### `deployment/`

- **`deploy.py`** — registry stage transitions (`transition_to_staging`,
  `transition_to_production`) and alias management (`set_model_alias`).
  Called by the pipeline; can also be run standalone:
  `python -m src.deployment.deploy`.
- **`logging_wrapper.py`** — `SimpleLoggingWrapper`, the pyfunc model that is
  actually served. Accepts a path, raw bytes, or base64 string per image
  (single or in an `instances`/`data` list), applies the training transform,
  returns `{"predictions": [...], "confidences": [...]}`, and appends each
  request to `artifacts/inference_log.json` for drift detection.
- **`export.py`** — resolves which model to serve (Production alias, else
  newest registered version, else the latest run's pyfunc artifact) and
  downloads it to `models/model` (`--dst` to change). The serving image and
  `start_model_server.sh` consume this directory.
- **`serve.py`** — FastAPI app that fronts the MLflow scoring server (run as
  a subprocess on an internal port): `/health` returns JSON (MLflow's own
  health endpoint returns a bare newline), `/invocations` and `/infer` are
  proxied, `/drift` runs the same KS-based detection as the CLI over the
  container's inference log. `PORT` (default 5001), `UPSTREAM_PORT`
  (default 5002), and `MODEL_PATH` (default `/app/models/model`) are
  environment-configurable.
- **`check_drift.py`** — CLI: reads the inference log, compares confidence
  scores and predicted-class distribution against the training baseline with
  two-sample KS tests, prints the result, and logs the metrics to the
  model's MLflow run. `--days N` window (default 7), `--run-id` to target a
  specific run, `--no-mlflow` to only print. Exits non-zero when drift is
  detected.
- **`drift_detector.py`** — an alternative `ImageDriftDetector` class with
  embedding-based checks; currently not imported anywhere (the CLI and the
  `/drift` endpoint use `check_drift.py`).
- **`evaluate_test_set.py`** — offline evaluation: loads a model checkpoint
  from `data/output/` and evaluates every image in `data/test/` against
  `metadata.csv`, writing `data/output/output.csv`.

## Scripts (`scripts/`)

- **`start_model_server.sh`** — serves the exported model
  (`models/model`) with `mlflow models serve` on port 5001, in the
  background, logging to `logs/model_server.log`. If the export is missing
  it runs `python -m src.deployment.export` first. Serving never contacts
  MLflow. Stop with `pkill -f "mlflow models serve"`.
- **`test_inference.py`** — sends every image in `data/test/` to the model
  server (base64 `instances` contract), compares predictions against
  `metadata.csv`, and writes results to `logs/inference_test_results.json`.
- **`test_inference.sh`** — the same test as a shell loop (curl + base64);
  the Python version produces the accuracy summary.
- **`deploy_do_app.sh`** — creates or updates the DigitalOcean App Platform
  app from `infra/digitalocean/app.yaml` with a chosen image tag and waits
  for the deployment to become ACTIVE. Requires `doctl`. This deploys only;
  images are built and pushed by `.github/workflows/train.yml`.

## Testing

There is no test suite in this repository yet (tests are an optional part of
the course task). The closest things to smoke tests:

- `scripts/test_inference.py` against a running model server
- the smoke test in `.github/workflows/deploy.yml` after a DigitalOcean deploy

## Troubleshooting

- **`unable to open database file`** — the MLflow state directory is
  missing or the pipeline was started from another directory. Run from the
  project root; `mlflow-state/` is created automatically on the next
  `python -m src.pipeline`.
- **`export.py` finds nothing to export** — no training run exists yet in
  `mlflow-state/`. Run `uv run python -m src.pipeline` first.
- **Model server returns 503 on `/health`** — the scoring server is still
  loading (torch import takes a while) or crashed; check
  `logs/model_server.log`.
- **Prediction errors with string input** — a request string must be either
  an existing file path (on the server's filesystem) or base64-encoded
  image bytes; the portable contract is base64.
