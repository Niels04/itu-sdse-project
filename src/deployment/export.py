"""
Export the serving model from the MLflow registry to a local directory.

The pyfunc model (SimpleLoggingWrapper, see src/deployment/logging_wrapper.py)
is the flavor the serving image runs; Dockerfile.model-serve copies it from
the export destination into the image. The export resolves the model the way
the pipeline decided it should be served:

1. The model with the "Production" alias - what the training pipeline's
   promotion gate chose, so a rejected run is never silently served.
2. If no Production alias exists, the newest registered model version.
3. If nothing is registered at all, the model_pyfunc artifact of the
   latest training run (CI's behavior before the registry existed).

Usage:
    python -m src.deployment.export [--dst models/model]
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from ..config import ARTIFACT_PATH, EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_NAME

DEFAULT_DST = "models/model"


def resolve_model_uri() -> tuple[str, str]:
    """
    Resolve the URI of the model that should be served.

    Returns (uri, description) where description explains which model was
    chosen and why, for logging.
    """
    client = MlflowClient()

    try:
        mv = client.get_model_version_by_alias(name=MODEL_NAME, alias="Production")
        source = getattr(mv, "source", None) or mv.storage_location
        return source, f"{MODEL_NAME} version {mv.version} (Production alias)"
    except Exception:
        pass

    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if versions:
            mv = max(versions, key=lambda v: int(v.version))
            source = getattr(mv, "source", None) or mv.storage_location
            return (
                source,
                f"{MODEL_NAME} version {mv.version} (newest registered; no Production alias)",
            )
    except Exception:
        pass

    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        max_results=1,
        order_by=["start_time DESC"],
    )
    if runs.empty:
        raise RuntimeError(
            f"No registered {MODEL_NAME} model and no runs in "
            f"experiment {EXPERIMENT_NAME} - nothing to export."
        )
    run_id = runs.iloc[0].run_id
    return (
        f"runs:/{run_id}/{ARTIFACT_PATH}",
        f"run {run_id} {ARTIFACT_PATH} artifact (latest run; nothing registered)",
    )


def export_serving_model(dst: str = DEFAULT_DST) -> str:
    """
    Download the resolved model to dst, replacing any previous export.

    Downloads to a temporary directory first so a failed download never
    leaves a half-written dst behind.
    """
    model_uri, description = resolve_model_uri()
    print(f"Exporting: {description}")
    print(f"  Source: {model_uri}")

    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dst_path.parent) as tmp:
        download_dir = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri, dst_path=tmp
        )
        if dst_path.exists():
            shutil.rmtree(dst_path)
        shutil.move(str(download_dir), str(dst_path))

    print(f"  Exported to: {dst_path.resolve()}")
    return str(dst_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dst",
        default=DEFAULT_DST,
        help=f"Export destination directory (default: {DEFAULT_DST})",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    export_serving_model(args.dst)


if __name__ == "__main__":
    main()
