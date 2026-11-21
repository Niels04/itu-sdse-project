# dbutils.widgets.text("Model version", "1")
# model_version = dbutils.widgets.get("Model version")

model_name = "cnn_model"
experiment_name = "vial_image_classification"
artifact_path = "model_pyfunc"

model_version = 1  # fallback, the real version is looked up from the registry below

artifact_dir = "artifacts"
mlflow_state_dir = "mlflow-state"
drift_baseline_stats_path = artifact_dir + "/drift_baseline_stats.json"
inference_log_path = artifact_dir + "/inference_log.json"
model_results_path = artifact_dir + "/model_results.json"
drift_default_days = 7
drift_ks_pvalue_threshold = 0.05

# Where the exported servable model goes (this is what gets baked into the Docker image)
export_dir = "models/model"
model_path = export_dir

# Serving ports
port = 5001
upstream_port = 5002
upstream_host = "127.0.0.1"

# Where the endpoint lives in production
PROD_ENDPOINT = "https://mlops-itu-jtk.ondigitalocean.app"

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
import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint
import warnings

warnings.filterwarnings('ignore')

mlflow_tracking_uri = f"sqlite:///{mlflow_state_dir}/mlflow.db"
mlflow_artifact_location = "./mlflow-state/artifacts"

def setup_mlflow_tracking(experiment_name=None):
    os.makedirs(mlflow_state_dir, exist_ok=True)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    if experiment_name is None:
        return
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=mlflow_artifact_location)
    mlflow.set_experiment(experiment_name)

setup_mlflow_tracking(experiment_name)
print("MLflow tracking URI:", mlflow.get_tracking_uri())

# MODEL SELECTION
# Model selection involves choosing the most suitable statistical model from a set of candidates. 
# In straightforward cases, this process uses an existing dataset. 
# When candidate models offer comparable predictive or explanatory power, the simplest model is generally the preferred choice.

experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_ids = [experiment.experiment_id]
experiment_ids

# The best run of the experiment is the one with the highest test F1-score
experiment_best = mlflow.search_runs(
    experiment_ids=experiment_ids,
    order_by=["metrics.test_f1_score DESC"],
    max_results=1,
).iloc[0]
print("Best run:", experiment_best["run_id"])
print("Test F1-score:", experiment_best.get("metrics.test_f1_score"))
experiment_best

with open(model_results_path, "r") as f:
    model_results = json.load(f)
pprint(model_results["classification_report"])



# Get production model
client = MlflowClient()

def get_production_model():
    """Fetch the current production model, by alias or by stage.

    Returns (model_uri, version, run_id) or (None, None, None) if there is
    no model in production yet.
    """
    # First, try to get the model with the alias "Production"
    try:
        model = client.get_model_version_by_alias(name=model_name, alias="Production")
        if model:
            print('Production model name: ', model_name)
            print('Production model version:', model.version)
            print('Production model run id:', model.run_id)
            return f"models:/{model_name}@Production", model.version, model.run_id
    except Exception:
        pass

    # Fall back to checking stage="Production"
    for model in client.search_model_versions(f"name='{model_name}'"):
        if dict(model).get('current_stage') == 'Production':
            print('Production model name: ', model_name)
            print('Production model version:', model.version)
            print('Production model run id:', model.run_id)
            return f"models:/{model_name}/{model.version}", model.version, model.run_id

    print('No model in production')
    return None, None, None

production_model_uri, prod_model_version, prod_model_run_id = get_production_model()



# Compare prod and best trained model
train_model_score = experiment_best.get("metrics.test_f1_score")
if train_model_score is None:
    train_model_score = experiment_best.get("metrics.f1_score")

model_status = {}
run_id = None

if production_model_uri is not None:
    prod_run = mlflow.get_run(prod_model_run_id)
    prod_model_score = prod_run.data.metrics.get("test_f1_score") or prod_run.data.metrics.get("f1_score")

    model_status["current"] = train_model_score
    model_status["prod"] = prod_model_score

    print("Comparing models:")
    print(f"  Current best (training): {train_model_score}")
    print(f"  Production: {prod_model_score}")

    if train_model_score > prod_model_score:
        print("New model is better. Promoting the new model.")
        run_id = experiment_best["run_id"]
    else:
        print("Current production model is better. Keeping it.")
else:
    print("No model in production. The trained model will be promoted.")
    run_id = experiment_best["run_id"]

print(f"Selected run: {run_id}")



# Promote best model
# A model version can be assigned to one or more stages. 
# MLflow provides predefined stages for common use cases: None, Staging, Production, and Archived. 
# With the necessary permissions, you can transition a model version between stages or request a transition to a different stage.
def wait_for_deployment(model_name, version, stage='Staging', max_attempts=20):
    status = False
    for attempt in range(max_attempts):
        model_version_details = dict(client.get_model_version(name=model_name, version=version))
        if model_version_details['current_stage'] == stage:
            print(f'Transition completed to {stage}')
            status = True
            break
        else:
            print(f"Waiting for transition to {stage} (current: {model_version_details['current_stage']})")
            time.sleep(2)
    return status

# Find the registered version that belongs to the winning run
versions = [mv for mv in client.search_model_versions(f"name='{model_name}'") if mv.run_id == run_id]
if not versions:
    print("Run was not registered yet, registering it now")
    model_details = mlflow.register_model(f"runs:/{run_id}/{artifact_path}", name=model_name)
    version = model_details.version
else:
    version = versions[0].version

print(f"Promoting {model_name} version {version}")

# Always via Staging
model_version_details = dict(client.get_model_version(name=model_name, version=version))
if model_version_details['current_stage'] != 'Staging':
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging",
        archive_existing_versions=True,
    )
    wait_for_deployment(model_name, version, 'Staging')

# And to Production (we only get here if the comparison above decided we should)
if run_id is not None:
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    wait_for_deployment(model_name, version, 'Production')
    client.set_registered_model_alias(name=model_name, alias="Production", version=str(version))
    client.set_model_version_tag(model_name, version, "deployed_to", "production")
    print(f"{model_name} version {version} is now in Production")



# Export the serving model

# The serving image needs the model as a plain directory (no MLflow server involved at serving time), so we 
# download the winning pyfunc model to `models/model`. 
# This is the directory that later gets baked into the Docker image.

# We also write the model version and its training timestamp into the export, so the `/version` endpoint of the 
# service can report them - the monitoring platform watches that endpoint to notice when a new model version has been deployed.
import shutil
import tempfile
from datetime import datetime, timezone

def resolve_model_uri():
    """Resolve which model to serve: Production alias, else the newest
    registered version, else the latest run's artifact."""
    try:
        mv = client.get_model_version_by_alias(name=model_name, alias="Production")
        source = getattr(mv, "source", None) or mv.storage_location
        return source, f"{model_name} version {mv.version} (Production alias)"
    except Exception:
        pass

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            mv = max(versions, key=lambda v: int(v.version))
            source = getattr(mv, "source", None) or mv.storage_location
            return source, f"{model_name} version {mv.version} (newest registered; no Production alias)"
    except Exception:
        pass

    runs = mlflow.search_runs(experiment_names=[experiment_name], max_results=1, order_by=["start_time DESC"])
    if runs.empty:
        raise RuntimeError(f"No registered {model_name} model and no runs in experiment {experiment_name} - nothing to export")
    return f"runs:/{runs.iloc[0].run_id}/{artifact_path}", "latest run artifact (nothing registered)"

model_uri, description = resolve_model_uri()
print(f"Exporting: {description}")
print(f"  Source: {model_uri}")

Path(export_dir).parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(dir=Path(export_dir).parent) as tmp:
    download_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=tmp)
    if Path(export_dir).exists():
        shutil.rmtree(export_dir)
    shutil.move(str(download_dir), export_dir)
print(f"  Exported to: {Path(export_dir).resolve()}")

# Bake the version info into the export for the /version endpoint
try:
    prod_mv = client.get_model_version_by_alias(name=model_name, alias="Production")
    trained_at = datetime.fromtimestamp(client.get_run(prod_mv.run_id).info.start_time / 1000, tz=timezone.utc).isoformat()
    version_info = {"model_version": str(prod_mv.version), "trained_at": trained_at}
except Exception:
    version_info = {"model_version": str(version), "trained_at": None}

with open(Path(export_dir) / "version.json", "w") as f:
    json.dump(version_info, f, indent=2)
print("Version info:", version_info)