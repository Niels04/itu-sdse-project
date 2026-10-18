#!/bin/bash
set -e

# Serve the exported pyfunc model (models/model) with `mlflow models serve`.
# The model is served straight from the local directory: serving never
# contacts MLflow, and replacing the files requires a restart.
#
# If models/model is missing, it is exported from the local MLflow state
# (./mlflow-state, written by training - no server involved) with
# `python -m src.deployment.export`, which resolves the model the
# pipeline's promotion gate chose (the Production alias).
#
# Usage: ./scripts/start_model_server.sh
# Stop with: pkill -f "mlflow models serve"

PORT=5001
LOG_FILE="./logs/model_server.log"
MODEL_DIR="models/model"

mkdir -p ./logs

if [ ! -f "$MODEL_DIR/MLmodel" ]; then
    echo "$MODEL_DIR not found; exporting from the local MLflow state..."
    uv run python -m src.deployment.export
fi

# Free the port from a previous model server, if any
if lsof -i :"$PORT" > /dev/null 2>&1; then
    echo "Port $PORT is already in use. Stopping the existing model server..."
    lsof -ti :"$PORT" | xargs kill -9 2>/dev/null || true
    sleep 2
fi

echo "Starting model server on port $PORT (logs: $LOG_FILE)..."
nohup uv run mlflow models serve \
    -m "$MODEL_DIR" \
    -p "$PORT" \
    --host 0.0.0.0 \
    --workers 1 \
    --no-conda \
    > "$LOG_FILE" 2>&1 &

echo "Model server starting in background."
echo "  Model: $MODEL_DIR (local directory; restart after replacing it)"
echo "  Inference endpoint: http://localhost:$PORT/invocations"
echo "  Log file: $LOG_FILE"
echo "To stop: pkill -f 'mlflow models serve'"
