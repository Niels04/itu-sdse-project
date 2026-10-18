"""
Drift Check CLI Command

Provides a simple CLI command that reads logged inference requests, checks for drift,
and logs the drift metrics to the same MLflow run as the model.

This is a standalone command that students can call manually or automate themselves.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import ks_2samp

from mlflow.tracking import MlflowClient

from ..config import (
    MLFLOW_TRACKING_URI,
    INFERENCE_LOG_PATH,
    DRIFT_DEFAULT_DAYS,
    DRIFT_BASELINE_STATS_PATH,
    DRIFT_KS_PVALUE_THRESHOLD,
    MODEL_NAME,
    EXPERIMENT_NAME,
)


def setup_mlflow() -> MlflowClient:
    """Setup MLflow tracking and return a client."""
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def load_inference_logs(log_path: Path = INFERENCE_LOG_PATH) -> List[Dict[str, Any]]:
    """
    Load inference logs from the JSON file.
    
    Parameters
    ----------
    log_path : Path
        Path to the inference log JSON file.
        
    Returns
    -------
    List[Dict[str, Any]]
        List of log entries, or empty list if file doesn't exist or is invalid.
    """
    if not log_path.exists():
        print(f"Warning: Inference log not found at {log_path}")
        return []
    
    try:
        with open(log_path, 'r') as f:
            logs = json.load(f)
            if not isinstance(logs, list):
                print(f"Warning: Inference log at {log_path} is not a list")
                return []
            return logs
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse inference log: {e}")
        return []
    except Exception as e:
        print(f"Warning: Failed to load inference log: {e}")
        return []


def filter_logs_by_time(logs: List[Dict[str, Any]], days: int = DRIFT_DEFAULT_DAYS) -> List[Dict[str, Any]]:
    """
    Filter logs to only include entries from the last N days.
    
    Parameters
    ----------
    logs : List[Dict[str, Any]]
        List of log entries.
    days : int
        Number of days to look back.
        
    Returns
    -------
    List[Dict[str, Any]]
        Filtered list of log entries.
    """
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
            # If timestamp is invalid, include the log entry
            filtered.append(log)
    
    return filtered


def load_baseline_stats() -> Optional[Dict[str, Any]]:
    """
    Load baseline statistics from the JSON file.
    
    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary with baseline statistics, or None if file doesn't exist.
    """
    if not DRIFT_BASELINE_STATS_PATH.exists():
        print(f"Warning: Baseline stats not found at {DRIFT_BASELINE_STATS_PATH}")
        print("Drift detection may not work properly without baseline statistics.")
        return None
    
    try:
        with open(DRIFT_BASELINE_STATS_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load baseline stats: {e}")
        return None


def calculate_confidence_metrics(confidences: List[float]) -> Dict[str, float]:
    """
    Calculate statistics for confidence scores.
    
    Parameters
    ----------
    confidences : List[float]
        List of confidence scores.
        
    Returns
    -------
    Dict[str, float]
        Dictionary with confidence statistics.
    """
    confidences_array = np.array(confidences)
    
    return {
        'mean': float(np.mean(confidences_array)) if len(confidences_array) > 0 else 0.0,
        'std': float(np.std(confidences_array)) if len(confidences_array) > 0 else 0.0,
        'min': float(np.min(confidences_array)) if len(confidences_array) > 0 else 0.0,
        'max': float(np.max(confidences_array)) if len(confidences_array) > 0 else 0.0,
    }


def check_confidence_drift(
    current_confidences: List[float],
    baseline_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Check for drift in confidence score distribution.
    
    Parameters
    ----------
    current_confidences : List[float]
        List of current confidence scores.
    baseline_stats : Dict[str, Any]
        Dictionary with baseline statistics.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with drift detection results for confidence.
    """
    if not baseline_stats or 'confidence_statistics' not in baseline_stats:
        return {
            'ks_statistic': 0.0,
            'ks_pvalue': 1.0,
            'mean_diff': 0.0,
            'current_mean': 0.0,
            'baseline_mean': 0.0,
            'drift_detected': False,
            'threshold_ks': DRIFT_KS_PVALUE_THRESHOLD,
        }
    
    baseline = baseline_stats['confidence_statistics']
    current_confidences_array = np.array(current_confidences)
    
    if len(current_confidences_array) < 2:
        return {
            'ks_statistic': 0.0,
            'ks_pvalue': 1.0,
            'mean_diff': 0.0,
            'current_mean': float(np.mean(current_confidences_array)) if len(current_confidences_array) > 0 else 0.0,
            'baseline_mean': baseline.get('mean', 0.0),
            'drift_detected': False,
            'threshold_ks': DRIFT_KS_PVALUE_THRESHOLD,
        }
    
    baseline_mean = baseline['mean']
    baseline_std = baseline['std']
    current_mean = float(np.mean(current_confidences_array))
    
    # Calculate KS test
    baseline_samples = np.random.normal(baseline_mean, baseline_std, size=len(current_confidences_array))
    ks_statistic, ks_pvalue = ks_2samp(baseline_samples, current_confidences_array)
    
    # Calculate mean difference
    mean_diff = abs(current_mean - baseline_mean)
    
    # Check drift thresholds
    ks_drift = ks_pvalue < DRIFT_KS_PVALUE_THRESHOLD
    mean_drift = mean_diff > 0.1
    # bool() casts numpy.bool_ to a plain bool (JSON-serializable)
    drift_detected = bool(ks_drift or mean_drift)
    
    return {
        'ks_statistic': float(ks_statistic),
        'ks_pvalue': float(ks_pvalue),
        'mean_diff': mean_diff,
        'current_mean': current_mean,
        'baseline_mean': baseline_mean,
        'drift_detected': drift_detected,
        'threshold_ks': DRIFT_KS_PVALUE_THRESHOLD,
        'threshold_mean': 0.1,
    }


def check_class_distribution_drift(
    predictions: List[int],
    baseline_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Check for drift in class distribution.
    
    Parameters
    ----------
    predictions : List[int]
        List of predicted classes (0 or 1).
    baseline_stats : Dict[str, Any]
        Dictionary with baseline statistics.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with drift detection results for class distribution.
    """
    if not baseline_stats or 'class_distribution' not in baseline_stats:
        return {
            'ks_statistic': 0.0,
            'ks_pvalue': 1.0,
            'drift_detected': False,
            'threshold_ks': DRIFT_KS_PVALUE_THRESHOLD,
        }
    
    baseline = baseline_stats['class_distribution']
    baseline_total = baseline['total']
    current_total = len(predictions)
    
    if current_total == 0:
        return {
            'ks_statistic': 0.0,
            'ks_pvalue': 1.0,
            'drift_detected': False,
            'threshold_ks': DRIFT_KS_PVALUE_THRESHOLD,
        }
    
    # Calculate distributions
    baseline_prop_0 = baseline['class_0_count'] / baseline_total
    baseline_prop_1 = baseline['class_1_count'] / baseline_total
    
    baseline_dist = np.array([baseline_prop_0, baseline_prop_1])
    
    # Calculate KS test
    baseline_samples = np.random.choice([0, 1], size=baseline_total, p=baseline_dist)
    ks_statistic, ks_pvalue = ks_2samp(baseline_samples, predictions)
    
    # Check drift thresholds
    ks_drift = ks_pvalue < DRIFT_KS_PVALUE_THRESHOLD
    drift_detected = bool(ks_drift)
    
    return {
        'ks_statistic': float(ks_statistic),
        'ks_pvalue': float(ks_pvalue),
        'drift_detected': drift_detected,
        'threshold_ks': DRIFT_KS_PVALUE_THRESHOLD,
    }


def get_model_run_id(model_name: str = MODEL_NAME) -> Optional[str]:
    """
    Get the run ID associated with the latest version of a model.
    
    Parameters
    ----------
    model_name : str
        Name of the model.
        
    Returns
    -------
    Optional[str]
        Run ID of the model's latest version, or None if not found.
    """
    client = MlflowClient()
    
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            latest = max(versions, key=lambda v: int(v.version))
            return latest.run_id
    except Exception as e:
        print(f"Warning: Could not get model run ID: {e}")
    
    return None


def get_latest_run_id() -> Optional[str]:
    """
    Get the run ID of the latest run in the training experiment.
    
    Returns
    -------
    Optional[str]
        Run ID of the latest run, or None if not found.
    """
    import mlflow
    client = MlflowClient()
    
    try:
        # Get the project's training experiment
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

        if experiment is None:
            print(f"Warning: Experiment {EXPERIMENT_NAME} not found")
            return None
        
        experiment_id = experiment.experiment_id
        
        # Search for latest run
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            max_results=1,
            order_by=["attribute.start_time DESC"]
        )
        
        if runs and len(runs) > 0:
            return runs[0].info.run_id
        
    except Exception as e:
        print(f"Warning: Could not get latest run ID: {e}")
    
    return None


def detect_drift(
    logs: List[Dict[str, Any]],
    days: int = DRIFT_DEFAULT_DAYS,
    baseline_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main drift detection function.
    
    Parameters
    ----------
    logs : List[Dict[str, Any]]
        List of inference log entries.
    days : int
        Number of days to look back for logs.
    baseline_stats : Optional[Dict[str, Any]]
        Dictionary with baseline statistics.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with all drift detection results.
    """
    # Filter logs by time
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
    
    # Extract data from logs
    confidences = []
    predictions = []
    
    for log in filtered_logs:
        if 'confidence' in log and log['confidence'] is not None:
            confidences.append(float(log['confidence']))
        if 'prediction' in log and log['prediction'] is not None:
            predictions.append(int(log['prediction']))
    
    # Calculate overall statistics
    num_samples = len(filtered_logs)
    
    # Check confidence drift
    confidence_results = check_confidence_drift(confidences, baseline_stats or {})
    
    # Check class distribution drift
    class_dist_results = check_class_distribution_drift(predictions, baseline_stats or {})
    
    # Determine overall drift
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


def log_metrics_to_mlflow(
    run_id: str,
    drift_results: Dict[str, Any],
    days: int = DRIFT_DEFAULT_DAYS,
) -> None:
    """
    Log drift metrics to MLflow run.
    
    Parameters
    ----------
    run_id : str
        MLflow run ID to log metrics to.
    drift_results : Dict[str, Any]
        Dictionary with drift detection results.
    days : int
        Number of days checked.
    """
    client = MlflowClient()
    
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Log overall drift status
    client.log_metric(run_id, "drift.overall_detected", int(drift_results['overall_drift_detected']))
    client.log_metric(run_id, "drift.num_samples", drift_results['num_samples'])
    
    # Log confidence drift metrics
    if drift_results.get('confidence_drift'):
        cd = drift_results['confidence_drift']
        client.log_metric(run_id, "drift.confidence.ks_statistic", cd.get('ks_statistic', 0.0))
        client.log_metric(run_id, "drift.confidence.ks_pvalue", cd.get('ks_pvalue', 1.0))
        client.log_metric(run_id, "drift.confidence.mean_diff", cd.get('mean_diff', 0.0))
        client.log_metric(run_id, "drift.confidence.current_mean", cd.get('current_mean', 0.0))
        client.log_metric(run_id, "drift.confidence.baseline_mean", cd.get('baseline_mean', 0.0))
        client.log_metric(run_id, "drift.confidence.detected", int(cd.get('drift_detected', False)))
    
    # Log class distribution drift metrics
    if drift_results.get('class_distribution_drift'):
        cdd = drift_results['class_distribution_drift']
        client.log_metric(run_id, "drift.class.ks_statistic", cdd.get('ks_statistic', 0.0))
        client.log_metric(run_id, "drift.class.ks_pvalue", cdd.get('ks_pvalue', 1.0))
        client.log_metric(run_id, "drift.class.detected", int(cdd.get('drift_detected', False)))
    
    # Log confidence stats
    if drift_results.get('confidence_stats'):
        cs = drift_results['confidence_stats']
        client.log_metric(run_id, "drift.confidence_stats.mean", cs.get('mean', 0.0))
        client.log_metric(run_id, "drift.confidence_stats.std", cs.get('std', 0.0))
        client.log_metric(run_id, "drift.confidence_stats.min", cs.get('min', 0.0))
        client.log_metric(run_id, "drift.confidence_stats.max", cs.get('max', 0.0))
    
    # Log class distribution
    if drift_results.get('class_distribution'):
        cd = drift_results['class_distribution']
        client.log_metric(run_id, "drift.class_counts.class_0", cd.get('class_0_count', 0))
        client.log_metric(run_id, "drift.class_counts.class_1", cd.get('class_1_count', 0))
    
    # Log timestamp as a tag
    client.set_tag(run_id, "drift_check_timestamp", timestamp)
    client.set_tag(run_id, "drift_check_days", str(days))
    
    print(f"Drift metrics logged to MLflow run: {run_id}")


def print_drift_results(drift_results: Dict[str, Any], days: int = DRIFT_DEFAULT_DAYS) -> None:
    """
    Print human-readable drift results to console.
    
    Parameters
    ----------
    drift_results : Dict[str, Any]
        Dictionary with drift detection results.
    days : int
        Number of days checked.
    """
    print("\n" + "=" * 60)
    print("DRIFT CHECK RESULTS")
    print("=" * 60)
    print(f"Time window: Last {days} days")
    print(f"Samples analyzed: {drift_results.get('num_samples', 0)}")
    print(f"\nOverall drift detected: {drift_results.get('overall_drift_detected', False)}")
    
    # Print confidence drift
    if drift_results.get('confidence_drift'):
        cd = drift_results['confidence_drift']
        print(f"\n--- Confidence Drift ---")
        print(f"KS statistic: {cd.get('ks_statistic', 0.0):.4f}, p-value: {cd.get('ks_pvalue', 1.0):.6f} (threshold: {cd.get('threshold_ks', DRIFT_KS_PVALUE_THRESHOLD)})")
        print(f"Mean difference: {cd.get('mean_diff', 0.0):.4f} (threshold: 0.1)")
        print(f"Current mean: {cd.get('current_mean', 0.0):.4f}, Baseline mean: {cd.get('baseline_mean', 0.0):.4f}")
        print(f"Drift detected: {cd.get('drift_detected', False)}")
    
    # Print class distribution drift
    if drift_results.get('class_distribution_drift'):
        cdd = drift_results['class_distribution_drift']
        print(f"\n--- Class Distribution Drift ---")
        print(f"KS statistic: {cdd.get('ks_statistic', 0.0):.4f}, p-value: {cdd.get('ks_pvalue', 1.0):.6f} (threshold: {cdd.get('threshold_ks', DRIFT_KS_PVALUE_THRESHOLD)})")
        print(f"Drift detected: {cdd.get('drift_detected', False)}")
    
    # Print confidence stats
    if drift_results.get('confidence_stats'):
        cs = drift_results['confidence_stats']
        print(f"\n--- Confidence Statistics ---")
        print(f"Mean: {cs.get('mean', 0.0):.4f}, Std: {cs.get('std', 0.0):.4f}")
        print(f"Min: {cs.get('min', 0.0):.4f}, Max: {cs.get('max', 0.0):.4f}")
    
    # Print class distribution
    if drift_results.get('class_distribution'):
        cd = drift_results['class_distribution']
        print(f"\n--- Class Distribution ---")
        print(f"Class 0: {cd.get('class_0_count', 0)} ({cd.get('class_0_count', 0) / max(cd.get('total', 1), 1) * 100:.1f}%)")
        print(f"Class 1: {cd.get('class_1_count', 0)} ({cd.get('class_1_count', 0) / max(cd.get('total', 1), 1) * 100:.1f}%)")
    
    print("=" * 60 + "\n")


def main():
    """Main entry point for the drift check CLI."""
    parser = argparse.ArgumentParser(
        description="Check for drift in model inference requests and log metrics to MLflow"
    )
    parser.add_argument(
        '--log-path',
        type=str,
        default=str(INFERENCE_LOG_PATH),
        help=f"Path to inference log JSON file (default: {INFERENCE_LOG_PATH})",
    )
    parser.add_argument(
        '--days',
        type=int,
        default=DRIFT_DEFAULT_DAYS,
        help=f"Number of days to look back for drift checking (default: {DRIFT_DEFAULT_DAYS})",
    )
    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help="Specific MLflow run ID to log metrics to. If not provided, auto-detects from model registry.",
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=MODEL_NAME,
        help=f"Model name to get run ID from (default: {MODEL_NAME})",
    )
    parser.add_argument(
        '--no-mlflow',
        action='store_true',
        help="Skip logging to MLflow (only print results)",
    )
    parser.add_argument(
        '--baseline-path',
        type=str,
        default=None,
        help="Path to custom baseline statistics JSON file. If not provided, uses DRIFT_BASELINE_STATS_PATH from config.",
    )
    
    args = parser.parse_args()
    
    # Setup MLflow
    setup_mlflow()
    
    # Load baseline stats
    if args.baseline_path:
        baseline_path = Path(args.baseline_path)
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                baseline_stats = json.load(f)
            print(f"Loaded baseline stats from {baseline_path}")
        else:
            print(f"Warning: Baseline stats not found at {baseline_path}")
            baseline_stats = None
    else:
        baseline_stats = load_baseline_stats()
    
    # Load inference logs
    log_path = Path(args.log_path)
    logs = load_inference_logs(log_path)
    
    if not logs:
        print(f"Error: No inference logs found at {log_path}")
        print("Make sure the model has been served and requests have been logged.")
        sys.exit(1)
    
    print(f"Loaded {len(logs)} inference log entries from {log_path}")
    
    # Detect drift
    drift_results = detect_drift(logs, days=args.days, baseline_stats=baseline_stats)
    
    # Print results
    print_drift_results(drift_results, days=args.days)
    
    # Log to MLflow if requested
    if not args.no_mlflow:
        # Get run ID
        run_id = args.run_id
        if run_id is None:
            # Try to get from model
            run_id = get_model_run_id(args.model_name)
        
        if run_id is None:
            # Try to get latest run
            run_id = get_latest_run_id()
        
        if run_id is None:
            print("Warning: Could not determine MLflow run ID. Skipping MLflow logging.")
            print("To log metrics, specify --run-id or ensure a model is registered.")
        else:
            print(f"Logging drift metrics to MLflow run: {run_id}")
            log_metrics_to_mlflow(run_id, drift_results, days=args.days)
    else:
        print("Skipping MLflow logging (--no-mlflow flag)")
    
    # Exit with error code if drift detected
    if drift_results.get('overall_drift_detected', False):
        print("DRIFT DETECTED!")
        sys.exit(1)
    else:
        print("No drift detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
