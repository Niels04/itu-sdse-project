"""
Drift Detection Module

Provides functionality for detecting data drift in model inputs and predictions.
Includes the Kolmogorov-Smirnov test and other statistical methods for
drift detection.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.stats import ks_2samp, wasserstein_distance

from ..config import (
    DRIFT_KS_PVALUE_THRESHOLD,
    DRIFT_BASELINE_STATS_PATH,
)

# Baseline image statistics are stored as single scalars (one mean/std per
# channel), so there are no distributions to compare. Drift is flagged when
# a channel statistic moves away from the baseline by more than this
# tolerance (image statistics are in the [0, 1] normalized range).
IMAGE_STAT_TOLERANCE = 0.05


class ImageDriftDetector:
    """
    Drift detector for image classification models.
    
    Detects various types of drift:
    - Image statistics drift (mean, std per channel)
    - Class distribution drift
    - Prediction distribution drift
    - Confidence score drift
    - Feature embedding drift
    
    Uses the Kolmogorov-Smirnov test for statistical drift detection.
    """
    
    def __init__(self, baseline_stats_path: Path = DRIFT_BASELINE_STATS_PATH):
        """
        Initialize the drift detector with baseline statistics.
        
        Parameters
        ----------
        baseline_stats_path : Path, default=DRIFT_BASELINE_STATS_PATH from config
            Path to the JSON file containing baseline statistics.
        """
        self.baseline_stats_path = baseline_stats_path
        self.baseline_stats = self._load_baseline_stats()
        
    def _load_baseline_stats(self) -> Dict:
        """
        Load baseline statistics from JSON file.
        
        Returns
        -------
        Dict
            Dictionary with baseline statistics.
        """
        if not self.baseline_stats_path.exists():
            raise FileNotFoundError(
                f"Baseline statistics file not found at {self.baseline_stats_path}. "
                "Please train a model first to generate baseline statistics."
            )
        
        with open(self.baseline_stats_path, 'r') as f:
            stats = json.load(f)
        
        print(f"Loaded baseline statistics from {self.baseline_stats_path}")
        return stats
    
    def detect_image_statistics_drift(
        self,
        current_mean_r: float,
        current_mean_g: float,
        current_mean_b: float,
        current_std_r: float,
        current_std_g: float,
        current_std_b: float,
    ) -> Dict[str, float]:
        """
        Detect drift in image channel statistics (mean and std).
        
        Compares current image statistics with the baseline scalars and
        flags drift when any statistic differs by more than
        IMAGE_STAT_TOLERANCE.
        
        Parameters
        ----------
        current_mean_r/g/b : float
            Current mean values for each RGB channel.
        current_std_r/g/b : float
            Current standard deviation values for each RGB channel.
            
        Returns
        -------
        Dict[str, float]
            Dictionary with the difference from baseline for each
            statistic and whether drift is detected.
        """
        baseline = self.baseline_stats['image_statistics']
        
        diffs = {
            'mean_r_diff': abs(current_mean_r - baseline['mean_r']),
            'mean_g_diff': abs(current_mean_g - baseline['mean_g']),
            'mean_b_diff': abs(current_mean_b - baseline['mean_b']),
            'std_r_diff': abs(current_std_r - baseline['std_r']),
            'std_g_diff': abs(current_std_g - baseline['std_g']),
            'std_b_diff': abs(current_std_b - baseline['std_b']),
        }
        
        drift_detected = any(
            diff > IMAGE_STAT_TOLERANCE for diff in diffs.values()
        )
        
        return {
            **diffs,
            'drift_detected': drift_detected,
            'threshold': IMAGE_STAT_TOLERANCE,
        }
    
    def detect_class_distribution_drift(
        self,
        current_class_counts: Dict[int, int],
    ) -> Dict[str, float]:
        """
        Detect drift in class distribution using the KS test.
        
        Parameters
        ----------
        current_class_counts : Dict[int, int]
            Dictionary with current class counts (0 and 1).
            
        Returns
        -------
        Dict[str, float]
            Dictionary with KS statistic/p-value and whether drift is detected.
        """
        baseline = self.baseline_stats['class_distribution']
        
        # Convert to numpy arrays for comparison
        baseline_total = baseline['total']
        current_total = current_class_counts.get(0, 0) + current_class_counts.get(1, 0)
        
        if current_total == 0:
            return {
                'ks_statistic': 0.0,
                'ks_pvalue': 1.0,
                'drift_detected': False,
                'threshold_ks': DRIFT_KS_PVALUE_THRESHOLD,
            }
        
        # Create proportional distributions
        baseline_prop_0 = baseline['class_0_count'] / baseline_total
        baseline_prop_1 = baseline['class_1_count'] / baseline_total
        current_prop_0 = current_class_counts.get(0, 0) / current_total
        current_prop_1 = current_class_counts.get(1, 0) / current_total
        
        baseline_dist = np.array([baseline_prop_0, baseline_prop_1])
        current_dist = np.array([current_prop_0, current_prop_1])
        
        # Calculate KS test (on cumulative distributions)
        ks_statistic, ks_pvalue = ks_2samp(
            np.random.choice([0, 1], size=baseline_total, p=baseline_dist),
            np.random.choice([0, 1], size=current_total, p=current_dist),
        )
        
        # Check drift thresholds
        ks_drift = ks_pvalue < DRIFT_KS_PVALUE_THRESHOLD
        drift_detected = ks_drift
        
        return {
            'ks_statistic': float(ks_statistic),
            'ks_pvalue': float(ks_pvalue),
            'drift_detected': drift_detected,
            'threshold_ks': DRIFT_KS_PVALUE_THRESHOLD,
        }
    
    def detect_confidence_drift(
        self,
        current_confidences: List[float],
    ) -> Dict[str, float]:
        """
        Detect drift in confidence score distribution.
        
        Parameters
        ----------
        current_confidences : List[float]
            List of confidence scores from current predictions.
            
        Returns
        -------
        Dict[str, float]
            Dictionary with KS statistic/p-value, mean difference,
            and whether drift is detected.
        """
        baseline = self.baseline_stats['confidence_statistics']
        current_confidences = np.array(current_confidences)
        
        # Generate baseline distribution (normal approximation)
        baseline_mean = baseline['mean']
        baseline_std = baseline['std']
        
        if len(current_confidences) < 2:
            return {
                'ks_statistic': 0.0,
                'ks_pvalue': 1.0,
                'mean_diff': 0.0,
                'drift_detected': False,
                'threshold_ks': DRIFT_KS_PVALUE_THRESHOLD,
                'threshold_mean': 0.1,
            }
        
        current_mean = float(np.mean(current_confidences))
        
        # Calculate KS test
        baseline_samples = np.random.normal(
            baseline_mean, baseline_std, size=10000
        )
        ks_statistic, ks_pvalue = ks_2samp(baseline_samples, current_confidences)
        
        # Calculate mean difference
        mean_diff = abs(current_mean - baseline_mean)
        
        # Check drift thresholds
        ks_drift = ks_pvalue < DRIFT_KS_PVALUE_THRESHOLD
        mean_drift = mean_diff > 0.1
        drift_detected = ks_drift or mean_drift
        
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
    
    def detect_embedding_drift(
        self,
        current_embeddings: np.ndarray,
    ) -> Dict[str, float]:
        """
        Detect drift in feature embeddings using the Wasserstein distance.
        
        Parameters
        ----------
        current_embeddings : np.ndarray
            Array of feature embeddings from current data.
            
        Returns
        -------
        Dict[str, float]
            Dictionary with the Wasserstein distance and whether drift is detected.
        """
        baseline_embedding_stats = self.baseline_stats.get('embedding_statistics', {})
        
        if not baseline_embedding_stats or len(current_embeddings) == 0:
            return {
                'wasserstein_distance': 0.0,
                'drift_detected': False,
                'threshold': 0.1,
            }
        
        # Generate baseline distribution (multivariate normal)
        baseline_mean = np.array(baseline_embedding_stats['mean'])
        baseline_std = np.array(baseline_embedding_stats['std'])
        
        # Calculate Wasserstein distance (mean of per-feature distances)
        distances = []
        for i in range(min(len(baseline_mean), current_embeddings.shape[1])):
            baseline_samples = np.random.normal(
                baseline_mean[i], baseline_std[i], size=10000
            )
            current_values = current_embeddings[:, i]
            dist = wasserstein_distance(baseline_samples, current_values)
            distances.append(dist)
        
        wasserstein_distance = float(np.mean(distances))
        
        # Check drift threshold
        drift_detected = wasserstein_distance > 0.1
        
        return {
            'wasserstein_distance': wasserstein_distance,
            'drift_detected': drift_detected,
            'threshold': 0.1,
        }
    
    def detect_drift(
        self,
        batch_data: Dict[str, object],
    ) -> Dict[str, object]:
        """
        Main method to detect all types of drift from a batch of data.
        
        Parameters
        ----------
        batch_data : Dict[str, object]
            Dictionary containing:
            - 'image_means': Tuple of (mean_r, mean_g, mean_b, std_r, std_g, std_b)
            - 'class_counts': Dict with class 0 and 1 counts
            - 'confidences': List of confidence scores
            - 'embeddings': Numpy array of feature embeddings (optional)
            
        Returns
        -------
        Dict[str, object]
            Dictionary with drift detection results for all types of drift.
        """
        results = {
            'image_statistics_drift': None,
            'class_distribution_drift': None,
            'confidence_drift': None,
            'embedding_drift': None,
            'overall_drift_detected': False,
        }
        
        # Detect image statistics drift
        if 'image_means' in batch_data:
            mean_r, mean_g, mean_b, std_r, std_g, std_b = batch_data['image_means']
            results['image_statistics_drift'] = self.detect_image_statistics_drift(
                mean_r, mean_g, mean_b, std_r, std_g, std_b
            )
        
        # Detect class distribution drift
        if 'class_counts' in batch_data:
            results['class_distribution_drift'] = self.detect_class_distribution_drift(
                batch_data['class_counts']
            )
        
        # Detect confidence drift
        if 'confidences' in batch_data:
            results['confidence_drift'] = self.detect_confidence_drift(
                batch_data['confidences']
            )
        
        # Detect embedding drift
        if 'embeddings' in batch_data and batch_data['embeddings'] is not None:
            results['embedding_drift'] = self.detect_embedding_drift(
                batch_data['embeddings']
            )
        
        # Check if any drift was detected
        results['overall_drift_detected'] = any(
            result.get('drift_detected', False)
            for result in results.values()
            if isinstance(result, dict)
        )
        
        return results


if __name__ == "__main__":
    """
    Test the drift detector.
    
    Usage:
        python -m src.deployment.drift_detector
    """
    print("Testing ImageDriftDetector...")
    
    try:
        # Try to create drift detector (will fail if baseline stats don't exist)
        detector = ImageDriftDetector()
        print("Drift detector created successfully!")
        
        # Test with some dummy data
        test_data = {
            'image_means': (0.5, 0.5, 0.5, 0.25, 0.25, 0.25),
            'class_counts': {0: 50, 1: 50},
            'confidences': [0.5, 0.6, 0.4, 0.7, 0.3] * 20,
        }
        
        results = detector.detect_drift(test_data)
        print(f"\nDrift detection results:")
        print(f"  Overall drift detected: {results['overall_drift_detected']}")
        if results['image_statistics_drift']:
            print(f"  Image stats drift: {results['image_statistics_drift']['drift_detected']}")
        if results['class_distribution_drift']:
            print(f"  Class distribution drift: {results['class_distribution_drift']['drift_detected']}")
        if results['confidence_drift']:
            print(f"  Confidence drift: {results['confidence_drift']['drift_detected']}")
        
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Please train a model first to generate baseline statistics.")
