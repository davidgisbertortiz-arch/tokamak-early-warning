"""
Probability calibration utilities.

Provides Platt scaling, isotonic regression, and calibration metrics.
Well-calibrated probabilities are essential for reliable uncertainty estimates.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from typing import Tuple, Optional
import warnings


def calibrate_probabilities(
    y_cal: np.ndarray,
    prob_cal: np.ndarray,
    method: str = "isotonic"
) -> IsotonicRegression:
    """
    Fit a calibration model on calibration set probabilities.
    
    Args:
        y_cal: Ground truth labels from calibration set
        prob_cal: Predicted probabilities from calibration set
        method: Calibration method ('isotonic' or 'platt')
        
    Returns:
        Fitted calibrator that maps uncalibrated -> calibrated probabilities
    """
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(prob_cal, y_cal)
        return calibrator
    elif method == "platt":
        # Platt scaling: logistic regression on probabilities
        from sklearn.linear_model import LogisticRegression
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(prob_cal.reshape(-1, 1), y_cal)
        return calibrator
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def apply_calibration(
    calibrator,
    probabilities: np.ndarray,
    method: str = "isotonic"
) -> np.ndarray:
    """
    Apply fitted calibrator to new probabilities.
    
    Args:
        calibrator: Fitted calibration model
        probabilities: Uncalibrated probabilities
        method: Calibration method used
        
    Returns:
        Calibrated probabilities
    """
    if method == "isotonic":
        return calibrator.predict(probabilities)
    elif method == "platt":
        return calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, dict]:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures how well predicted probabilities match actual frequencies.
    Lower is better; < 0.05 is typically considered well-calibrated.
    
    Args:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities for positive class
        n_bins: Number of bins for calibration
        
    Returns:
        Tuple of (ECE value, dict with per-bin details for plotting)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_details = {
        "bin_edges": bin_edges,
        "bin_accuracies": [],
        "bin_confidences": [],
        "bin_counts": []
    }
    
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        # Find samples in this bin
        if i == n_bins - 1:
            # Include right edge for last bin
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        else:
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        
        bin_count = mask.sum()
        bin_details["bin_counts"].append(bin_count)
        
        if bin_count > 0:
            bin_accuracy = y_true[mask].mean()  # Actual positive rate
            bin_confidence = y_prob[mask].mean()  # Mean predicted probability
            
            bin_details["bin_accuracies"].append(bin_accuracy)
            bin_details["bin_confidences"].append(bin_confidence)
            
            # Weighted contribution to ECE
            ece += (bin_count / total_samples) * abs(bin_accuracy - bin_confidence)
        else:
            bin_details["bin_accuracies"].append(np.nan)
            bin_details["bin_confidences"].append(np.nan)
    
    return ece, bin_details


def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> dict:
    """
    Prepare data for reliability diagram plotting.
    
    Args:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        Dictionary with plotting data
    """
    ece, bin_details = expected_calibration_error(y_true, y_prob, n_bins)
    
    return {
        "ece": ece,
        "bin_midpoints": (bin_details["bin_edges"][:-1] + bin_details["bin_edges"][1:]) / 2,
        "bin_accuracies": np.array(bin_details["bin_accuracies"]),
        "bin_confidences": np.array(bin_details["bin_confidences"]),
        "bin_counts": np.array(bin_details["bin_counts"]),
        "perfect_calibration": np.linspace(0, 1, n_bins)
    }
