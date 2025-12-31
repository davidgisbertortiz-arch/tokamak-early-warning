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


class TemperatureScaling:
    """
    Temperature scaling for neural network calibration.
    
    Learns a single temperature parameter T that scales logits:
    calibrated_prob = sigmoid(logit / T)
    
    T > 1 makes probabilities less confident (closer to 0.5)
    T < 1 makes probabilities more confident (closer to 0 or 1)
    
    Reference: Guo et al. (2017). "On Calibration of Modern Neural Networks"
    """
    
    def __init__(self):
        self.temperature: float = 1.0
        self._fitted = False
    
    def fit(
        self,
        y_true: np.ndarray,
        logits: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> "TemperatureScaling":
        """
        Fit temperature parameter to minimize negative log-likelihood.
        
        Args:
            y_true: Ground truth labels
            logits: Model logits (pre-sigmoid output)
            lr: Learning rate for optimization
            max_iter: Maximum iterations
            
        Returns:
            self
        """
        try:
            import torch
            import torch.nn as nn
            
            y_t = torch.tensor(y_true, dtype=torch.float32)
            logits_t = torch.tensor(logits, dtype=torch.float32)
            
            # Initialize temperature
            temperature = nn.Parameter(torch.ones(1))
            optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
            criterion = nn.BCEWithLogitsLoss()
            
            def closure():
                optimizer.zero_grad()
                scaled_logits = logits_t / temperature
                loss = criterion(scaled_logits, y_t)
                loss.backward()
                return loss
            
            optimizer.step(closure)
            self.temperature = temperature.item()
            
        except ImportError:
            # Fallback: grid search if PyTorch not available
            from scipy.optimize import minimize_scalar
            
            def nll(T):
                if T <= 0:
                    return np.inf
                scaled_probs = 1 / (1 + np.exp(-logits / T))
                scaled_probs = np.clip(scaled_probs, 1e-7, 1 - 1e-7)
                return -np.mean(y_true * np.log(scaled_probs) + 
                               (1 - y_true) * np.log(1 - scaled_probs))
            
            result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
            self.temperature = result.x
        
        self._fitted = True
        return self
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Model logits
            
        Returns:
            Calibrated probabilities
        """
        if not self._fitted:
            raise RuntimeError("TemperatureScaling not fitted. Call fit() first.")
        
        scaled_logits = logits / self.temperature
        return 1 / (1 + np.exp(-scaled_logits))
    
    def calibrate_probs(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to probabilities.
        
        First converts probs to logits, then applies scaling.
        
        Args:
            probs: Model probabilities
            
        Returns:
            Calibrated probabilities
        """
        # Convert probs to logits
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = np.log(probs / (1 - probs))
        return self.calibrate(logits)


def print_calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_prob_calibrated: np.ndarray = None,
    n_bins: int = 10
) -> None:
    """
    Print calibration metrics report.
    
    Args:
        y_true: Ground truth labels
        y_prob: Original predicted probabilities
        y_prob_calibrated: Calibrated probabilities (optional)
        n_bins: Number of bins for ECE
    """
    print(f"\n{'='*60}")
    print(" Calibration Report")
    print(f"{'='*60}")
    
    ece_orig, _ = expected_calibration_error(y_true, y_prob, n_bins)
    print(f"\n  Original ECE:     {ece_orig:.4f}")
    
    if y_prob_calibrated is not None:
        ece_cal, _ = expected_calibration_error(y_true, y_prob_calibrated, n_bins)
        print(f"  Calibrated ECE:   {ece_cal:.4f}")
        print(f"  ECE improvement:  {(ece_orig - ece_cal) / ece_orig * 100:.1f}%")
    
    # Interpretation
    print(f"\n  Interpretation:")
    if ece_orig < 0.05:
        print("    Original model is well-calibrated (ECE < 0.05)")
    elif ece_orig < 0.10:
        print("    Original model has moderate calibration (0.05 < ECE < 0.10)")
    else:
        print("    Original model is poorly calibrated (ECE > 0.10)")
        print("    Consider using calibrated probabilities for alarm decisions")
