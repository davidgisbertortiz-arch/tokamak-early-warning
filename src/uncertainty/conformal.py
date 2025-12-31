"""
Split Conformal Prediction for binary classification.

Provides prediction sets with finite-sample coverage guarantees.
Key insight: prediction sets {0}, {1}, or {0,1} allow expressing uncertainty.

Reference: Vovk et al., "Algorithmic Learning in a Random World" (2005)
"""

import numpy as np
import pandas as pd
from typing import List, Set, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ConformalResult:
    """Container for conformal prediction results."""
    prediction_sets: List[Set[int]]  # {0}, {1}, or {0,1} for each sample
    probabilities: np.ndarray  # Calibrated or raw probabilities
    scores: np.ndarray  # Nonconformity scores
    threshold: float  # Calibrated threshold (1-alpha quantile)
    
    @property
    def predictions(self) -> np.ndarray:
        """Point predictions (1 if {1} or {0,1}, 0 if {0})."""
        return np.array([1 if 1 in s else 0 for s in self.prediction_sets])
    
    @property
    def is_uncertain(self) -> np.ndarray:
        """Boolean array: True where prediction set has size > 1."""
        return np.array([len(s) > 1 for s in self.prediction_sets])
    
    @property 
    def set_sizes(self) -> np.ndarray:
        """Size of each prediction set (1 or 2)."""
        return np.array([len(s) for s in self.prediction_sets])


class ConformalClassifier:
    """
    Split Conformal Prediction wrapper for binary classifiers.
    
    Guarantees: P(y_true ∈ prediction_set) ≥ 1 - alpha (marginal coverage)
    
    Usage:
        conformal = ConformalClassifier(model, alpha=0.10)
        conformal.calibrate(X_cal, y_cal)
        result = conformal.predict(X_test)
        # result.prediction_sets contains {0}, {1}, or {0,1}
    """
    
    def __init__(
        self,
        model,
        alpha: float = 0.10,
        score_type: str = "lac"  # "lac" (least ambiguous) or "aps" (adaptive)
    ):
        """
        Args:
            model: Fitted sklearn classifier with predict_proba
            alpha: Miscoverage rate (1 - alpha = target coverage)
            score_type: Nonconformity score type
                - "lac": Least Ambiguous set-valued Classifier (simpler)
                - "aps": Adaptive Prediction Sets (tighter sets on average)
        """
        self.model = model
        self.alpha = alpha
        self.score_type = score_type
        self.threshold: Optional[float] = None
        self.cal_scores: Optional[np.ndarray] = None
        self._is_calibrated = False
    
    def _compute_scores(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Compute nonconformity scores.
        
        For LAC: score = 1 - p(y_true)
        Lower score = more conforming to training data
        """
        probs = self.model.predict_proba(X)
        
        if y is not None:
            # Calibration: use true labels
            if self.score_type == "lac":
                # Score = 1 - probability of true class
                scores = 1 - probs[np.arange(len(y)), y]
            else:
                raise ValueError(f"Unknown score type: {self.score_type}")
        else:
            # Prediction: return scores for both classes
            # score[i, c] = 1 - prob[i, c] for class c
            scores = 1 - probs
        
        return scores
    
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> "ConformalClassifier":
        """
        Calibrate conformal predictor on held-out calibration set.
        
        CRITICAL: X_cal, y_cal must NOT overlap with training data.
        For tokamak data, must be from different discharge_IDs.
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration labels
            
        Returns:
            self (for chaining)
        """
        y_cal = np.asarray(y_cal)
        self.cal_scores = self._compute_scores(X_cal, y_cal)
        
        # Threshold = (1-alpha)(1 + 1/n) quantile of calibration scores
        # This ensures finite-sample coverage guarantee
        n_cal = len(y_cal)
        q_level = np.ceil((1 - self.alpha) * (n_cal + 1)) / n_cal
        q_level = min(q_level, 1.0)  # Cap at 1.0
        
        self.threshold = np.quantile(self.cal_scores, q_level)
        self._is_calibrated = True
        
        return self
    
    def predict(self, X: np.ndarray) -> ConformalResult:
        """
        Generate prediction sets for new samples.
        
        Args:
            X: Features to predict
            
        Returns:
            ConformalResult with prediction sets and metadata
        """
        if not self._is_calibrated:
            raise RuntimeError("Must call calibrate() before predict()")
        
        # Get scores for both classes
        probs = self.model.predict_proba(X)
        scores = 1 - probs  # scores[i, c] for class c
        
        # Build prediction sets: include class c if score[c] <= threshold
        prediction_sets = []
        for i in range(len(X)):
            pred_set = set()
            for c in [0, 1]:
                if scores[i, c] <= self.threshold:
                    pred_set.add(c)
            
            # Ensure non-empty sets (include argmax if empty)
            if len(pred_set) == 0:
                pred_set.add(int(probs[i].argmax()))
            
            prediction_sets.append(pred_set)
        
        return ConformalResult(
            prediction_sets=prediction_sets,
            probabilities=probs[:, 1],  # P(Y=1)
            scores=scores[:, 1],  # Score for positive class
            threshold=self.threshold
        )
    
    def evaluate_coverage(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        discharge_ids: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Evaluate coverage on test set, including per-discharge coverage.
        
        Args:
            X: Test features
            y_true: Test labels
            discharge_ids: Optional discharge IDs for per-discharge coverage
            
        Returns:
            Dictionary with coverage metrics
        """
        result = self.predict(X)
        y_true = np.asarray(y_true)
        
        # Global coverage
        covered = np.array([y_true[i] in result.prediction_sets[i] 
                          for i in range(len(y_true))])
        
        metrics = {
            "global_coverage": covered.mean(),
            "target_coverage": 1 - self.alpha,
            "coverage_gap": covered.mean() - (1 - self.alpha),
            "avg_set_size": result.set_sizes.mean(),
            "pct_uncertain": result.is_uncertain.mean(),
            "n_samples": len(y_true)
        }
        
        # Per-discharge coverage (critical for temporal data)
        if discharge_ids is not None:
            discharge_ids = np.asarray(discharge_ids)
            unique_discharges = np.unique(discharge_ids)
            
            per_discharge_coverage = []
            per_discharge_n_samples = []
            
            for did in unique_discharges:
                mask = discharge_ids == did
                discharge_covered = covered[mask]
                per_discharge_coverage.append(discharge_covered.mean())
                per_discharge_n_samples.append(mask.sum())
            
            per_discharge_coverage = np.array(per_discharge_coverage)
            
            metrics["per_discharge"] = {
                "discharge_ids": unique_discharges,
                "coverages": per_discharge_coverage,
                "n_samples": np.array(per_discharge_n_samples),
                "mean_coverage": per_discharge_coverage.mean(),
                "std_coverage": per_discharge_coverage.std(),
                "min_coverage": per_discharge_coverage.min(),
                "pct_below_target": (per_discharge_coverage < (1 - self.alpha)).mean()
            }
        
        return metrics


def print_coverage_report(metrics: Dict[str, Any], dataset_name: str = "Test") -> None:
    """Print formatted coverage report."""
    print(f"\n{'='*60}")
    print(f" Conformal Coverage Report - {dataset_name}")
    print(f"{'='*60}")
    
    print(f"\n  Global Coverage:  {metrics['global_coverage']:.1%} "
          f"(target: {metrics['target_coverage']:.1%})")
    print(f"  Coverage Gap:     {metrics['coverage_gap']:+.1%}")
    print(f"  Avg Set Size:     {metrics['avg_set_size']:.2f}")
    print(f"  % Uncertain:      {metrics['pct_uncertain']:.1%}")
    
    if "per_discharge" in metrics:
        pd_metrics = metrics["per_discharge"]
        print(f"\n  Per-Discharge Coverage:")
        print(f"    Mean:           {pd_metrics['mean_coverage']:.1%}")
        print(f"    Std:            {pd_metrics['std_coverage']:.1%}")
        print(f"    Min:            {pd_metrics['min_coverage']:.1%}")
        print(f"    % Below Target: {pd_metrics['pct_below_target']:.1%}")
