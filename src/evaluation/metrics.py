"""
Evaluation metrics for imbalanced classification.

Primary metric: PR-AUC (Average Precision) - most important for imbalanced data
Secondary metric: ROC-AUC - complementary view of model performance
"""

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_curve
)
import numpy as np
from typing import Dict, Any


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None
) -> Dict[str, Any]:
    """
    Compute evaluation metrics for binary classification.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (0/1)
        y_prob: Predicted probabilities for positive class (optional)
        
    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}
    
    # Probability-based metrics (if available)
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
        
        # Store curves for plotting
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
        
        metrics["pr_curve"] = {"precision": precision, "recall": recall}
        metrics["roc_curve"] = {"fpr": fpr, "tpr": tpr}
    
    # Classification report as dict
    metrics["classification_report"] = classification_report(
        y_true, y_pred, output_dict=True
    )
    
    return metrics


def print_evaluation_report(
    metrics: Dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str = "Test"
) -> None:
    """
    Print formatted evaluation report.
    
    Args:
        metrics: Dictionary from evaluate_model()
        y_true: Ground truth labels
        y_pred: Predicted labels
        dataset_name: Name of the dataset (for display)
    """
    print(f"\n{'='*60}")
    print(f" {dataset_name} Set Evaluation")
    print(f"{'='*60}")
    
    if "roc_auc" in metrics:
        print(f"\nROC-AUC:  {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:   {metrics['pr_auc']:.4f}  (Primary metric for imbalanced data)")
    
    print(f"\n{'-'*60}")
    print("Classification Report:")
    print(f"{'-'*60}")
    print(classification_report(y_true, y_pred))
