"""
Baseline Logistic Regression model for density limit prediction.

Uses class_weight="balanced" to handle severe class imbalance (~1-2% positive).
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
from typing import Tuple


def create_baseline_pipeline(
    max_iter: int = 500,
    random_state: int = 42
) -> Pipeline:
    """
    Create baseline pipeline with scaling and logistic regression.
    
    Args:
        max_iter: Maximum iterations for convergence
        random_state: Random seed for reproducibility
        
    Returns:
        sklearn Pipeline ready for fitting
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",  # Critical for imbalanced data
            random_state=random_state,
            solver="lbfgs"
        ))
    ])
    return pipeline


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_iter: int = 500,
    random_state: int = 42
) -> Pipeline:
    """
    Train the baseline logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        max_iter: Maximum iterations for convergence
        random_state: Random seed
        
    Returns:
        Fitted pipeline
    """
    pipeline = create_baseline_pipeline(max_iter=max_iter, random_state=random_state)
    pipeline.fit(X_train, y_train)
    return pipeline
