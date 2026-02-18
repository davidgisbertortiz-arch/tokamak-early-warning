"""Tests for src/evaluation/metrics.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.evaluation.metrics import evaluate_model, print_evaluation_report


@pytest.fixture
def binary_data():
    """Simple binary classification results."""
    np.random.seed(0)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.15, 0.6, 0.05, 0.9, 0.85, 0.4, 0.75, 0.8])
    return y_true, y_pred, y_prob


def test_evaluate_model_with_probs(binary_data):
    y_true, y_pred, y_prob = binary_data
    metrics = evaluate_model(y_true, y_pred, y_prob)

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "pr_curve" in metrics
    assert "roc_curve" in metrics
    assert "classification_report" in metrics

    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["pr_auc"] <= 1


def test_evaluate_model_without_probs(binary_data):
    y_true, y_pred, _ = binary_data
    metrics = evaluate_model(y_true, y_pred)

    assert "roc_auc" not in metrics
    assert "classification_report" in metrics


def test_print_evaluation_report(binary_data, capsys):
    y_true, y_pred, y_prob = binary_data
    metrics = evaluate_model(y_true, y_pred, y_prob)
    print_evaluation_report(metrics, y_true, y_pred, "Test")
    captured = capsys.readouterr()
    assert "Test Set Evaluation" in captured.out
    assert "ROC-AUC" in captured.out
    assert "PR-AUC" in captured.out


def test_print_evaluation_report_no_probs(binary_data, capsys):
    y_true, y_pred, _ = binary_data
    metrics = evaluate_model(y_true, y_pred)
    print_evaluation_report(metrics, y_true, y_pred, "Val")
    captured = capsys.readouterr()
    assert "Val Set Evaluation" in captured.out
