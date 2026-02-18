"""Tests for src/uncertainty/conformal.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.uncertainty.conformal import (
    ConformalClassifier,
    ConformalResult,
    print_coverage_report,
)
# Also exercises src/uncertainty/__init__
from src.uncertainty import ConformalClassifier as ConformalFromPkg


# ── Helpers ──────────────────────────────────────────────────────────────────

@pytest.fixture
def trained_model():
    """A simple fitted logistic-regression pipeline."""
    np.random.seed(42)
    X = np.random.randn(300, 4)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=42)),
    ])
    pipe.fit(X, y)
    return pipe, X, y


@pytest.fixture
def cal_test_split(trained_model):
    """Split data into cal / test."""
    model, X, y = trained_model
    X_cal, y_cal = X[:100], y[:100]
    X_test, y_test = X[100:], y[100:]
    return model, X_cal, y_cal, X_test, y_test


# ── ConformalResult properties ───────────────────────────────────────────────

def test_conformal_result_properties():
    sets = [{0}, {1}, {0, 1}]
    r = ConformalResult(
        prediction_sets=sets,
        probabilities=np.array([0.1, 0.9, 0.5]),
        scores=np.array([0.9, 0.1, 0.5]),
        threshold=0.5,
    )
    np.testing.assert_array_equal(r.predictions, [0, 1, 1])
    np.testing.assert_array_equal(r.is_uncertain, [False, False, True])
    np.testing.assert_array_equal(r.set_sizes, [1, 1, 2])


# ── Calibration & prediction ────────────────────────────────────────────────

def test_calibrate_and_predict(cal_test_split):
    model, X_cal, y_cal, X_test, y_test = cal_test_split
    cc = ConformalClassifier(model, alpha=0.10)

    assert not cc._is_calibrated
    cc.calibrate(X_cal, y_cal)
    assert cc._is_calibrated
    assert cc.threshold is not None

    result = cc.predict(X_test)
    assert len(result.prediction_sets) == len(X_test)
    assert all(isinstance(s, set) for s in result.prediction_sets)
    # Each set must be non-empty and subset of {0, 1}
    for s in result.prediction_sets:
        assert len(s) >= 1
        assert s.issubset({0, 1})


def test_predict_before_calibrate(cal_test_split):
    model, X_cal, y_cal, X_test, _ = cal_test_split
    cc = ConformalClassifier(model, alpha=0.10)
    with pytest.raises(RuntimeError):
        cc.predict(X_test)


def test_coverage_guarantee(cal_test_split):
    """Marginal coverage should be >= 1-alpha (in expectation)."""
    model, X_cal, y_cal, X_test, y_test = cal_test_split
    cc = ConformalClassifier(model, alpha=0.10)
    cc.calibrate(X_cal, y_cal)
    result = cc.predict(X_test)

    covered = np.array([y_test[i] in result.prediction_sets[i]
                        for i in range(len(y_test))])
    # Allow small slack for finite-sample randomness
    assert covered.mean() >= 0.85


def test_unknown_score_type(cal_test_split):
    model, X_cal, y_cal, *_ = cal_test_split
    cc = ConformalClassifier(model, alpha=0.10, score_type="bad")
    with pytest.raises(ValueError):
        cc.calibrate(X_cal, y_cal)


# ── evaluate_coverage ────────────────────────────────────────────────────────

def test_evaluate_coverage(cal_test_split):
    model, X_cal, y_cal, X_test, y_test = cal_test_split
    cc = ConformalClassifier(model, alpha=0.10)
    cc.calibrate(X_cal, y_cal)

    metrics = cc.evaluate_coverage(X_test, y_test)
    assert "global_coverage" in metrics
    assert "avg_set_size" in metrics
    assert "pct_uncertain" in metrics


def test_evaluate_coverage_per_discharge(cal_test_split):
    model, X_cal, y_cal, X_test, y_test = cal_test_split
    discharge_ids = np.repeat([1, 2], len(X_test) // 2)
    if len(discharge_ids) < len(X_test):
        discharge_ids = np.append(discharge_ids, [2] * (len(X_test) - len(discharge_ids)))

    cc = ConformalClassifier(model, alpha=0.10)
    cc.calibrate(X_cal, y_cal)
    metrics = cc.evaluate_coverage(X_test, y_test, discharge_ids=discharge_ids)

    assert "per_discharge" in metrics
    pd_m = metrics["per_discharge"]
    assert "mean_coverage" in pd_m
    assert "pct_below_target" in pd_m


# ── print_coverage_report ────────────────────────────────────────────────────

def test_print_coverage_report(cal_test_split, capsys):
    model, X_cal, y_cal, X_test, y_test = cal_test_split
    cc = ConformalClassifier(model, alpha=0.10)
    cc.calibrate(X_cal, y_cal)
    discharge_ids = np.ones(len(X_test), dtype=int)
    metrics = cc.evaluate_coverage(X_test, y_test, discharge_ids=discharge_ids)

    print_coverage_report(metrics, "Test")
    out = capsys.readouterr().out
    assert "Conformal Coverage Report" in out
    assert "Global Coverage" in out
    assert "Per-Discharge" in out


def test_print_coverage_report_no_discharge(cal_test_split, capsys):
    model, X_cal, y_cal, X_test, y_test = cal_test_split
    cc = ConformalClassifier(model, alpha=0.10)
    cc.calibrate(X_cal, y_cal)
    metrics = cc.evaluate_coverage(X_test, y_test)

    print_coverage_report(metrics, "Val")
    out = capsys.readouterr().out
    assert "Conformal Coverage Report" in out
