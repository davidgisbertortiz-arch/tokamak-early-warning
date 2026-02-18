"""Tests for src/uncertainty/calibration.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.uncertainty.calibration import (
    calibrate_probabilities,
    apply_calibration,
    expected_calibration_error,
    reliability_diagram_data,
    TemperatureScaling,
    print_calibration_report,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def cal_data():
    """Simulated calibration data."""
    np.random.seed(0)
    n = 200
    prob = np.random.beta(2, 5, n)  # skewed toward low
    y = (np.random.rand(n) < prob).astype(int)
    return y, prob


# ── calibrate / apply ────────────────────────────────────────────────────────

def test_isotonic_calibrate_apply(cal_data):
    y, prob = cal_data
    calibrator = calibrate_probabilities(y, prob, method="isotonic")
    calibrated = apply_calibration(calibrator, prob, method="isotonic")
    assert calibrated.shape == prob.shape
    assert np.all(calibrated >= 0) and np.all(calibrated <= 1)


def test_platt_calibrate_apply(cal_data):
    y, prob = cal_data
    calibrator = calibrate_probabilities(y, prob, method="platt")
    calibrated = apply_calibration(calibrator, prob, method="platt")
    assert calibrated.shape == prob.shape
    assert np.all(calibrated >= 0) and np.all(calibrated <= 1)


def test_unknown_method(cal_data):
    y, prob = cal_data
    with pytest.raises(ValueError):
        calibrate_probabilities(y, prob, method="unknown")


def test_apply_unknown_method(cal_data):
    y, prob = cal_data
    calibrator = calibrate_probabilities(y, prob, method="isotonic")
    with pytest.raises(ValueError):
        apply_calibration(calibrator, prob, method="unknown")


# ── ECE ──────────────────────────────────────────────────────────────────────

def test_ece_perfect_calibration():
    """Perfect calibration → ECE should be very small."""
    np.random.seed(42)
    n = 10000
    prob = np.random.rand(n)
    y = (np.random.rand(n) < prob).astype(int)
    ece, details = expected_calibration_error(y, prob, n_bins=10)
    assert ece < 0.05


def test_ece_worst_case():
    """All predictions = 0 but labels = 1  → ECE should be high."""
    y = np.ones(100, dtype=int)
    prob = np.zeros(100)
    ece, details = expected_calibration_error(y, prob, n_bins=10)
    assert ece > 0.5


def test_ece_bins_structure(cal_data):
    y, prob = cal_data
    ece, details = expected_calibration_error(y, prob, n_bins=5)
    assert len(details["bin_edges"]) == 6
    assert len(details["bin_accuracies"]) == 5
    assert len(details["bin_confidences"]) == 5
    assert len(details["bin_counts"]) == 5


# ── reliability_diagram_data ─────────────────────────────────────────────────

def test_reliability_diagram_data(cal_data):
    y, prob = cal_data
    data = reliability_diagram_data(y, prob, n_bins=10)
    assert "ece" in data
    assert "bin_midpoints" in data
    assert len(data["bin_midpoints"]) == 10


# ── TemperatureScaling ───────────────────────────────────────────────────────

def test_temperature_scaling_fit_and_calibrate():
    np.random.seed(1)
    logits = np.random.randn(200)
    y = (logits > 0).astype(int)

    ts = TemperatureScaling()
    ts.fit(y, logits)
    assert ts._fitted
    assert ts.temperature > 0

    calibrated = ts.calibrate(logits)
    assert calibrated.shape == logits.shape
    assert np.all(calibrated >= 0) and np.all(calibrated <= 1)


def test_temperature_scaling_calibrate_probs():
    np.random.seed(1)
    logits = np.random.randn(100)
    y = (logits > 0).astype(int)

    ts = TemperatureScaling()
    ts.fit(y, logits)

    probs = 1 / (1 + np.exp(-logits))
    calibrated = ts.calibrate_probs(probs)
    assert calibrated.shape == probs.shape
    assert np.all(calibrated >= 0) and np.all(calibrated <= 1)


def test_temperature_scaling_not_fitted():
    ts = TemperatureScaling()
    with pytest.raises(RuntimeError):
        ts.calibrate(np.array([0.0, 1.0]))


# ── print_calibration_report ─────────────────────────────────────────────────

def test_print_calibration_report(cal_data, capsys):
    y, prob = cal_data
    calibrator = calibrate_probabilities(y, prob, method="isotonic")
    calibrated = apply_calibration(calibrator, prob, method="isotonic")

    print_calibration_report(y, prob, calibrated, n_bins=10)
    out = capsys.readouterr().out
    assert "Calibration Report" in out
    assert "Original ECE" in out
    assert "Calibrated ECE" in out


def test_print_calibration_report_no_calibrated(cal_data, capsys):
    y, prob = cal_data
    print_calibration_report(y, prob)
    out = capsys.readouterr().out
    assert "Original ECE" in out
