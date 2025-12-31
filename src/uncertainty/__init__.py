"""
Uncertainty quantification for tokamak early warning system.

Provides calibrated probabilities and conformal prediction sets
with per-discharge coverage guarantees.
"""

from .conformal import ConformalClassifier
from .calibration import calibrate_probabilities, expected_calibration_error

__all__ = [
    "ConformalClassifier",
    "calibrate_probabilities", 
    "expected_calibration_error",
]
