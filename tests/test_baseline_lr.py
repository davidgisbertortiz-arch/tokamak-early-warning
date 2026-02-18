"""Tests for src/models/baseline_lr.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.models.baseline_lr import create_baseline_pipeline, train_baseline
from src.models import create_baseline_pipeline as pkg_create  # __init__ coverage


@pytest.fixture
def simple_dataset():
    np.random.seed(42)
    X = np.random.randn(200, 6)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_create_baseline_pipeline():
    pipe = create_baseline_pipeline(max_iter=100, random_state=0)
    assert hasattr(pipe, "fit")
    assert hasattr(pipe, "predict")


def test_train_baseline(simple_dataset):
    X, y = simple_dataset
    model = train_baseline(X, y, max_iter=200, random_state=42)
    preds = model.predict(X)
    proba = model.predict_proba(X)

    assert preds.shape == (200,)
    assert proba.shape == (200, 2)
    assert set(preds).issubset({0, 1})


def test_pkg_import():
    """Importing from src.models should work."""
    assert callable(pkg_create)
