"""
Tests for data splitting and leakage prevention.

Critical test: Ensure no discharge_ID appears in multiple splits,
which would cause temporal leakage.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data.dataset import (
    split_by_discharge,
    split_by_discharge_with_cal,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    
    # Create 20 discharges with varying lengths
    data = []
    for discharge_id in range(1, 21):
        n_samples = np.random.randint(50, 150)
        times = np.linspace(0, 1, n_samples)
        
        # Create features
        for t in times:
            data.append({
                "discharge_ID": discharge_id,
                "time": t,
                "density": np.random.randn(),
                "elongation": np.random.randn(),
                "minor_radius": np.random.randn() + 1,
                "plasma_current": np.random.randn() * 1e6,
                "toroidal_B_field": np.random.randn() + 2,
                "triangularity": np.random.randn() * 0.3,
                "density_limit_phase": 1 if t > 0.8 and np.random.random() > 0.3 else 0
            })
    
    df = pd.DataFrame(data)
    return df


def test_split_by_discharge_no_leakage(sample_df):
    """
    Test that no discharge_ID appears in multiple splits.
    
    This is CRITICAL for preventing temporal leakage:
    time points from the same discharge must stay together.
    """
    train_df, val_df, test_df = split_by_discharge(sample_df, random_state=42)
    
    train_ids = set(train_df["discharge_ID"].unique())
    val_ids = set(val_df["discharge_ID"].unique())
    test_ids = set(test_df["discharge_ID"].unique())
    
    # Check no overlap
    assert train_ids.isdisjoint(val_ids), "Train and val have overlapping discharge_IDs!"
    assert train_ids.isdisjoint(test_ids), "Train and test have overlapping discharge_IDs!"
    assert val_ids.isdisjoint(test_ids), "Val and test have overlapping discharge_IDs!"
    
    # Check all discharges are accounted for
    all_ids = train_ids | val_ids | test_ids
    original_ids = set(sample_df["discharge_ID"].unique())
    assert all_ids == original_ids, "Some discharge_IDs are missing from splits!"


def test_split_by_discharge_with_cal_no_leakage(sample_df):
    """
    Test that 4-way split has no discharge_ID overlap.
    """
    train_df, cal_df, val_df, test_df = split_by_discharge_with_cal(sample_df, random_state=42)
    
    train_ids = set(train_df["discharge_ID"].unique())
    cal_ids = set(cal_df["discharge_ID"].unique())
    val_ids = set(val_df["discharge_ID"].unique())
    test_ids = set(test_df["discharge_ID"].unique())
    
    # Check pairwise disjoint
    all_sets = [("train", train_ids), ("cal", cal_ids), ("val", val_ids), ("test", test_ids)]
    for i, (name1, set1) in enumerate(all_sets):
        for name2, set2 in all_sets[i+1:]:
            assert set1.isdisjoint(set2), f"{name1} and {name2} have overlapping discharge_IDs!"


def test_split_fractions_approximate(sample_df):
    """Test that split fractions are approximately correct."""
    train_df, val_df, test_df = split_by_discharge(
        sample_df, 
        train_frac=0.70, 
        val_frac=0.15, 
        test_frac=0.15,
        random_state=42
    )
    
    n_total = sample_df["discharge_ID"].nunique()
    n_train = train_df["discharge_ID"].nunique()
    n_val = val_df["discharge_ID"].nunique()
    n_test = test_df["discharge_ID"].nunique()
    
    # Allow some tolerance due to integer rounding
    assert abs(n_train / n_total - 0.70) < 0.1
    assert abs(n_val / n_total - 0.15) < 0.1
    assert abs(n_test / n_total - 0.15) < 0.1


def test_split_reproducibility(sample_df):
    """Test that splits are reproducible with same random_state."""
    train1, val1, test1 = split_by_discharge(sample_df, random_state=42)
    train2, val2, test2 = split_by_discharge(sample_df, random_state=42)
    
    assert set(train1["discharge_ID"].unique()) == set(train2["discharge_ID"].unique())
    assert set(val1["discharge_ID"].unique()) == set(val2["discharge_ID"].unique())
    assert set(test1["discharge_ID"].unique()) == set(test2["discharge_ID"].unique())


def test_split_different_seeds_different_results(sample_df):
    """Test that different seeds produce different splits."""
    train1, _, _ = split_by_discharge(sample_df, random_state=42)
    train2, _, _ = split_by_discharge(sample_df, random_state=123)
    
    # Different seeds should (usually) produce different splits
    # There's a tiny chance they're the same, but very unlikely
    assert set(train1["discharge_ID"].unique()) != set(train2["discharge_ID"].unique())
