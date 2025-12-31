"""
Tests for temporal feature engineering.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.features.temporal import (
    compute_temporal_derivatives,
    compute_rolling_statistics,
    compute_greenwald_proxy,
    compute_interaction_features,
    engineer_features,
    get_engineered_feature_names,
    FeatureConfig,
    BASE_FEATURES
)


@pytest.fixture
def sample_discharge_df():
    """Create a sample discharge DataFrame."""
    np.random.seed(42)
    n_points = 50
    
    df = pd.DataFrame({
        "discharge_ID": 1,
        "time": np.linspace(0, 1, n_points),
        "density": np.linspace(1, 2, n_points) + np.random.randn(n_points) * 0.1,
        "elongation": np.ones(n_points) * 1.5 + np.random.randn(n_points) * 0.05,
        "minor_radius": np.ones(n_points) * 0.5 + np.random.randn(n_points) * 0.01,
        "plasma_current": np.ones(n_points) * 1e6 + np.random.randn(n_points) * 1e4,
        "toroidal_B_field": np.ones(n_points) * 2.5 + np.random.randn(n_points) * 0.1,
        "triangularity": np.ones(n_points) * 0.3 + np.random.randn(n_points) * 0.05,
        "density_limit_phase": 0
    })
    
    return df


def test_temporal_derivatives_shape(sample_discharge_df):
    """Test that derivatives have correct shape."""
    deriv_df = compute_temporal_derivatives(sample_discharge_df, BASE_FEATURES)
    
    assert len(deriv_df) == len(sample_discharge_df)
    assert all(f"d_{col}_dt" in deriv_df.columns for col in BASE_FEATURES)


def test_temporal_derivatives_sign():
    """Test derivative sign for known increasing signal."""
    df = pd.DataFrame({
        "time": [0.0, 0.1, 0.2, 0.3, 0.4],
        "density": [1.0, 2.0, 3.0, 4.0, 5.0],  # Steadily increasing
    })
    
    deriv_df = compute_temporal_derivatives(df, ["density"])
    
    # After first point, derivative should be positive (10.0)
    assert deriv_df["d_density_dt"].iloc[1:].mean() > 0


def test_rolling_statistics_shape(sample_discharge_df):
    """Test rolling statistics output shape."""
    windows = (3, 5)
    roll_df = compute_rolling_statistics(sample_discharge_df, BASE_FEATURES, windows)
    
    assert len(roll_df) == len(sample_discharge_df)
    
    # Check all expected columns exist
    for col in BASE_FEATURES:
        for w in windows:
            assert f"{col}_roll_mean_{w}" in roll_df.columns
            assert f"{col}_roll_std_{w}" in roll_df.columns


def test_greenwald_proxy_computation(sample_discharge_df):
    """Test Greenwald proxy computation."""
    proxy_df = compute_greenwald_proxy(sample_discharge_df)
    
    assert "greenwald_limit_proxy" in proxy_df.columns
    assert "greenwald_fraction" in proxy_df.columns
    assert "d_greenwald_frac_dt" in proxy_df.columns
    
    # Values should be finite (no inf from division)
    assert np.isfinite(proxy_df["greenwald_fraction"].dropna()).all()


def test_interaction_features_computation(sample_discharge_df):
    """Test interaction features."""
    interact_df = compute_interaction_features(sample_discharge_df)
    
    expected_cols = [
        "density_per_current",
        "density_times_B",
        "elongation_times_triangularity",
        "current_per_B",
        "density_times_a2"
    ]
    
    for col in expected_cols:
        assert col in interact_df.columns


def test_engineer_features_no_cross_discharge_leakage():
    """
    Test that feature engineering doesn't leak across discharges.
    
    Rolling stats should reset at discharge boundaries.
    """
    # Create two discharges with very different values
    df = pd.DataFrame({
        "discharge_ID": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "time": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        "density": [1, 1, 1, 1, 1, 100, 100, 100, 100, 100],
        "elongation": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "minor_radius": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "plasma_current": [1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6],
        "toroidal_B_field": [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
        "triangularity": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        "density_limit_phase": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    })
    
    config = FeatureConfig(rolling_windows=(3,))
    engineered_df = engineer_features(df, config)
    
    # Rolling mean for discharge 2 should be ~100, not influenced by discharge 1
    discharge_2_density_mean = engineered_df[
        engineered_df["discharge_ID"] == 2
    ]["density_roll_mean_3"].iloc[-1]
    
    assert discharge_2_density_mean > 50  # Should be ~100, not mixed with discharge 1


def test_get_engineered_feature_names():
    """Test that feature name list is complete."""
    config = FeatureConfig(
        rolling_windows=(3, 5),
        include_greenwald_proxy=True,
        include_interactions=True
    )
    
    feature_names = get_engineered_feature_names(config)
    
    # Should include base features
    for col in BASE_FEATURES:
        assert col in feature_names
    
    # Should include derivatives
    for col in BASE_FEATURES:
        assert f"d_{col}_dt" in feature_names
    
    # Should include rolling stats
    assert "density_roll_mean_3" in feature_names
    assert "density_roll_mean_5" in feature_names
    
    # Should include greenwald proxy
    assert "greenwald_fraction" in feature_names
    
    # Should include interactions
    assert "density_per_current" in feature_names


def test_feature_config_defaults():
    """Test default feature configuration."""
    config = FeatureConfig()
    
    assert config.rolling_windows == (3, 5, 10)
    assert config.include_greenwald_proxy == True
    assert config.include_interactions == True
