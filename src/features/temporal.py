"""
Physics-informed temporal feature engineering for tokamak early warning.

Creates features that capture temporal dynamics and physics-relevant proxies
that precede density limit / MARFE-like instabilities.

Key feature categories:
1. Temporal derivatives (d/dt) - rate of change precedes instabilities
2. Rolling statistics - capture short-term fluctuations and trends
3. Greenwald-like density proxy - operational limit indicator
4. Physics-motivated interactions - capture coupled dynamics

References:
- Greenwald, M. (2002). Density limits in toroidal plasmas. PPCF 44, R27.
- MARFE: Multifaceted Asymmetric Radiation From the Edge
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for temporal feature engineering."""
    # Rolling window sizes (in number of samples)
    rolling_windows: Tuple[int, ...] = (3, 5, 10)
    
    # Derivative estimation method
    derivative_method: str = "finite_diff"  # "finite_diff" or "savgol"
    
    # Whether to include interaction features
    include_interactions: bool = True
    
    # Whether to include Greenwald-like proxy
    include_greenwald_proxy: bool = True
    
    # Fill method for NaN values from rolling operations
    fill_method: str = "bfill"  # "bfill", "ffill", or "zero"


# Original feature columns from the dataset
BASE_FEATURES = [
    "density",
    "elongation", 
    "minor_radius",
    "plasma_current",
    "toroidal_B_field",
    "triangularity",
]


def compute_temporal_derivatives(
    df: pd.DataFrame,
    columns: List[str],
    time_col: str = "time"
) -> pd.DataFrame:
    """
    Compute temporal derivatives (d/dt) for specified columns.
    
    Physical motivation: Rate of change often precedes instabilities.
    Rising density derivative can indicate approach to density limit.
    
    Args:
        df: DataFrame sorted by time within each discharge
        columns: Columns to differentiate
        time_col: Name of time column
        
    Returns:
        DataFrame with derivative columns (d_{col}_dt)
    """
    result = pd.DataFrame(index=df.index)
    
    for col in columns:
        # Finite difference: d(col)/dt = (col[i] - col[i-1]) / (t[i] - t[i-1])
        dt = df[time_col].diff()
        dcol = df[col].diff()
        
        # Avoid division by zero; use small epsilon
        dt = dt.replace(0, np.nan)
        derivative = dcol / dt
        
        result[f"d_{col}_dt"] = derivative
    
    return result


def compute_rolling_statistics(
    df: pd.DataFrame,
    columns: List[str],
    windows: Tuple[int, ...] = (3, 5, 10)
) -> pd.DataFrame:
    """
    Compute rolling window statistics for specified columns.
    
    Physical motivation:
    - Rolling mean: smoothed trend, reduces noise
    - Rolling std: fluctuation level, high std may precede instability
    - Rolling min/max: extrema detection
    - Rolling slope: trend direction over window
    
    Args:
        df: DataFrame with feature columns
        columns: Columns to compute statistics for
        windows: Window sizes (in number of samples)
        
    Returns:
        DataFrame with rolling statistic columns
    """
    result = pd.DataFrame(index=df.index)
    
    for col in columns:
        for w in windows:
            # Rolling mean
            result[f"{col}_roll_mean_{w}"] = df[col].rolling(w, min_periods=1).mean()
            
            # Rolling standard deviation (fluctuation level)
            result[f"{col}_roll_std_{w}"] = df[col].rolling(w, min_periods=1).std()
            
            # Rolling min/max for extrema
            result[f"{col}_roll_min_{w}"] = df[col].rolling(w, min_periods=1).min()
            result[f"{col}_roll_max_{w}"] = df[col].rolling(w, min_periods=1).max()
    
    return result


def compute_greenwald_proxy(
    df: pd.DataFrame,
    density_col: str = "density",
    current_col: str = "plasma_current",
    minor_radius_col: str = "minor_radius"
) -> pd.DataFrame:
    """
    Compute Greenwald-like density limit proxy.
    
    The Greenwald density limit is: n_G = I_p / (π * a²)
    where I_p is plasma current and a is minor radius.
    
    The Greenwald fraction f_G = n / n_G indicates proximity to the limit:
    - f_G < 0.8: typically safe
    - f_G > 0.8-1.0: approaching limit, increased risk
    - f_G > 1.0: exceeds empirical limit (but can be sustained in some conditions)
    
    This proxy may precede density-limit disruptions as it approaches ~1.
    
    Note: Exact units matter for real n_G calculation. Here we compute a
    dimensionless proxy that captures the same physics trend.
    
    Args:
        df: DataFrame with density, plasma_current, minor_radius columns
        
    Returns:
        DataFrame with Greenwald proxy columns
    """
    result = pd.DataFrame(index=df.index)
    
    # Greenwald limit proxy: I_p / (π * a²)
    # Use absolute value of current (can be negative depending on direction)
    a_squared = df[minor_radius_col] ** 2
    
    # Avoid division by zero
    a_squared = a_squared.replace(0, np.nan)
    
    n_greenwald_proxy = np.abs(df[current_col]) / (np.pi * a_squared)
    
    # Greenwald fraction: actual density / Greenwald limit
    # This should approach 1 near the density limit
    result["greenwald_limit_proxy"] = n_greenwald_proxy
    
    # Avoid division by zero for fraction
    n_greenwald_proxy = n_greenwald_proxy.replace(0, np.nan)
    result["greenwald_fraction"] = df[density_col] / n_greenwald_proxy
    
    # Rate of change of Greenwald fraction (precursor signal)
    result["d_greenwald_frac_dt"] = result["greenwald_fraction"].diff()
    
    return result


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute physics-motivated interaction features.
    
    Physical motivation:
    - density / plasma_current: related to Greenwald physics
    - density * B_field: relates to beta (plasma pressure / magnetic pressure)
    - elongation * triangularity: shape coupling
    - current / B_field: q-like safety factor proxy
    
    These interactions capture coupled dynamics that single features miss.
    
    Args:
        df: DataFrame with base feature columns
        
    Returns:
        DataFrame with interaction feature columns
    """
    result = pd.DataFrame(index=df.index)
    
    # Density to current ratio (Greenwald-related)
    result["density_per_current"] = df["density"] / (np.abs(df["plasma_current"]) + 1e-10)
    
    # Density times B-field (beta-related proxy)
    result["density_times_B"] = df["density"] * np.abs(df["toroidal_B_field"])
    
    # Shape coupling
    result["elongation_times_triangularity"] = df["elongation"] * df["triangularity"]
    
    # Safety factor proxy: q ~ a * B / (R * I_p) 
    # Simplified: current / B ratio (inverse q-like)
    result["current_per_B"] = np.abs(df["plasma_current"]) / (np.abs(df["toroidal_B_field"]) + 1e-10)
    
    # Density times minor radius squared (relates to total particle content)
    result["density_times_a2"] = df["density"] * (df["minor_radius"] ** 2)
    
    return result


def engineer_features_per_discharge(
    df: pd.DataFrame,
    config: FeatureConfig = None
) -> pd.DataFrame:
    """
    Engineer temporal features for a single discharge.
    
    This function should be called per discharge to ensure temporal
    derivatives and rolling statistics don't cross discharge boundaries.
    
    Args:
        df: DataFrame for a single discharge, sorted by time
        config: Feature engineering configuration
        
    Returns:
        DataFrame with all engineered features
    """
    if config is None:
        config = FeatureConfig()
    
    result = df.copy()
    
    # 1. Temporal derivatives
    deriv_df = compute_temporal_derivatives(df, BASE_FEATURES)
    result = pd.concat([result, deriv_df], axis=1)
    
    # 2. Rolling statistics
    roll_df = compute_rolling_statistics(df, BASE_FEATURES, config.rolling_windows)
    result = pd.concat([result, roll_df], axis=1)
    
    # 3. Greenwald proxy
    if config.include_greenwald_proxy:
        greenwald_df = compute_greenwald_proxy(df)
        result = pd.concat([result, greenwald_df], axis=1)
    
    # 4. Interaction features
    if config.include_interactions:
        interact_df = compute_interaction_features(df)
        result = pd.concat([result, interact_df], axis=1)
    
    # Handle NaN values from rolling/derivative operations
    if config.fill_method == "bfill":
        result = result.bfill()
    elif config.fill_method == "ffill":
        result = result.ffill()
    elif config.fill_method == "zero":
        result = result.fillna(0)
    
    # Final fillna with 0 for any remaining NaN
    result = result.fillna(0)
    
    return result


def engineer_features(
    df: pd.DataFrame,
    config: FeatureConfig = None,
    discharge_col: str = "discharge_ID",
    verbose: bool = False
) -> pd.DataFrame:
    """
    Engineer temporal features for entire dataset, respecting discharge boundaries.
    
    CRITICAL: Features are computed per-discharge to prevent cross-discharge
    leakage in temporal operations (derivatives, rolling stats).
    
    Args:
        df: Full DataFrame with all discharges
        config: Feature engineering configuration
        discharge_col: Name of discharge ID column
        verbose: Print progress
        
    Returns:
        DataFrame with all engineered features, preserving original index
    """
    if config is None:
        config = FeatureConfig()
    
    discharge_ids = df[discharge_col].unique()
    
    if verbose:
        print(f"Engineering features for {len(discharge_ids)} discharges...")
    
    result_dfs = []
    
    for i, discharge_id in enumerate(discharge_ids):
        discharge_df = df[df[discharge_col] == discharge_id].sort_values("time")
        engineered = engineer_features_per_discharge(discharge_df, config)
        result_dfs.append(engineered)
        
        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(discharge_ids)} discharges")
    
    result = pd.concat(result_dfs, axis=0)
    
    # Restore original index order
    result = result.loc[df.index]
    
    if verbose:
        print(f"  Feature engineering complete. Shape: {result.shape}")
    
    return result


def get_engineered_feature_names(config: FeatureConfig = None) -> List[str]:
    """
    Get list of all engineered feature names.
    
    Useful for selecting features after engineering.
    
    Args:
        config: Feature engineering configuration
        
    Returns:
        List of feature column names
    """
    if config is None:
        config = FeatureConfig()
    
    features = list(BASE_FEATURES)  # Start with base features
    
    # Derivative features
    for col in BASE_FEATURES:
        features.append(f"d_{col}_dt")
    
    # Rolling statistics
    for col in BASE_FEATURES:
        for w in config.rolling_windows:
            features.extend([
                f"{col}_roll_mean_{w}",
                f"{col}_roll_std_{w}",
                f"{col}_roll_min_{w}",
                f"{col}_roll_max_{w}",
            ])
    
    # Greenwald proxy
    if config.include_greenwald_proxy:
        features.extend([
            "greenwald_limit_proxy",
            "greenwald_fraction",
            "d_greenwald_frac_dt",
        ])
    
    # Interaction features
    if config.include_interactions:
        features.extend([
            "density_per_current",
            "density_times_B",
            "elongation_times_triangularity",
            "current_per_B",
            "density_times_a2",
        ])
    
    return features


def create_feature_documentation() -> Dict[str, str]:
    """
    Create documentation for all engineered features.
    
    Returns:
        Dictionary mapping feature names to descriptions
    """
    docs = {}
    
    # Base features
    docs["density"] = "Plasma electron density"
    docs["elongation"] = "Plasma cross-section elongation (κ)"
    docs["minor_radius"] = "Plasma minor radius (a)"
    docs["plasma_current"] = "Plasma current (I_p)"
    docs["toroidal_B_field"] = "Toroidal magnetic field strength"
    docs["triangularity"] = "Plasma cross-section triangularity (δ)"
    
    # Derivatives
    for col in BASE_FEATURES:
        docs[f"d_{col}_dt"] = f"Time derivative of {col}. Rising derivatives often precede instabilities."
    
    # Rolling stats (generic descriptions)
    docs["*_roll_mean_*"] = "Rolling mean - smoothed trend over window"
    docs["*_roll_std_*"] = "Rolling std - fluctuation level; high values may precede instability"
    docs["*_roll_min_*"] = "Rolling minimum - lower bound over window"
    docs["*_roll_max_*"] = "Rolling maximum - upper bound over window"
    
    # Greenwald proxy
    docs["greenwald_limit_proxy"] = (
        "Greenwald density limit proxy: I_p / (π * a²). "
        "Empirical limit for density in tokamaks."
    )
    docs["greenwald_fraction"] = (
        "Ratio of actual density to Greenwald limit. "
        "Values approaching 1.0 indicate proximity to density limit."
    )
    docs["d_greenwald_frac_dt"] = (
        "Rate of change of Greenwald fraction. "
        "Rapidly increasing values suggest imminent density limit approach."
    )
    
    # Interactions
    docs["density_per_current"] = (
        "Density normalized by plasma current. Related to Greenwald physics."
    )
    docs["density_times_B"] = (
        "Density times B-field. Proxy related to plasma beta (pressure ratio)."
    )
    docs["elongation_times_triangularity"] = (
        "Shape coupling term. Captures combined effect of plasma shaping."
    )
    docs["current_per_B"] = (
        "Current to B-field ratio. Inverse proxy for safety factor q."
    )
    docs["density_times_a2"] = (
        "Density times minor radius squared. Related to total particle inventory."
    )
    
    return docs
