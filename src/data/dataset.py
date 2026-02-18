"""
Dataset loading and splitting utilities for MIT-PSFC tokamak data.

CRITICAL: Always split by discharge_ID to prevent temporal leakage.
Time points within a discharge are temporally correlated — mixing them
across train/test would give unrealistically optimistic results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "density",
    "elongation",
    "minor_radius",
    "plasma_current",
    "toroidal_B_field",
    "triangularity",
]

TARGET_COLUMN = "density_limit_phase"


# ── Data loading ─────────────────────────────────────────────────────────────


def load_density_limit_data(path: str | Path = "data/raw/DL_DataFrame.h5") -> pd.DataFrame:
    """Load the density-limit HDF5 dataset.

    Parameters
    ----------
    path : str | Path
        Path to the HDF5 file.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least ``discharge_ID``, ``time``,
        the six feature columns, and ``density_limit_phase``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run  ./scripts/fetch_data.sh  first to download it."
        )
    df = pd.read_hdf(path)
    return df


# ── Splitting by discharge ───────────────────────────────────────────────────


def split_by_discharge(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split *df* into train / val / test **by discharge_ID**.

    Every sample from the same discharge ends up in the same split,
    preventing temporal leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``discharge_ID`` column.
    train_frac, val_frac, test_frac : float
        Target proportions (of discharges, not samples).  They are
        normalised internally so they don't need to sum to exactly 1.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train_df, val_df, test_df)``
    """
    total = train_frac + val_frac + test_frac
    train_frac, val_frac, test_frac = (
        train_frac / total,
        val_frac / total,
        test_frac / total,
    )

    discharge_ids = np.array(df["discharge_ID"].unique())
    rng = np.random.RandomState(random_state)
    rng.shuffle(discharge_ids)

    n = len(discharge_ids)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    # remainder goes to test
    train_ids = set(discharge_ids[:n_train])
    val_ids = set(discharge_ids[n_train : n_train + n_val])
    test_ids = set(discharge_ids[n_train + n_val :])

    train_df = df[df["discharge_ID"].isin(train_ids)].copy()
    val_df = df[df["discharge_ID"].isin(val_ids)].copy()
    test_df = df[df["discharge_ID"].isin(test_ids)].copy()

    return train_df, val_df, test_df


def split_by_discharge_with_cal(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    cal_frac: float = 0.10,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Four-way split: train / calibration / val / test by discharge_ID.

    The calibration split is used for conformal prediction / Platt scaling.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``discharge_ID`` column.
    train_frac, cal_frac, val_frac, test_frac : float
        Target proportions (of discharges).  Normalised internally.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train_df, cal_df, val_df, test_df)``
    """
    total = train_frac + cal_frac + val_frac + test_frac
    train_frac /= total
    cal_frac /= total
    val_frac /= total
    test_frac /= total

    discharge_ids = np.array(df["discharge_ID"].unique())
    rng = np.random.RandomState(random_state)
    rng.shuffle(discharge_ids)

    n = len(discharge_ids)
    n_train = int(round(n * train_frac))
    n_cal = int(round(n * cal_frac))
    n_val = int(round(n * val_frac))

    train_ids = set(discharge_ids[:n_train])
    cal_ids = set(discharge_ids[n_train : n_train + n_cal])
    val_ids = set(discharge_ids[n_train + n_cal : n_train + n_cal + n_val])
    test_ids = set(discharge_ids[n_train + n_cal + n_val :])

    train_df = df[df["discharge_ID"].isin(train_ids)].copy()
    cal_df = df[df["discharge_ID"].isin(cal_ids)].copy()
    val_df = df[df["discharge_ID"].isin(val_ids)].copy()
    test_df = df[df["discharge_ID"].isin(test_ids)].copy()

    return train_df, cal_df, val_df, test_df


# ── Helpers ──────────────────────────────────────────────────────────────────


def get_features_target(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = TARGET_COLUMN,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix *X* and label vector *y*.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str] | None
        Defaults to :data:`FEATURE_COLUMNS`.
    target_col : str
        Defaults to :data:`TARGET_COLUMN`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(X, y)`` with shapes ``(n, d)`` and ``(n,)``.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS
    X = df[feature_cols].values
    y = df[target_col].values
    return X, y


def print_split_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
) -> None:
    """Print a summary table for a 3-way split."""
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n_discharges = split_df["discharge_ID"].nunique()
        n_samples = len(split_df)
        pos_rate = split_df[target_col].mean() * 100 if len(split_df) > 0 else 0.0
        print(
            f"  {name}: {n_discharges:4d} discharges, "
            f"{n_samples:6,d} samples, "
            f"positive rate {pos_rate:.2f}%"
        )
