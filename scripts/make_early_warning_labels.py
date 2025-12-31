#!/usr/bin/env python3
"""
Create early warning labels for prediction ahead of density limit events.

This script transforms the density_limit_phase label into an "early warning" label:
- For each discharge, find the first time point where density_limit_phase == 1
- Create label=1 for time points within [t_event - dt, t_event) before the event
- Optionally remove all time points after the event starts

This enables the model to predict disruptions BEFORE they happen, rather than
just detecting them as they occur.

Usage:
    python scripts/make_early_warning_labels.py --dt 0.1

Args:
    --dt: Warning horizon in seconds (default: 0.1s)
    --keep-post-event: Keep samples after event starts (default: remove them)
    --output: Output file path (default: data/processed/DL_early_warning.h5)
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.data.dataset import load_density_limit_data


def create_early_warning_labels(
    df: pd.DataFrame,
    dt: float = 0.1,
    keep_post_event: bool = False
) -> pd.DataFrame:
    """
    Transform density_limit_phase into early warning labels.
    
    Args:
        df: DataFrame with discharge_ID, time, and density_limit_phase columns
        dt: Warning horizon in seconds (predict this far ahead)
        keep_post_event: If False, remove samples after event starts
        
    Returns:
        DataFrame with new 'early_warning' column
    """
    result_dfs = []
    
    for discharge_id in df["discharge_ID"].unique():
        discharge_df = df[df["discharge_ID"] == discharge_id].copy()
        discharge_df = discharge_df.sort_values("time")
        
        # Find first event time (if any)
        event_mask = discharge_df["density_limit_phase"] == 1
        
        if event_mask.any():
            t_event = discharge_df.loc[event_mask, "time"].iloc[0]
            
            # Create early warning label: 1 if within [t_event - dt, t_event)
            time_to_event = t_event - discharge_df["time"]
            discharge_df["early_warning"] = (
                (time_to_event <= dt) & (time_to_event > 0)
            ).astype(int)
            
            # Optionally remove post-event samples (they're trivial to classify)
            if not keep_post_event:
                discharge_df = discharge_df[discharge_df["time"] < t_event]
        else:
            # No event in this discharge - all zeros
            discharge_df["early_warning"] = 0
        
        result_dfs.append(discharge_df)
    
    return pd.concat(result_dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Create early warning labels for density limit prediction"
    )
    parser.add_argument(
        "--dt", type=float, default=0.1,
        help="Warning horizon in seconds (default: 0.1)"
    )
    parser.add_argument(
        "--keep-post-event", action="store_true",
        help="Keep samples after event starts (default: remove them)"
    )
    parser.add_argument(
        "--output", type=str, default="data/processed/DL_early_warning.h5",
        help="Output file path"
    )
    args = parser.parse_args()
    
    print("="*60)
    print(" Creating Early Warning Labels")
    print("="*60)
    
    # Load data
    print("\n[1/3] Loading dataset...")
    df = load_density_limit_data("data/raw/DL_DataFrame.h5")
    print(f"  Loaded {len(df):,} samples from {df['discharge_ID'].nunique()} discharges")
    
    # Create labels
    print(f"\n[2/3] Creating early warning labels (dt={args.dt}s)...")
    df_ew = create_early_warning_labels(
        df, 
        dt=args.dt, 
        keep_post_event=args.keep_post_event
    )
    
    # Stats
    original_events = df["density_limit_phase"].sum()
    new_warnings = df_ew["early_warning"].sum()
    print(f"  Original density_limit_phase=1: {original_events:,}")
    print(f"  New early_warning=1: {new_warnings:,}")
    print(f"  Samples after transformation: {len(df_ew):,}")
    print(f"  New positive rate: {df_ew['early_warning'].mean()*100:.2f}%")
    
    # Save
    print(f"\n[3/3] Saving to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_ew.to_hdf(args.output, key="data", mode="w")
    print(f"  Saved {len(df_ew):,} samples")
    
    print("\n" + "="*60)
    print(" Done! You can now train on the early warning labels.")
    print("="*60)


if __name__ == "__main__":
    main()
