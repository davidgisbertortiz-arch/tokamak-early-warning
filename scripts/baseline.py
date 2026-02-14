#!/usr/bin/env python3
"""
Baseline model training and evaluation script.

This script:
1. Loads the MIT-PSFC Open Density Limit dataset
2. Splits by discharge_ID (preventing temporal leakage)
3. Trains a LogisticRegression baseline with balanced class weights
4. Reports ROC-AUC, PR-AUC, and classification metrics

Usage:
    python scripts/baseline.py

Prerequisites:
    ./scripts/fetch_data.sh  (to download data first)
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import (
    load_density_limit_data,
    split_by_discharge,
    get_features_target,
    print_split_stats,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)
from src.models.baseline_lr import train_baseline
from src.evaluation.metrics import evaluate_model, print_evaluation_report
from tokamak_early_warning.config import DEFAULT_DATA_PATH, DEFAULT_SEED, set_global_seed


def main():
    print("="*60)
    print(" Tokamak Early Warning - Baseline Model")
    print("="*60)
    
    # --- Load Data ---
    print("\n[1/4] Loading dataset...")
    set_global_seed(DEFAULT_SEED)
    try:
        df = load_density_limit_data(DEFAULT_DATA_PATH)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    print(f"  Loaded {len(df):,} samples from {df['discharge_ID'].nunique()} discharges")
    print(f"  Features: {FEATURE_COLUMNS}")
    print(f"  Target: {TARGET_COLUMN}")
    print(f"  Overall positive rate: {df[TARGET_COLUMN].mean()*100:.2f}%")
    
    # --- Split by Discharge ID ---
    print("\n[2/4] Splitting by discharge_ID (70/15/15)...")
    print("  (This prevents temporal leakage between train/val/test)")
    
    train_df, val_df, test_df = split_by_discharge(
        df, 
        train_frac=0.70, 
        val_frac=0.15, 
        test_frac=0.15,
        random_state=DEFAULT_SEED
    )
    
    print_split_stats(train_df, val_df, test_df)
    
    # --- Prepare Features ---
    X_train, y_train = get_features_target(train_df)
    X_val, y_val = get_features_target(val_df)
    X_test, y_test = get_features_target(test_df)
    
    # --- Train Model ---
    print("\n[3/4] Training LogisticRegression (class_weight='balanced')...")
    model = train_baseline(X_train, y_train, max_iter=500, random_state=DEFAULT_SEED)
    print("  Training complete!")
    
    # --- Evaluate ---
    print("\n[4/4] Evaluating model...")
    
    # Validation set
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    val_metrics = evaluate_model(y_val, y_val_pred, y_val_prob)
    print_evaluation_report(val_metrics, y_val, y_val_pred, "Validation")
    
    # Test set
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_prob)
    print_evaluation_report(test_metrics, y_test, y_test_pred, "Test")
    
    # --- Summary ---
    print("\n" + "="*60)
    print(" Summary")
    print("="*60)
    print(f"\n  Validation PR-AUC: {val_metrics['pr_auc']:.4f}")
    print(f"  Test PR-AUC:       {test_metrics['pr_auc']:.4f}")
    print(f"\n  Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(f"  Test ROC-AUC:       {test_metrics['roc_auc']:.4f}")
    print("\n  Note: PR-AUC is the primary metric due to class imbalance (~1-2% positive)")
    print("="*60)


if __name__ == "__main__":
    main()
