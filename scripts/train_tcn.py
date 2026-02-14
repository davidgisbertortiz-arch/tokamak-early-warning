#!/usr/bin/env python3
"""
Train TCN (Temporal Convolutional Network) model for tokamak early warning.

This script:
1. Loads the MIT-PSFC Open Density Limit dataset
2. Engineers physics-informed temporal features
3. Splits by discharge_ID (preventing temporal leakage)
4. Creates windowed sequences for temporal modeling
5. Trains TCN with class-weighted loss
6. Calibrates probabilities using temperature scaling
7. Evaluates with PR-AUC, ROC-AUC, and alarm-level metrics
8. Saves model and results

Usage:
    python scripts/train_tcn.py
    python scripts/train_tcn.py --seq-len 30 --epochs 50
    python scripts/train_tcn.py --help

Prerequisites:
    pip install -r requirements.txt
    ./scripts/fetch_data.sh
"""

import sys
import argparse
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# Core modules
from src.data.dataset import (
    load_density_limit_data,
    split_by_discharge_with_cal,
    TARGET_COLUMN
)
from src.features.temporal import (
    engineer_features,
    get_engineered_feature_names,
    FeatureConfig
)
from src.models.tcn import (
    TCNClassifier,
    TCNConfig,
    create_sequences,
    train_tcn_model
)
from src.evaluation.metrics import evaluate_model, print_evaluation_report
from src.evaluation.alarm_metrics import (
    AlarmConfig,
    compute_alarm_metrics,
    compute_tradeoff_curve,
    find_optimal_config,
    print_alarm_metrics_report,
    metrics_to_dict
)
from src.uncertainty.calibration import (
    calibrate_probabilities,
    apply_calibration,
    expected_calibration_error,
    TemperatureScaling,
    print_calibration_report
)
from tokamak_early_warning.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_SEED,
    set_global_seed,
    utc_timestamp,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train TCN model for tokamak early warning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data
    parser.add_argument(
        "--data-path", type=str, default=DEFAULT_DATA_PATH,
        help="Path to HDF5 dataset"
    )
    
    # Feature engineering
    parser.add_argument(
        "--rolling-windows", type=int, nargs="+", default=[3, 5, 10],
        help="Rolling window sizes for feature engineering"
    )
    
    # TCN architecture
    parser.add_argument(
        "--seq-len", type=int, default=20,
        help="Sequence length (temporal window size)"
    )
    parser.add_argument(
        "--hidden-channels", type=int, default=32,
        help="Hidden channels in TCN"
    )
    parser.add_argument(
        "--n-blocks", type=int, default=3,
        help="Number of TCN blocks"
    )
    parser.add_argument(
        "--kernel-size", type=int, default=3,
        help="Convolution kernel size"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2,
        help="Dropout rate"
    )
    
    # Training
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--pos-weight", type=float, default=10.0,
        help="Positive class weight for loss"
    )
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Early stopping patience"
    )
    
    # Calibration
    parser.add_argument(
        "--calibration-method", type=str, default="temperature",
        choices=["temperature", "isotonic", "platt", "none"],
        help="Probability calibration method"
    )
    
    # Output
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for saving results"
    )
    parser.add_argument(
        "--save-model", action="store_true",
        help="Save trained model"
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Random seed"
    )
    
    # Verbosity
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet
    
    print("=" * 70)
    print(" Tokamak Early Warning - TCN Training")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seeds
    set_global_seed(args.seed)
    
    # --- Load Data ---
    print("\n[1/7] Loading dataset...")
    try:
        df = load_density_limit_data(args.data_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    print(f"  Loaded {len(df):,} samples from {df['discharge_ID'].nunique()} discharges")
    print(f"  Overall positive rate: {df[TARGET_COLUMN].mean()*100:.2f}%")
    
    # --- Feature Engineering ---
    print("\n[2/7] Engineering temporal features...")
    feature_config = FeatureConfig(
        rolling_windows=tuple(args.rolling_windows),
        include_greenwald_proxy=True,
        include_interactions=True
    )
    
    df_engineered = engineer_features(df, feature_config, verbose=verbose)
    feature_cols = get_engineered_feature_names(feature_config)
    
    print(f"  Generated {len(feature_cols)} features")
    
    # --- Split by Discharge ID ---
    print("\n[3/7] Splitting by discharge_ID (60/10/15/15)...")
    print("  (This prevents temporal leakage between train/cal/val/test)")
    
    train_df, cal_df, val_df, test_df = split_by_discharge_with_cal(
        df_engineered,
        train_frac=0.60,
        cal_frac=0.10,
        val_frac=0.15,
        test_frac=0.15,
        random_state=args.seed
    )
    
    # Print split stats
    for name, split_df in [("Train", train_df), ("Cal", cal_df), ("Val", val_df), ("Test", test_df)]:
        n_discharges = split_df["discharge_ID"].nunique()
        n_samples = len(split_df)
        pos_rate = split_df[TARGET_COLUMN].mean() * 100
        n_events = sum(1 for _, g in split_df.groupby("discharge_ID") if g[TARGET_COLUMN].any())
        print(f"  {name}: {n_discharges:4d} discharges, {n_samples:6,d} samples, "
              f"{pos_rate:.2f}% pos, {n_events} events")
    
    # --- Create Sequences ---
    print("\n[4/7] Creating windowed sequences...")
    
    tcn_config = TCNConfig(
        seq_len=args.seq_len,
        hidden_channels=args.hidden_channels,
        n_blocks=args.n_blocks,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        n_epochs=args.epochs,
        patience=args.patience,
        pos_weight=args.pos_weight,
        random_state=args.seed
    )
    
    X_train, y_train, _ = create_sequences(train_df, feature_cols, TARGET_COLUMN, tcn_config.seq_len)
    X_cal, y_cal, _ = create_sequences(cal_df, feature_cols, TARGET_COLUMN, tcn_config.seq_len)
    X_val, y_val, _ = create_sequences(val_df, feature_cols, TARGET_COLUMN, tcn_config.seq_len)
    X_test, y_test, discharge_ids_test = create_sequences(test_df, feature_cols, TARGET_COLUMN, tcn_config.seq_len)
    
    print(f"  Train: {X_train.shape[0]:,} sequences")
    print(f"  Cal:   {X_cal.shape[0]:,} sequences")
    print(f"  Val:   {X_val.shape[0]:,} sequences")
    print(f"  Test:  {X_test.shape[0]:,} sequences")
    print(f"  Sequence shape: {X_train.shape[1:]} (seq_len={args.seq_len}, n_features={len(feature_cols)})")
    
    # --- Train TCN ---
    print("\n[5/7] Training TCN model...")
    
    classifier = TCNClassifier(config=tcn_config, verbose=verbose)
    classifier.feature_cols = feature_cols
    classifier.fit(X_train, y_train, X_val, y_val)
    
    # Get predictions
    y_train_prob = classifier.predict_proba(X_train)[:, 1]
    y_cal_prob = classifier.predict_proba(X_cal)[:, 1]
    y_val_prob = classifier.predict_proba(X_val)[:, 1]
    y_test_prob = classifier.predict_proba(X_test)[:, 1]
    
    # --- Calibration ---
    print("\n[6/7] Calibrating probabilities...")
    
    if args.calibration_method == "temperature":
        # Temperature scaling (best for neural networks)
        # Convert probabilities to logits for temperature scaling
        y_cal_prob_clipped = np.clip(y_cal_prob, 1e-7, 1 - 1e-7)
        cal_logits = np.log(y_cal_prob_clipped / (1 - y_cal_prob_clipped))
        
        temp_scaler = TemperatureScaling()
        temp_scaler.fit(y_cal, cal_logits)
        print(f"  Temperature: {temp_scaler.temperature:.3f}")
        
        # Apply to test set
        y_test_prob_clipped = np.clip(y_test_prob, 1e-7, 1 - 1e-7)
        test_logits = np.log(y_test_prob_clipped / (1 - y_test_prob_clipped))
        y_test_prob_cal = temp_scaler.calibrate(test_logits)
        
    elif args.calibration_method in ["isotonic", "platt"]:
        calibrator = calibrate_probabilities(y_cal, y_cal_prob, method=args.calibration_method)
        y_test_prob_cal = apply_calibration(calibrator, y_test_prob, method=args.calibration_method)
    else:
        y_test_prob_cal = y_test_prob
    
    # Print calibration report
    print_calibration_report(y_test, y_test_prob, y_test_prob_cal)
    
    # --- Evaluate ---
    print("\n[7/7] Evaluating model...")
    
    # Sample-level metrics (uncalibrated)
    y_test_pred = (y_test_prob >= 0.5).astype(int)
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_prob)
    print_evaluation_report(test_metrics, y_test, y_test_pred, "Test (Uncalibrated)")
    
    # Sample-level metrics (calibrated)
    y_test_pred_cal = (y_test_prob_cal >= 0.5).astype(int)
    test_metrics_cal = evaluate_model(y_test, y_test_pred_cal, y_test_prob_cal)
    print_evaluation_report(test_metrics_cal, y_test, y_test_pred_cal, "Test (Calibrated)")
    
    # --- Alarm-level Evaluation ---
    print("\n" + "=" * 70)
    print(" Alarm-Level Evaluation")
    print("=" * 70)
    
    # Create test DataFrame with predictions for alarm evaluation
    # We need to map sequences back to the original test_df
    # Since sequences drop the first (seq_len-1) samples per discharge, we need to align
    
    test_df_for_alarm = []
    seq_idx = 0
    for discharge_id in test_df["discharge_ID"].unique():
        discharge_df = test_df[test_df["discharge_ID"] == discharge_id].sort_values("time").copy()
        n_seq = len(discharge_df) - tcn_config.seq_len + 1
        if n_seq > 0:
            # First (seq_len-1) samples have no prediction
            discharge_df = discharge_df.iloc[tcn_config.seq_len - 1:]
            discharge_df["prob"] = y_test_prob_cal[seq_idx:seq_idx + n_seq]
            test_df_for_alarm.append(discharge_df)
            seq_idx += n_seq
    
    test_df_alarm = pd.concat(test_df_for_alarm, axis=0)
    
    # Compute alarm metrics with default config
    alarm_config = AlarmConfig(threshold=0.5, n_consecutive=2, use_ema=True, ema_alpha=0.3)
    alarm_metrics = compute_alarm_metrics(
        test_df_alarm, 
        test_df_alarm["prob"].values,
        alarm_config
    )
    print_alarm_metrics_report(alarm_metrics, "Test Set Alarms")
    
    # Compute trade-off curves
    print("\n  Computing trade-off curves...")
    tradeoff_df = compute_tradeoff_curve(
        test_df_alarm,
        test_df_alarm["prob"].values,
        thresholds=np.linspace(0.1, 0.9, 17),
        n_consecutive_values=[1, 2, 3],
        use_ema=True,
        ema_alpha=0.3
    )
    
    # Find optimal configuration
    optimal = find_optimal_config(
        tradeoff_df,
        min_event_recall=0.8,
        max_false_alarms_per_discharge=5.0,
        prefer="warning_time"
    )
    
    print(f"\n  Optimal Configuration (targeting ≥80% recall, ≤5 FA/discharge):")
    print(f"    Threshold:       {optimal['threshold']:.2f}")
    print(f"    Consecutive:     {optimal['n_consecutive']}")
    print(f"    Event Recall:    {optimal['metrics']['event_recall']:.1%}")
    print(f"    FA/Discharge:    {optimal['metrics']['false_alarms_per_discharge']:.2f}")
    print(f"    Median Warning:  {optimal['metrics']['median_warning_time']:.4f}s")
    
    # Evaluate with optimal config
    optimal_config = AlarmConfig(
        threshold=optimal['threshold'],
        n_consecutive=optimal['n_consecutive'],
        use_ema=True,
        ema_alpha=0.3
    )
    optimal_alarm_metrics = compute_alarm_metrics(
        test_df_alarm,
        test_df_alarm["prob"].values,
        optimal_config
    )
    print_alarm_metrics_report(optimal_alarm_metrics, "Test Set Alarms (Optimal Config)")
    
    # --- Save Results ---
    print("\n" + "=" * 70)
    print(" Saving Results")
    print("=" * 70)
    
    timestamp = utc_timestamp()
    
    # Save metrics summary
    results = {
        "timestamp": timestamp,
        "config": {
            "seed": args.seed,
            "data_path": args.data_path,
            "seq_len": args.seq_len,
            "hidden_channels": args.hidden_channels,
            "n_blocks": args.n_blocks,
            "kernel_size": args.kernel_size,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "pos_weight": args.pos_weight,
            "calibration_method": args.calibration_method,
            "n_features": len(feature_cols),
        },
        "sample_metrics": {
            "pr_auc": test_metrics_cal["pr_auc"],
            "roc_auc": test_metrics_cal["roc_auc"],
            "ece_original": expected_calibration_error(y_test, y_test_prob)[0],
            "ece_calibrated": expected_calibration_error(y_test, y_test_prob_cal)[0],
        },
        "alarm_metrics_default": metrics_to_dict(alarm_metrics),
        "alarm_metrics_optimal": metrics_to_dict(optimal_alarm_metrics),
        "optimal_config": optimal,
    }
    
    results_path = output_dir / "tcn_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved results to {results_path}")
    
    # Save trade-off curves
    tradeoff_path = output_dir / "tcn_tradeoff_curves.csv"
    tradeoff_df.to_csv(tradeoff_path, index=False)
    print(f"  Saved trade-off curves to {tradeoff_path}")
    
    # Save model
    if args.save_model:
        model_path = output_dir / "tcn_model.pt"
        classifier.save(str(model_path))
        print(f"  Saved model to {model_path}")
    
    # --- Summary ---
    print("\n" + "=" * 70)
    print(" Summary")
    print("=" * 70)
    
    print(f"\n  Sample-Level Metrics:")
    print(f"    PR-AUC:           {test_metrics_cal['pr_auc']:.4f}")
    print(f"    ROC-AUC:          {test_metrics_cal['roc_auc']:.4f}")
    print(f"    ECE (calibrated): {results['sample_metrics']['ece_calibrated']:.4f}")
    
    print(f"\n  Alarm-Level Metrics (Optimal Config):")
    print(f"    Event Recall:     {optimal_alarm_metrics.event_recall:.1%}")
    print(f"    FA/Discharge:     {optimal_alarm_metrics.false_alarms_per_discharge:.2f}")
    print(f"    Median Warning:   {optimal_alarm_metrics.median_warning_time:.4f}s")
    
    print(f"\n  Results saved to: {output_dir}/")
    print("\n  Done!")
    
    return results


if __name__ == "__main__":
    main()
