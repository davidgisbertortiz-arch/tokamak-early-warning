#!/usr/bin/env python3
"""
Evaluate alarm policy on trained models.

This script:
1. Loads a trained model (TCN or baseline)
2. Computes predictions on test set
3. Evaluates alarm-level metrics with configurable policy
4. Generates trade-off curves and visualizations
5. Saves results to /results directory

Usage:
    python scripts/evaluate_alarm_policy.py --model tcn
    python scripts/evaluate_alarm_policy.py --model baseline --threshold 0.3
    python scripts/evaluate_alarm_policy.py --help

Prerequisites:
    - For TCN: run scripts/train_tcn.py first
    - For baseline: data available via scripts/fetch_data.sh
"""

import sys
import argparse
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Core modules
from src.data.dataset import (
    load_density_limit_data,
    split_by_discharge_with_cal,
    get_features_target,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)
from src.features.temporal import (
    engineer_features,
    get_engineered_feature_names,
    FeatureConfig
)
from src.models.baseline_lr import train_baseline
from src.models.tcn import TCNClassifier, TCNConfig, create_sequences
from src.evaluation.metrics import evaluate_model
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
    reliability_diagram_data
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
        description="Evaluate alarm policy on trained models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model selection
    parser.add_argument(
        "--model", type=str, default="both",
        choices=["tcn", "baseline", "both"],
        help="Model to evaluate"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to saved TCN model (optional)"
    )
    
    # Data
    parser.add_argument(
        "--data-path", type=str, default=DEFAULT_DATA_PATH,
        help="Path to HDF5 dataset"
    )
    
    # Alarm policy
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Probability threshold for alarm"
    )
    parser.add_argument(
        "--n-consecutive", type=int, default=2,
        help="Consecutive steps required for alarm"
    )
    parser.add_argument(
        "--use-ema", action="store_true", default=True,
        help="Use EMA smoothing"
    )
    parser.add_argument(
        "--ema-alpha", type=float, default=0.3,
        help="EMA smoothing factor"
    )
    
    # Output
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for saving results"
    )
    parser.add_argument(
        "--save-figures", action="store_true", default=True,
        help="Save visualization figures"
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Random seed"
    )
    
    return parser.parse_args()


def plot_pr_curve(metrics_dict: dict, output_path: Path, title: str = "Precision-Recall Curve"):
    """Plot and save precision-recall curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for name, metrics in metrics_dict.items():
        if "pr_curve" in metrics:
            precision = metrics["pr_curve"]["precision"]
            recall = metrics["pr_curve"]["recall"]
            pr_auc = metrics["pr_auc"]
            ax.plot(recall, precision, label=f"{name} (PR-AUC={pr_auc:.3f})", linewidth=2)
    
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_tradeoff_curves(tradeoff_df: pd.DataFrame, output_path: Path, model_name: str = "Model"):
    """Plot trade-off curves: threshold vs recall/FAR/warning time."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Filter for n_consecutive=2 (middle option)
    df_filtered = tradeoff_df[tradeoff_df["n_consecutive"] == 2]
    
    # Event Recall vs Threshold
    ax = axes[0]
    ax.plot(df_filtered["threshold"], df_filtered["event_recall"], 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Event Recall", fontsize=12)
    ax.set_title(f"{model_name}: Event Recall vs Threshold", fontsize=12)
    ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Target 80%')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 1])
    
    # False Alarm Rate vs Threshold
    ax = axes[1]
    ax.plot(df_filtered["threshold"], df_filtered["false_alarms_per_discharge"], 'r-o', linewidth=2, markersize=6)
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("False Alarms / Discharge", fontsize=12)
    ax.set_title(f"{model_name}: False Alarm Rate vs Threshold", fontsize=12)
    ax.axhline(y=5.0, color='b', linestyle='--', alpha=0.5, label='Target ≤5')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Warning Time vs Threshold
    ax = axes[2]
    ax.plot(df_filtered["threshold"], df_filtered["median_warning_time"], 'g-o', linewidth=2, markersize=6)
    ax.fill_between(
        df_filtered["threshold"],
        df_filtered["p25_warning_time"],
        df_filtered["p75_warning_time"],
        alpha=0.3, color='green', label='P25-P75'
    )
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Warning Time (seconds)", fontsize=12)
    ax.set_title(f"{model_name}: Warning Time vs Threshold", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_warning_time_histogram(warning_times: list, output_path: Path, model_name: str = "Model"):
    """Plot histogram of warning times."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.hist(warning_times, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=np.median(warning_times), color='red', linestyle='--', linewidth=2, 
               label=f'Median: {np.median(warning_times):.4f}s')
    ax.axvline(x=np.mean(warning_times), color='orange', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(warning_times):.4f}s')
    
    ax.set_xlabel("Warning Time (seconds)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"{model_name}: Warning Time Distribution", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_far_vs_recall(tradeoff_df: pd.DataFrame, output_path: Path, model_name: str = "Model"):
    """Plot FAR vs Recall trade-off curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for n_consec in tradeoff_df["n_consecutive"].unique():
        df_subset = tradeoff_df[tradeoff_df["n_consecutive"] == n_consec]
        ax.plot(df_subset["false_alarms_per_discharge"], df_subset["event_recall"],
                '-o', linewidth=2, markersize=6, label=f'n_consecutive={n_consec}')
    
    # Mark target region
    ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=5.0, color='r', linestyle='--', alpha=0.5)
    ax.fill_between([0, 5], [0.8, 0.8], [1, 1], alpha=0.1, color='green', label='Target Region')
    
    ax.set_xlabel("False Alarms / Discharge", fontsize=12)
    ax.set_ylabel("Event Recall", fontsize=12)
    ax.set_title(f"{model_name}: FAR vs Recall Trade-off", fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, y_prob_cal: np.ndarray,
                            output_path: Path, model_name: str = "Model"):
    """Plot reliability diagram for calibration."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, (probs, title) in zip(axes, [(y_prob, "Before Calibration"), 
                                          (y_prob_cal, "After Calibration")]):
        rel_data = reliability_diagram_data(y_true, probs, n_bins=10)
        
        # Plot bars
        valid_mask = ~np.isnan(rel_data["bin_accuracies"])
        ax.bar(rel_data["bin_midpoints"][valid_mask], 
               rel_data["bin_accuracies"][valid_mask],
               width=0.08, alpha=0.7, edgecolor='black', label='Accuracy')
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
        
        ax.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax.set_ylabel("Fraction of Positives", fontsize=12)
        ax.set_title(f"{model_name}: {title}\nECE = {rel_data['ece']:.4f}", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_baseline_model(df_test: pd.DataFrame, df_train: pd.DataFrame, df_cal: pd.DataFrame,
                            args, output_dir: Path) -> dict:
    """Evaluate baseline logistic regression model."""
    print("\n" + "=" * 70)
    print(" Evaluating Baseline (Logistic Regression)")
    print("=" * 70)
    
    # Train baseline model
    X_train, y_train = get_features_target(df_train)
    X_cal, y_cal = get_features_target(df_cal)
    X_test, y_test = get_features_target(df_test)
    
    print("\n  Training baseline model...")
    model = train_baseline(X_train, y_train, random_state=args.seed)
    
    # Get predictions
    y_test_prob = model.predict_proba(X_test)[:, 1]
    y_cal_prob = model.predict_proba(X_cal)[:, 1]
    
    # Calibrate
    print("  Calibrating probabilities (isotonic)...")
    calibrator = calibrate_probabilities(y_cal.values, y_cal_prob, method="isotonic")
    y_test_prob_cal = apply_calibration(calibrator, y_test_prob, method="isotonic")
    
    # Sample-level metrics
    y_test_pred = (y_test_prob_cal >= 0.5).astype(int)
    sample_metrics = evaluate_model(y_test.values, y_test_pred, y_test_prob_cal)
    
    print(f"\n  Sample-Level Metrics:")
    print(f"    PR-AUC:  {sample_metrics['pr_auc']:.4f}")
    print(f"    ROC-AUC: {sample_metrics['roc_auc']:.4f}")
    
    # Alarm evaluation
    df_test_copy = df_test.copy()
    df_test_copy["prob"] = y_test_prob_cal
    
    # Compute alarm metrics
    alarm_config = AlarmConfig(
        threshold=args.threshold,
        n_consecutive=args.n_consecutive,
        use_ema=args.use_ema,
        ema_alpha=args.ema_alpha
    )
    alarm_metrics = compute_alarm_metrics(df_test_copy, y_test_prob_cal, alarm_config)
    print_alarm_metrics_report(alarm_metrics, "Baseline Alarm Evaluation")
    
    # Trade-off curves
    print("\n  Computing trade-off curves...")
    tradeoff_df = compute_tradeoff_curve(
        df_test_copy, y_test_prob_cal,
        thresholds=np.linspace(0.1, 0.9, 17),
        n_consecutive_values=[1, 2, 3],
        use_ema=args.use_ema,
        ema_alpha=args.ema_alpha
    )
    
    # Find optimal config
    optimal = find_optimal_config(tradeoff_df, min_event_recall=0.8, max_false_alarms_per_discharge=5.0)
    
    # Evaluate with optimal config
    optimal_config = AlarmConfig(
        threshold=optimal["threshold"],
        n_consecutive=optimal["n_consecutive"],
        use_ema=args.use_ema,
        ema_alpha=args.ema_alpha
    )
    optimal_metrics = compute_alarm_metrics(df_test_copy, y_test_prob_cal, optimal_config)
    
    print(f"\n  Optimal Config (≥80% recall, ≤5 FA/discharge):")
    print(f"    Threshold:       {optimal['threshold']:.2f}")
    print(f"    Consecutive:     {optimal['n_consecutive']}")
    print(f"    Event Recall:    {optimal_metrics.event_recall:.1%}")
    print(f"    FA/Discharge:    {optimal_metrics.false_alarms_per_discharge:.2f}")
    print(f"    Median Warning:  {optimal_metrics.median_warning_time:.4f}s")
    
    # Save figures
    if args.save_figures:
        plot_tradeoff_curves(tradeoff_df, output_dir / "baseline_tradeoff.png", "Baseline LR")
        plot_far_vs_recall(tradeoff_df, output_dir / "baseline_far_vs_recall.png", "Baseline LR")
        if optimal_metrics.warning_times:
            plot_warning_time_histogram(optimal_metrics.warning_times, 
                                       output_dir / "baseline_warning_time_hist.png", "Baseline LR")
        plot_reliability_diagram(y_test.values, y_test_prob, y_test_prob_cal,
                                output_dir / "baseline_reliability.png", "Baseline LR")
    
    return {
        "sample_metrics": {
            "pr_auc": sample_metrics["pr_auc"],
            "roc_auc": sample_metrics["roc_auc"],
            "ece_original": expected_calibration_error(y_test.values, y_test_prob)[0],
            "ece_calibrated": expected_calibration_error(y_test.values, y_test_prob_cal)[0],
        },
        "alarm_metrics": metrics_to_dict(alarm_metrics),
        "alarm_metrics_optimal": metrics_to_dict(optimal_metrics),
        "optimal_config": optimal,
        "tradeoff_df": tradeoff_df,
    }


def evaluate_tcn_model(df_engineered: pd.DataFrame, train_df: pd.DataFrame, 
                       cal_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                       feature_cols: list, args, output_dir: Path) -> dict:
    """Evaluate TCN model."""
    print("\n" + "=" * 70)
    print(" Evaluating TCN Model")
    print("=" * 70)
    
    # TCN configuration
    seq_len = 20
    tcn_config = TCNConfig(
        seq_len=seq_len,
        hidden_channels=32,
        n_blocks=3,
        kernel_size=3,
        dropout=0.2,
        batch_size=256,
        learning_rate=1e-3,
        n_epochs=30,
        patience=5,
        pos_weight=10.0,
        random_state=args.seed
    )
    
    # Create sequences
    print("\n  Creating sequences...")
    X_train, y_train, _ = create_sequences(train_df, feature_cols, TARGET_COLUMN, seq_len)
    X_cal, y_cal, _ = create_sequences(cal_df, feature_cols, TARGET_COLUMN, seq_len)
    X_val, y_val, _ = create_sequences(val_df, feature_cols, TARGET_COLUMN, seq_len)
    X_test, y_test, _ = create_sequences(test_df, feature_cols, TARGET_COLUMN, seq_len)
    
    print(f"    Train: {X_train.shape[0]:,} sequences")
    print(f"    Test:  {X_test.shape[0]:,} sequences")
    
    # Check for saved model
    model_path = Path(args.model_path) if args.model_path else output_dir / "tcn_model.pt"
    
    if model_path.exists():
        print(f"\n  Loading saved model from {model_path}...")
        classifier = TCNClassifier.load(str(model_path))
    else:
        print("\n  Training TCN model...")
        classifier = TCNClassifier(config=tcn_config, verbose=True)
        classifier.feature_cols = feature_cols
        classifier.fit(X_train, y_train, X_val, y_val)
    
    # Get predictions
    y_test_prob = classifier.predict_proba(X_test)[:, 1]
    y_cal_prob = classifier.predict_proba(X_cal)[:, 1]
    
    # Calibrate
    print("\n  Calibrating probabilities (isotonic)...")
    calibrator = calibrate_probabilities(y_cal, y_cal_prob, method="isotonic")
    y_test_prob_cal = apply_calibration(calibrator, y_test_prob, method="isotonic")
    
    # Sample-level metrics
    y_test_pred = (y_test_prob_cal >= 0.5).astype(int)
    sample_metrics = evaluate_model(y_test, y_test_pred, y_test_prob_cal)
    
    print(f"\n  Sample-Level Metrics:")
    print(f"    PR-AUC:  {sample_metrics['pr_auc']:.4f}")
    print(f"    ROC-AUC: {sample_metrics['roc_auc']:.4f}")
    
    # Create test DataFrame for alarm evaluation
    test_df_for_alarm = []
    seq_idx = 0
    for discharge_id in test_df["discharge_ID"].unique():
        discharge_df = test_df[test_df["discharge_ID"] == discharge_id].sort_values("time").copy()
        n_seq = len(discharge_df) - seq_len + 1
        if n_seq > 0:
            discharge_df = discharge_df.iloc[seq_len - 1:]
            discharge_df["prob"] = y_test_prob_cal[seq_idx:seq_idx + n_seq]
            test_df_for_alarm.append(discharge_df)
            seq_idx += n_seq
    
    test_df_alarm = pd.concat(test_df_for_alarm, axis=0)
    
    # Alarm evaluation
    alarm_config = AlarmConfig(
        threshold=args.threshold,
        n_consecutive=args.n_consecutive,
        use_ema=args.use_ema,
        ema_alpha=args.ema_alpha
    )
    alarm_metrics = compute_alarm_metrics(test_df_alarm, test_df_alarm["prob"].values, alarm_config)
    print_alarm_metrics_report(alarm_metrics, "TCN Alarm Evaluation")
    
    # Trade-off curves
    print("\n  Computing trade-off curves...")
    tradeoff_df = compute_tradeoff_curve(
        test_df_alarm, test_df_alarm["prob"].values,
        thresholds=np.linspace(0.1, 0.9, 17),
        n_consecutive_values=[1, 2, 3],
        use_ema=args.use_ema,
        ema_alpha=args.ema_alpha
    )
    
    # Find optimal config
    optimal = find_optimal_config(tradeoff_df, min_event_recall=0.8, max_false_alarms_per_discharge=5.0)
    
    # Evaluate with optimal config
    optimal_config = AlarmConfig(
        threshold=optimal["threshold"],
        n_consecutive=optimal["n_consecutive"],
        use_ema=args.use_ema,
        ema_alpha=args.ema_alpha
    )
    optimal_metrics = compute_alarm_metrics(test_df_alarm, test_df_alarm["prob"].values, optimal_config)
    
    print(f"\n  Optimal Config (≥80% recall, ≤5 FA/discharge):")
    print(f"    Threshold:       {optimal['threshold']:.2f}")
    print(f"    Consecutive:     {optimal['n_consecutive']}")
    print(f"    Event Recall:    {optimal_metrics.event_recall:.1%}")
    print(f"    FA/Discharge:    {optimal_metrics.false_alarms_per_discharge:.2f}")
    print(f"    Median Warning:  {optimal_metrics.median_warning_time:.4f}s")
    
    # Save figures
    if args.save_figures:
        plot_tradeoff_curves(tradeoff_df, output_dir / "tcn_tradeoff.png", "TCN")
        plot_far_vs_recall(tradeoff_df, output_dir / "tcn_far_vs_recall.png", "TCN")
        if optimal_metrics.warning_times:
            plot_warning_time_histogram(optimal_metrics.warning_times,
                                       output_dir / "tcn_warning_time_hist.png", "TCN")
        plot_reliability_diagram(y_test, y_test_prob, y_test_prob_cal,
                                output_dir / "tcn_reliability.png", "TCN")
    
    return {
        "sample_metrics": {
            "pr_auc": sample_metrics["pr_auc"],
            "roc_auc": sample_metrics["roc_auc"],
            "ece_original": expected_calibration_error(y_test, y_test_prob)[0],
            "ece_calibrated": expected_calibration_error(y_test, y_test_prob_cal)[0],
        },
        "alarm_metrics": metrics_to_dict(alarm_metrics),
        "alarm_metrics_optimal": metrics_to_dict(optimal_metrics),
        "optimal_config": optimal,
        "tradeoff_df": tradeoff_df,
    }


def plot_comparison(baseline_results: dict, tcn_results: dict, output_dir: Path):
    """Create comparison plots between baseline and TCN."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = ["Baseline LR", "TCN"]
    
    # PR-AUC comparison
    ax = axes[0]
    pr_aucs = [baseline_results["sample_metrics"]["pr_auc"],
               tcn_results["sample_metrics"]["pr_auc"]]
    bars = ax.bar(models, pr_aucs, color=['steelblue', 'darkorange'], edgecolor='black')
    ax.set_ylabel("PR-AUC", fontsize=12)
    ax.set_title("Sample-Level: PR-AUC", fontsize=12)
    ax.set_ylim([0, max(pr_aucs) * 1.2])
    for bar, val in zip(bars, pr_aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11)
    
    # Event Recall comparison
    ax = axes[1]
    recalls = [baseline_results["alarm_metrics_optimal"]["event_recall"],
               tcn_results["alarm_metrics_optimal"]["event_recall"]]
    bars = ax.bar(models, recalls, color=['steelblue', 'darkorange'], edgecolor='black')
    ax.set_ylabel("Event Recall", fontsize=12)
    ax.set_title("Alarm-Level: Event Recall", fontsize=12)
    ax.set_ylim([0, 1])
    ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Target 80%')
    ax.legend()
    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=11)
    
    # False Alarm Rate comparison
    ax = axes[2]
    fars = [baseline_results["alarm_metrics_optimal"]["false_alarms"]["per_discharge"],
            tcn_results["alarm_metrics_optimal"]["false_alarms"]["per_discharge"]]
    bars = ax.bar(models, fars, color=['steelblue', 'darkorange'], edgecolor='black')
    ax.set_ylabel("False Alarms / Discharge", fontsize=12)
    ax.set_title("Alarm-Level: False Alarm Rate", fontsize=12)
    ax.axhline(y=5.0, color='r', linestyle='--', alpha=0.5, label='Target ≤5')
    ax.legend()
    for bar, val in zip(bars, fars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    
    print("=" * 70)
    print(" Tokamak Early Warning - Alarm Policy Evaluation")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seeds
    set_global_seed(args.seed)
    
    # --- Load Data ---
    print("\n[1/3] Loading dataset...")
    try:
        df = load_density_limit_data(args.data_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    print(f"  Loaded {len(df):,} samples from {df['discharge_ID'].nunique()} discharges")
    
    # --- Feature Engineering ---
    print("\n[2/3] Engineering features...")
    feature_config = FeatureConfig(
        rolling_windows=(3, 5, 10),
        include_greenwald_proxy=True,
        include_interactions=True
    )
    df_engineered = engineer_features(df, feature_config, verbose=False)
    feature_cols = get_engineered_feature_names(feature_config)
    print(f"  Generated {len(feature_cols)} features")
    
    # --- Split Data ---
    print("\n[3/3] Splitting data...")
    train_df, cal_df, val_df, test_df = split_by_discharge_with_cal(
        df_engineered, random_state=args.seed
    )
    
    # Also keep non-engineered splits for baseline
    train_df_base, cal_df_base, val_df_base, test_df_base = split_by_discharge_with_cal(
        df, random_state=args.seed
    )
    
    print(f"  Test set: {test_df['discharge_ID'].nunique()} discharges, {len(test_df):,} samples")
    
    # --- Evaluate Models ---
    results = {}
    
    if args.model in ["baseline", "both"]:
        results["baseline"] = evaluate_baseline_model(
            test_df_base, train_df_base, cal_df_base, args, output_dir
        )
    
    if args.model in ["tcn", "both"]:
        results["tcn"] = evaluate_tcn_model(
            df_engineered, train_df, cal_df, val_df, test_df,
            feature_cols, args, output_dir
        )
    
    # --- Create Comparison ---
    if args.model == "both" and args.save_figures:
        print("\n  Creating comparison plots...")
        plot_comparison(results["baseline"], results["tcn"], output_dir)
        
        # Combined PR curve
        baseline_metrics = results["baseline"]["sample_metrics"]
        tcn_metrics = results["tcn"]["sample_metrics"]
        
        print("\n" + "=" * 70)
        print(" Comparison Summary")
        print("=" * 70)
        print(f"\n  {'Metric':<30} {'Baseline LR':>15} {'TCN':>15} {'Improvement':>15}")
        print(f"  {'-'*75}")
        print(f"  {'PR-AUC':<30} {baseline_metrics['pr_auc']:>15.4f} {tcn_metrics['pr_auc']:>15.4f} {(tcn_metrics['pr_auc']-baseline_metrics['pr_auc'])/baseline_metrics['pr_auc']*100:>14.1f}%")
        print(f"  {'ROC-AUC':<30} {baseline_metrics['roc_auc']:>15.4f} {tcn_metrics['roc_auc']:>15.4f} {(tcn_metrics['roc_auc']-baseline_metrics['roc_auc'])/baseline_metrics['roc_auc']*100:>14.1f}%")
        print(f"  {'ECE (calibrated)':<30} {baseline_metrics['ece_calibrated']:>15.4f} {tcn_metrics['ece_calibrated']:>15.4f}")
        
        baseline_alarm = results["baseline"]["alarm_metrics_optimal"]
        tcn_alarm = results["tcn"]["alarm_metrics_optimal"]
        print(f"\n  {'Event Recall':<30} {baseline_alarm['event_recall']:>14.1%} {tcn_alarm['event_recall']:>14.1%}")
        print(f"  {'FA / Discharge':<30} {baseline_alarm['false_alarms']['per_discharge']:>15.2f} {tcn_alarm['false_alarms']['per_discharge']:>15.2f}")
        print(f"  {'Median Warning (s)':<30} {baseline_alarm['warning_times']['median']:>15.4f} {tcn_alarm['warning_times']['median']:>15.4f}")
    
    # --- Save Results ---
    print("\n" + "=" * 70)
    print(" Saving Results")
    print("=" * 70)
    
    # Remove non-serializable items
    for model_name in results:
        if "tradeoff_df" in results[model_name]:
            results[model_name]["tradeoff_df"].to_csv(
                output_dir / f"{model_name}_tradeoff.csv", index=False
            )
            del results[model_name]["tradeoff_df"]
    
    results_path = output_dir / "alarm_evaluation_results.json"
    payload = {
        "timestamp": utc_timestamp(),
        "seed": args.seed,
        "data_path": args.data_path,
        "results": results,
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  Saved results to {results_path}")
    
    print(f"\n  All outputs saved to: {output_dir}/")
    print("\n  Done!")


if __name__ == "__main__":
    main()
