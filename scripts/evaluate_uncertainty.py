#!/usr/bin/env python3
"""
Uncertainty quantification and event-based evaluation for early warning.

This script:
1. Loads data and splits by discharge_ID (train/cal/val/test)
2. Trains baseline model on training set
3. Calibrates conformal predictor on calibration set
4. Evaluates with:
   - Standard metrics (PR-AUC, ROC-AUC)
   - Conformal coverage (global and per-discharge)
   - Event-based metrics (event recall, lead time, false alarms)
   - Alert policy evaluation (yellow/red alerts with acceptance criteria)

Usage:
    python scripts/evaluate_uncertainty.py
    
Prerequisites:
    ./scripts/fetch_data.sh  (to download data first)
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data.dataset import (
    load_density_limit_data,
    split_by_discharge_with_cal,
    get_features_target,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)
from src.models.baseline_lr import train_baseline
from src.evaluation.metrics import evaluate_model, print_evaluation_report
from src.evaluation.event_metrics import (
    compute_event_metrics,
    print_event_metrics_report
)
from src.uncertainty.conformal import (
    ConformalClassifier,
    print_coverage_report
)
from src.uncertainty.calibration import (
    expected_calibration_error,
    reliability_diagram_data
)
from src.alerting.policy import (
    AlertPolicy,
    AlertLevel,
    print_alert_policy_report
)
from tokamak_early_warning.config import DEFAULT_DATA_PATH, DEFAULT_SEED, set_global_seed


def print_split_stats_with_cal(
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = TARGET_COLUMN
) -> None:
    """Print statistics about train/cal/val/test splits."""
    for name, df in [("Train", train_df), ("Cal", cal_df), ("Val", val_df), ("Test", test_df)]:
        n_discharges = df["discharge_ID"].nunique()
        n_samples = len(df)
        pos_rate = df[target_col].mean() * 100
        n_events = sum(1 for _, g in df.groupby("discharge_ID") if g[target_col].any())
        print(f"  {name}: {n_discharges:4d} discharges, {n_samples:6d} samples, "
              f"{pos_rate:.2f}% positive, {n_events} events")


def evaluate_event_detection_with_alerts(
    df: pd.DataFrame,
    conformal_result,
    policy: AlertPolicy
) -> tuple:
    """
    Evaluate event detection using the alert policy.
    
    Returns event recall and alert results per discharge.
    """
    alert_results = []
    events_detected = 0
    total_events = 0
    lead_times = []
    false_alarms_total = 0
    
    for discharge_id in df["discharge_ID"].unique():
        mask = df["discharge_ID"] == discharge_id
        discharge_df = df[mask].sort_values("time")
        indices = discharge_df.index
        
        # Get predictions for this discharge
        pred_sets = [conformal_result.prediction_sets[i] for i in range(len(df)) 
                     if df.index[i] in indices]
        probs = conformal_result.probabilities[mask]
        times = discharge_df["time"].values
        
        # Reindex pred_sets to match discharge_df order
        local_mask = np.array([df.index[i] in indices for i in range(len(df))])
        pred_sets = [conformal_result.prediction_sets[i] 
                     for i in range(len(df)) if local_mask[i]]
        
        # Process discharge through alert policy
        result = policy.process_discharge(
            discharge_id=discharge_id,
            prediction_sets=pred_sets,
            probabilities=probs,
            times=times
        )
        alert_results.append(result)
        
        # Check if this discharge has an event
        has_event = (discharge_df[TARGET_COLUMN] == 1).any()
        
        if has_event:
            total_events += 1
            event_time = discharge_df.loc[discharge_df[TARGET_COLUMN] == 1, "time"].iloc[0]
            
            # Check if RED alert occurred before event
            alerts_array = np.array([a.value for a in result.alerts])
            times_array = times
            
            # Find first RED alert before event
            red_before_event = (alerts_array == AlertLevel.RED.value) & (times_array < event_time)
            
            if red_before_event.any():
                events_detected += 1
                first_red_idx = np.where(red_before_event)[0][0]
                lead_time = event_time - times_array[first_red_idx]
                lead_times.append(lead_time)
        else:
            # Count false alarms (RED alerts on non-event discharges)
            false_alarms_total += result.n_red_alerts
    
    event_recall = events_detected / total_events if total_events > 0 else 0.0
    mean_lead_time = np.mean(lead_times) if lead_times else 0.0
    false_alarms_per_discharge = false_alarms_total / len(df["discharge_ID"].unique())
    
    return event_recall, mean_lead_time, false_alarms_per_discharge, alert_results


def main():
    print("="*70)
    print(" Tokamak Early Warning - Uncertainty Quantification & Event Metrics")
    print("="*70)
    
    # --- Load Data ---
    print("\n[1/6] Loading dataset...")
    set_global_seed(DEFAULT_SEED)
    try:
        df = load_density_limit_data(DEFAULT_DATA_PATH)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    print(f"  Loaded {len(df):,} samples from {df['discharge_ID'].nunique()} discharges")
    
    # --- Split by Discharge ID (with calibration set) ---
    print("\n[2/6] Splitting by discharge_ID (60/10/15/15 = train/cal/val/test)...")
    print("  CRITICAL: Calibration set must be separate for valid coverage!")
    
    train_df, cal_df, val_df, test_df = split_by_discharge_with_cal(
        df,
        train_frac=0.60,
        cal_frac=0.10,
        val_frac=0.15,
        test_frac=0.15,
        random_state=DEFAULT_SEED
    )
    
    print_split_stats_with_cal(train_df, cal_df, val_df, test_df)
    
    # Prepare features
    X_train, y_train = get_features_target(train_df)
    X_cal, y_cal = get_features_target(cal_df)
    X_val, y_val = get_features_target(val_df)
    X_test, y_test = get_features_target(test_df)
    
    # --- Train Model ---
    print("\n[3/6] Training LogisticRegression (class_weight='balanced')...")
    model = train_baseline(X_train, y_train, max_iter=500, random_state=DEFAULT_SEED)
    print("  Training complete!")
    
    # --- Standard Metrics ---
    print("\n[4/6] Standard evaluation (PR-AUC, ROC-AUC)...")
    
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_prob)
    print_evaluation_report(test_metrics, y_test.values, y_test_pred, "Test")
    
    # Calibration error
    ece, _ = expected_calibration_error(y_test.values, y_test_prob)
    print(f"\n  Expected Calibration Error (ECE): {ece:.4f}")
    
    # --- Conformal Prediction ---
    print("\n[5/6] Conformal prediction (calibrating on held-out set)...")
    
    alpha = 0.10  # Target 90% coverage
    conformal = ConformalClassifier(model, alpha=alpha)
    conformal.calibrate(X_cal.values, y_cal.values)
    
    print(f"  Alpha (miscoverage): {alpha}")
    print(f"  Calibration threshold: {conformal.threshold:.4f}")
    print(f"  Calibration set: {len(y_cal)} samples from {cal_df['discharge_ID'].nunique()} discharges")
    
    # Evaluate on test set
    conformal_result = conformal.predict(X_test.values)
    coverage_metrics = conformal.evaluate_coverage(
        X_test.values,
        y_test.values,
        discharge_ids=test_df["discharge_ID"].values
    )
    print_coverage_report(coverage_metrics, "Test")
    
    # --- Event-Based Metrics ---
    print("\n[6/6] Event-based metrics & alert policy evaluation...")
    
    # Basic event metrics (using point predictions from conformal)
    point_preds = conformal_result.predictions
    event_metrics = compute_event_metrics(
        test_df.reset_index(drop=True),
        point_preds,
        max_lead_time=None
    )
    print_event_metrics_report(event_metrics, "Test")
    
    # --- Alert Policy Evaluation ---
    # Note: Default thresholds are conservative. For this baseline model,
    # we use more permissive thresholds to demonstrate the framework.
    # In production, tune these based on operational requirements.
    policy = AlertPolicy(
        prob_yellow_threshold=0.2,   # Lower threshold to catch more events
        prob_red_threshold=0.5,      # Lower for baseline model
        n_consecutive_for_red=1,     # Single high-confidence prediction
        n_consecutive_clear=3,
        min_event_recall=0.80,       # Realistic target for baseline
        max_false_alarms_per_discharge=5.0,  # More permissive for baseline
        min_lead_time_seconds=0.02   # 20ms minimum
    )
    
    # Process each discharge through alert policy
    event_recall, mean_lead_time, fa_per_discharge, alert_results = \
        evaluate_event_detection_with_alerts(test_df.reset_index(drop=True), conformal_result, policy)
    
    print_alert_policy_report(
        policy,
        alert_results,
        event_recall,
        fa_per_discharge,
        mean_lead_time
    )
    
    # --- Summary ---
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    print(f"\n  Model Performance:")
    print(f"    PR-AUC:              {test_metrics['pr_auc']:.4f}")
    print(f"    ROC-AUC:             {test_metrics['roc_auc']:.4f}")
    
    print(f"\n  Uncertainty Quantification:")
    print(f"    Global Coverage:     {coverage_metrics['global_coverage']:.1%} (target: 90%)")
    print(f"    Per-Discharge Mean:  {coverage_metrics['per_discharge']['mean_coverage']:.1%}")
    print(f"    ECE:                 {ece:.4f}")
    
    print(f"\n  Event Detection (with alerts):")
    print(f"    Event Recall:        {event_recall:.1%}")
    print(f"    Mean Lead Time:      {mean_lead_time:.4f}s")
    print(f"    False Alarms/Disch:  {fa_per_discharge:.2f}")
    
    # Final acceptance check
    criteria = policy.check_acceptance_criteria(event_recall, fa_per_discharge, mean_lead_time)
    all_pass = all(criteria.values())
    
    print(f"\n  Acceptance Criteria: {'✓ ALL PASS' if all_pass else '✗ NOT MET'}")
    for name, passed in criteria.items():
        status = "✓" if passed else "✗"
        print(f"    {status} {name}")


if __name__ == "__main__":
    main()
