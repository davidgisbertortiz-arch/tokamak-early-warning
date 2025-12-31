#!/usr/bin/env python3
"""
Generate publication-quality figures for the Tokamak Early Warning System.

This script creates all visualizations needed for the README and results.
Run after training to generate reproducible figures.

Usage:
    python scripts/make_figures.py              # Generate all figures
    python scripts/make_figures.py --quick      # Skip training, use existing models
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, auc, roc_curve

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import load_density_limit_data, split_by_discharge_with_cal
from src.features.temporal import engineer_features, get_engineered_feature_names, FeatureConfig
from src.models.baseline_lr import create_baseline_pipeline
from src.models.tcn import TCNClassifier, TCNConfig, create_sequences
from src.evaluation.alarm_metrics import AlarmConfig, compute_alarm_metrics, compute_tradeoff_curve

# ============================================================
# Configuration
# ============================================================

FEATURE_COLUMNS = ["density", "elongation", "minor_radius", 
                   "plasma_current", "toroidal_B_field", "triangularity"]
TARGET_COLUMN = "density_limit_phase"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'baseline': '#2196F3',  # Blue
    'tcn': '#4CAF50',       # Green
    'reference': '#9E9E9E', # Gray
    'event': '#F44336',     # Red
    'alarm': '#FF9800',     # Orange
}

# ============================================================
# Data Loading and Model Training
# ============================================================

def load_and_prepare_data():
    """Load dataset and engineer features."""
    print("Loading dataset...")
    df = load_density_limit_data("data/raw/DL_DataFrame.h5")
    
    print("Engineering features...")
    config = FeatureConfig(
        rolling_windows=(3, 5, 10),
        include_greenwald_proxy=True,
        include_interactions=True
    )
    df_eng = engineer_features(df, config)
    feature_cols = get_engineered_feature_names(config)
    
    print("Splitting by discharge_ID...")
    train_df, cal_df, val_df, test_df = split_by_discharge_with_cal(
        df_eng, train_frac=0.6, cal_frac=0.1, val_frac=0.15, random_state=42
    )
    
    return train_df, cal_df, val_df, test_df, feature_cols


def train_baseline(train_df, test_df, feature_cols):
    """Train baseline logistic regression."""
    print("\nTraining baseline model...")
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COLUMN].values
    X_test = test_df[feature_cols].values
    
    model = create_baseline_pipeline()
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X_test)[:, 1]
    return probs


def train_tcn(train_df, val_df, test_df, feature_cols, seq_len=20):
    """Train TCN model."""
    print("\nTraining TCN model...")
    
    X_train, y_train, _ = create_sequences(train_df, feature_cols, TARGET_COLUMN, seq_len)
    X_val, y_val, _ = create_sequences(val_df, feature_cols, TARGET_COLUMN, seq_len)
    X_test, y_test, seq_info = create_sequences(test_df, feature_cols, TARGET_COLUMN, seq_len)
    
    config = TCNConfig(
        n_features=len(feature_cols),
        seq_len=seq_len,
        hidden_channels=32,
        n_blocks=3,
        kernel_size=3,
        dropout=0.2,
        learning_rate=1e-3,
        batch_size=256,
        n_epochs=30
    )
    
    classifier = TCNClassifier(config, verbose=True)
    classifier.fit(X_train, y_train, X_val, y_val)
    
    probs = classifier.predict_proba(X_test)
    
    # Map back to original dataframe rows
    test_probs_full = np.zeros(len(test_df))
    test_probs_full[:] = np.nan
    for i, (_, end_idx) in enumerate(seq_info):
        test_probs_full[end_idx] = probs[i]
    
    return test_probs_full, classifier


# ============================================================
# Figure Generation Functions
# ============================================================

def fig1_pr_curves(test_df, baseline_probs, tcn_probs, save_path):
    """
    Figure 1: Precision-Recall curves comparing baseline vs TCN.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    y_test = test_df[TARGET_COLUMN].values
    
    # Baseline PR curve
    mask_bl = ~np.isnan(baseline_probs)
    prec_bl, rec_bl, _ = precision_recall_curve(y_test[mask_bl], baseline_probs[mask_bl])
    pr_auc_bl = auc(rec_bl, prec_bl)
    ax.plot(rec_bl, prec_bl, color=COLORS['baseline'], linewidth=2.5, 
            label=f'Baseline (LR): PR-AUC = {pr_auc_bl:.3f}')
    
    # TCN PR curve
    mask_tcn = ~np.isnan(tcn_probs)
    prec_tcn, rec_tcn, _ = precision_recall_curve(y_test[mask_tcn], tcn_probs[mask_tcn])
    pr_auc_tcn = auc(rec_tcn, prec_tcn)
    ax.plot(rec_tcn, prec_tcn, color=COLORS['tcn'], linewidth=2.5,
            label=f'TCN: PR-AUC = {pr_auc_tcn:.3f}')
    
    # Random baseline
    pos_rate = y_test.mean()
    ax.axhline(y=pos_rate, color=COLORS['reference'], linestyle='--', 
               label=f'Random ({pos_rate:.1%} positive rate)')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves: Baseline vs TCN', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")
    
    return pr_auc_bl, pr_auc_tcn


def fig2_reliability_diagram(test_df, probs, model_name, save_path):
    """
    Figure 2: Calibration reliability diagram.
    """
    y_test = test_df[TARGET_COLUMN].values
    mask = ~np.isnan(probs)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Reliability diagram
    prob_true, prob_pred = calibration_curve(y_test[mask], probs[mask], n_bins=10, strategy='uniform')
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.plot(prob_pred, prob_true, 's-', color=COLORS['tcn'], linewidth=2, 
             markersize=8, label=f'{model_name}')
    
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    
    # Right: Histogram of predictions
    ax2.hist(probs[mask & (y_test == 0)], bins=50, alpha=0.7, 
             color=COLORS['baseline'], label='Negative class', density=True)
    ax2.hist(probs[mask & (y_test == 1)], bins=50, alpha=0.7, 
             color=COLORS['event'], label='Positive class', density=True)
    
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_xlim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def fig3_far_vs_recall(test_df, probs, model_name, save_path):
    """
    Figure 3: False Alarm Rate vs Event Recall trade-off.
    """
    thresholds = np.linspace(0.05, 0.95, 19)
    
    recalls = []
    fars = []
    
    for thresh in thresholds:
        config = AlarmConfig(threshold=thresh, n_consecutive=1, use_ema=False)
        
        # Fill NaN with 0 for alarm computation
        probs_filled = np.nan_to_num(probs, nan=0.0)
        
        metrics = compute_alarm_metrics(test_df, probs_filled, config)
        recalls.append(metrics.event_recall)
        fars.append(metrics.false_alarms_per_discharge)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot trade-off curve
    scatter = ax.scatter(fars, recalls, c=thresholds, cmap='viridis', 
                         s=100, edgecolors='white', linewidths=1.5, zorder=3)
    ax.plot(fars, recalls, '-', color='gray', alpha=0.5, zorder=2)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Threshold', fontsize=11)
    
    # Reference lines
    ax.axhline(y=0.8, color=COLORS['event'], linestyle='--', alpha=0.7, 
               label='Target: 80% Event Recall')
    ax.axvline(x=5, color=COLORS['alarm'], linestyle='--', alpha=0.7,
               label='Target: ≤5 FA/discharge')
    
    # Highlight optimal region
    ax.fill_between([0, 5], [0.8, 0.8], [1, 1], alpha=0.1, color='green',
                    label='Optimal region')
    
    ax.set_xlabel('False Alarms per Discharge', fontsize=12)
    ax.set_ylabel('Event Recall', fontsize=12)
    ax.set_title(f'{model_name}: Event Recall vs False Alarm Trade-off', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0, max(fars) * 1.1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def fig4_warning_time_distribution(test_df, probs, model_name, save_path):
    """
    Figure 4: Distribution of warning times for detected events.
    """
    config = AlarmConfig(threshold=0.3, n_consecutive=2, use_ema=True, ema_alpha=0.3)
    
    # Fill NaN for computation
    probs_filled = np.nan_to_num(probs, nan=0.0)
    metrics = compute_alarm_metrics(test_df, probs_filled, config)
    
    warning_times = [wt for wt in metrics.warning_times if wt > 0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if len(warning_times) > 0:
        # Left: Histogram
        ax1.hist(warning_times, bins=20, color=COLORS['tcn'], alpha=0.7, 
                 edgecolor='white', linewidth=1.2)
        ax1.axvline(x=np.median(warning_times), color=COLORS['event'], 
                    linestyle='--', linewidth=2, label=f'Median: {np.median(warning_times):.3f}s')
        ax1.axvline(x=np.mean(warning_times), color=COLORS['alarm'],
                    linestyle=':', linewidth=2, label=f'Mean: {np.mean(warning_times):.3f}s')
        
        ax1.set_xlabel('Warning Time (seconds)', fontsize=12)
        ax1.set_ylabel('Number of Events', fontsize=12)
        ax1.set_title('Warning Time Distribution', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Right: Box plot with individual points
        bp = ax1.boxplot([warning_times], vert=False, widths=0.3,
                         patch_artist=True, positions=[0])
        for patch in bp['boxes']:
            patch.set_facecolor(COLORS['tcn'])
            patch.set_alpha(0.7)
    else:
        ax1.text(0.5, 0.5, 'No events detected\nwith current threshold', 
                 ha='center', va='center', fontsize=14, transform=ax1.transAxes)
        ax1.set_title('Warning Time Distribution', fontsize=14, fontweight='bold')
    
    # Right: Event recall vs threshold
    thresholds = np.linspace(0.1, 0.9, 17)
    recalls = []
    median_warnings = []
    
    for thresh in thresholds:
        cfg = AlarmConfig(threshold=thresh, n_consecutive=2, use_ema=True, ema_alpha=0.3)
        m = compute_alarm_metrics(test_df, probs_filled, cfg)
        recalls.append(m.event_recall)
        wts = [wt for wt in m.warning_times if wt > 0]
        median_warnings.append(np.median(wts) if wts else 0)
    
    ax2.plot(thresholds, recalls, 'o-', color=COLORS['tcn'], linewidth=2, 
             markersize=6, label='Event Recall')
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Event Recall', fontsize=12, color=COLORS['tcn'])
    ax2.tick_params(axis='y', labelcolor=COLORS['tcn'])
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(thresholds, median_warnings, 's--', color=COLORS['alarm'], 
                  linewidth=2, markersize=6, label='Median Warning Time')
    ax2_twin.set_ylabel('Median Warning Time (s)', fontsize=12, color=COLORS['alarm'])
    ax2_twin.tick_params(axis='y', labelcolor=COLORS['alarm'])
    
    ax2.set_title('Recall & Warning Time vs Threshold', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def fig5_model_comparison_summary(metrics_baseline, metrics_tcn, save_path):
    """
    Figure 5: Summary bar chart comparing models.
    """
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    
    metrics_names = ['PR-AUC', 'Event Recall', 'FA/Discharge', 'Median Warning (s)']
    baseline_vals = [
        metrics_baseline['pr_auc'],
        metrics_baseline['event_recall'],
        metrics_baseline['fa_per_discharge'],
        metrics_baseline['median_warning']
    ]
    tcn_vals = [
        metrics_tcn['pr_auc'],
        metrics_tcn['event_recall'],
        metrics_tcn['fa_per_discharge'],
        metrics_tcn['median_warning']
    ]
    
    x = np.array([0, 1])
    width = 0.35
    
    for i, (ax, name, bl_val, tcn_val) in enumerate(zip(axes, metrics_names, baseline_vals, tcn_vals)):
        bars = ax.bar(x, [bl_val, tcn_val], width, 
                      color=[COLORS['baseline'], COLORS['tcn']])
        
        ax.set_ylabel(name, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(['Baseline', 'TCN'], fontsize=10)
        ax.set_title(name, fontsize=12, fontweight='bold')
        
        # Add value labels on bars
        for bar, val in zip(bars, [bl_val, tcn_val]):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}' if val < 10 else f'{val:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Model Comparison Summary', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def fig6_single_discharge_demo(test_df, probs, model_name, save_path):
    """
    Figure 6: Single discharge visualization showing feature traces, 
    predictions, and alarm/event timing.
    """
    # Find a discharge with an event and predictions
    probs_filled = np.nan_to_num(probs, nan=0.0)
    
    # Find event discharges
    event_discharges = test_df[test_df[TARGET_COLUMN] == 1]['discharge_ID'].unique()
    
    if len(event_discharges) == 0:
        print("  No event discharges found for demo figure")
        return
    
    # Pick one with good prediction coverage
    selected_discharge = None
    for d_id in event_discharges:
        mask = test_df['discharge_ID'] == d_id
        if mask.sum() > 50 and probs_filled[mask].max() > 0.3:
            selected_discharge = d_id
            break
    
    if selected_discharge is None:
        selected_discharge = event_discharges[0]
    
    # Get discharge data
    discharge_mask = test_df['discharge_ID'] == selected_discharge
    discharge_df = test_df[discharge_mask].copy()
    discharge_probs = probs_filled[discharge_mask]
    
    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    time = discharge_df['time'].values
    
    # Panel 1: Density and Greenwald proxy
    ax1 = axes[0]
    ax1.plot(time, discharge_df['density'].values, color=COLORS['baseline'], 
             linewidth=1.5, label='Density')
    ax1.set_ylabel('Density', fontsize=11, color=COLORS['baseline'])
    ax1.tick_params(axis='y', labelcolor=COLORS['baseline'])
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_title(f'Discharge {selected_discharge}: Feature Traces & Predictions', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Plasma current
    ax2 = axes[1]
    ax2.plot(time, discharge_df['plasma_current'].values, color=COLORS['tcn'],
             linewidth=1.5, label='Plasma Current')
    ax2.set_ylabel('Plasma Current', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Predicted probability
    ax3 = axes[2]
    ax3.fill_between(time, 0, discharge_probs, color=COLORS['alarm'], alpha=0.3)
    ax3.plot(time, discharge_probs, color=COLORS['alarm'], linewidth=2, 
             label='P(density limit)')
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Threshold = 0.5')
    ax3.set_ylabel('Probability', fontsize=11)
    ax3.set_ylim([0, 1])
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Label (ground truth)
    ax4 = axes[3]
    labels = discharge_df[TARGET_COLUMN].values
    ax4.fill_between(time, 0, labels, color=COLORS['event'], alpha=0.5, 
                     step='mid', label='Density Limit Phase')
    ax4.set_ylabel('Label', fontsize=11)
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_ylim([-0.1, 1.1])
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Mark event onset
    event_times = time[labels == 1]
    if len(event_times) > 0:
        event_start = event_times[0]
        for ax in axes:
            ax.axvline(x=event_start, color=COLORS['event'], linestyle='-', 
                       linewidth=2, alpha=0.8)
        axes[0].annotate('Event onset', xy=(event_start, ax1.get_ylim()[1]),
                         xytext=(5, -15), textcoords='offset points',
                         fontsize=10, color=COLORS['event'])
    
    # Mark first alarm
    alarm_times = time[discharge_probs > 0.5]
    if len(alarm_times) > 0:
        first_alarm = alarm_times[0]
        for ax in axes:
            ax.axvline(x=first_alarm, color=COLORS['alarm'], linestyle='--',
                       linewidth=2, alpha=0.8)
        if len(event_times) > 0 and first_alarm < event_start:
            warning_time = event_start - first_alarm
            axes[2].annotate(f'First alarm\n(Warning: {warning_time:.3f}s)', 
                            xy=(first_alarm, 0.5),
                            xytext=(-60, 20), textcoords='offset points',
                            fontsize=9, color=COLORS['alarm'],
                            arrowprops=dict(arrowstyle='->', color=COLORS['alarm']))
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# Main Execution
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Generate publication figures')
    parser.add_argument('--quick', action='store_true',
                        help='Skip full TCN training, use minimal epochs')
    args = parser.parse_args()
    
    print("=" * 70)
    print(" Tokamak Early Warning - Figure Generation")
    print("=" * 70)
    
    # Ensure results directory exists
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Load data
    train_df, cal_df, val_df, test_df, feature_cols = load_and_prepare_data()
    
    # Train baseline
    baseline_probs = train_baseline(train_df, test_df, feature_cols)
    
    # Train TCN (or quick version)
    if args.quick:
        print("\n[Quick mode] Training TCN with minimal epochs...")
        # Temporarily modify for quick training
        tcn_probs, tcn_model = train_tcn(train_df, val_df, test_df, feature_cols, seq_len=20)
    else:
        tcn_probs, tcn_model = train_tcn(train_df, val_df, test_df, feature_cols, seq_len=20)
    
    print("\n" + "=" * 70)
    print(" Generating Figures")
    print("=" * 70)
    
    # Figure 1: PR Curves
    print("\n[1/6] PR Curves...")
    pr_auc_bl, pr_auc_tcn = fig1_pr_curves(
        test_df, baseline_probs, tcn_probs,
        RESULTS_DIR / "pr_curves.png"
    )
    
    # Figure 2: Reliability Diagram
    print("\n[2/6] Reliability Diagram...")
    fig2_reliability_diagram(
        test_df, tcn_probs, "TCN",
        RESULTS_DIR / "reliability_diagram.png"
    )
    
    # Figure 3: FAR vs Recall
    print("\n[3/6] FAR vs Recall Trade-off...")
    fig3_far_vs_recall(
        test_df, tcn_probs, "TCN",
        RESULTS_DIR / "far_vs_recall.png"
    )
    
    # Figure 4: Warning Time Distribution
    print("\n[4/6] Warning Time Distribution...")
    fig4_warning_time_distribution(
        test_df, tcn_probs, "TCN",
        RESULTS_DIR / "warning_time_distribution.png"
    )
    
    # Compute metrics for summary
    print("\n[5/6] Computing metrics for summary...")
    config = AlarmConfig(threshold=0.5, n_consecutive=2, use_ema=True, ema_alpha=0.3)
    
    baseline_probs_filled = np.nan_to_num(baseline_probs, nan=0.0)
    tcn_probs_filled = np.nan_to_num(tcn_probs, nan=0.0)
    
    bl_metrics = compute_alarm_metrics(test_df, baseline_probs_filled, config)
    tcn_metrics = compute_alarm_metrics(test_df, tcn_probs_filled, config)
    
    bl_warnings = [w for w in bl_metrics.warning_times if w > 0]
    tcn_warnings = [w for w in tcn_metrics.warning_times if w > 0]
    
    metrics_baseline = {
        'pr_auc': pr_auc_bl,
        'event_recall': bl_metrics.event_recall,
        'fa_per_discharge': bl_metrics.false_alarms_per_discharge,
        'median_warning': np.median(bl_warnings) if bl_warnings else 0.0
    }
    
    metrics_tcn = {
        'pr_auc': pr_auc_tcn,
        'event_recall': tcn_metrics.event_recall,
        'fa_per_discharge': tcn_metrics.false_alarms_per_discharge,
        'median_warning': np.median(tcn_warnings) if tcn_warnings else 0.0
    }
    
    # Figure 5: Model Comparison
    fig5_model_comparison_summary(
        metrics_baseline, metrics_tcn,
        RESULTS_DIR / "model_comparison.png"
    )
    
    # Figure 6: Single Discharge Demo
    print("\n[6/6] Single Discharge Demo...")
    fig6_single_discharge_demo(
        test_df, tcn_probs, "TCN",
        RESULTS_DIR / "discharge_demo.png"
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print(" Summary")
    print("=" * 70)
    print(f"\n  Baseline LR:")
    print(f"    PR-AUC:          {metrics_baseline['pr_auc']:.4f}")
    print(f"    Event Recall:    {metrics_baseline['event_recall']:.1%}")
    print(f"    FA/Discharge:    {metrics_baseline['fa_per_discharge']:.2f}")
    print(f"    Median Warning:  {metrics_baseline['median_warning']:.4f}s")
    
    print(f"\n  TCN:")
    print(f"    PR-AUC:          {metrics_tcn['pr_auc']:.4f}")
    print(f"    Event Recall:    {metrics_tcn['event_recall']:.1%}")
    print(f"    FA/Discharge:    {metrics_tcn['fa_per_discharge']:.2f}")
    print(f"    Median Warning:  {metrics_tcn['median_warning']:.4f}s")
    
    print(f"\n  Figures saved to: {RESULTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
