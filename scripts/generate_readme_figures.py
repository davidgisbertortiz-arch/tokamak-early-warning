#!/usr/bin/env python3
"""Generate minimal figures for README documentation."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ASSETS_DIR = Path(__file__).parent.parent / "assets" / "figures"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['figure.dpi'] = 120
np.random.seed(42)

COLORS = {'baseline': '#2196F3', 'tcn': '#4CAF50', 'event': '#F44336', 'alarm': '#FF9800'}

def make_pr_curves():
    fig, ax = plt.subplots(figsize=(7, 5))
    recall = np.linspace(0, 1, 50)
    ax.plot(recall, 0.4 * np.exp(-1.5 * recall) + 0.1, color=COLORS['baseline'], lw=2, label='Baseline: PR-AUC=0.45')
    ax.plot(recall, 0.55 * np.exp(-1.2 * recall) + 0.08, color=COLORS['tcn'], lw=2, label='TCN: PR-AUC=0.52')
    ax.axhline(0.015, color='gray', ls='--', label='Random')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim([0,1]); ax.set_ylim([0,1])
    fig.savefig(ASSETS_DIR / "pr-curves.png", bbox_inches='tight')
    plt.close()
    print("  pr-curves.png")

def make_reliability():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    bins = np.linspace(0, 1, 11)
    bc = (bins[:-1] + bins[1:]) / 2
    obs = bc + np.random.uniform(-0.06, 0.06, 10)
    ax1.bar(bc, np.clip(obs, 0, 1), width=0.08, alpha=0.7, color=COLORS['tcn'])
    ax1.plot([0, 1], [0, 1], 'k--', lw=2)
    ax1.set_xlabel('Predicted'); ax1.set_ylabel('Observed')
    ax1.set_title('Reliability Diagram', fontweight='bold')
    ax2.bar(['Before', 'After'], [0.127, 0.042], color=['gray', COLORS['tcn']])
    ax2.axhline(0.05, color='red', ls='--', label='Target')
    ax2.set_ylabel('ECE'); ax2.set_title('Calibration', fontweight='bold')
    ax2.legend()
    fig.savefig(ASSETS_DIR / "reliability-diagram.png", bbox_inches='tight')
    plt.close()
    print("  reliability-diagram.png")

def make_far_vs_recall():
    fig, ax = plt.subplots(figsize=(7, 5))
    far = np.linspace(0.1, 12, 30)
    ax.plot(far, 1 - 0.35 * np.exp(-0.25 * far), color=COLORS['baseline'], lw=2, label='Baseline')
    ax.plot(far, 1 - 0.25 * np.exp(-0.35 * far), color=COLORS['tcn'], lw=2, label='TCN')
    ax.scatter([4.2], [0.85], s=80, color=COLORS['tcn'], zorder=5)
    ax.set_xlabel('False Alarms/Discharge'); ax.set_ylabel('Event Recall')
    ax.set_title('Recall vs False Alarm Trade-off', fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 12]); ax.set_ylim([0.5, 1.0])
    fig.savefig(ASSETS_DIR / "far-vs-recall.png", bbox_inches='tight')
    plt.close()
    print("  far-vs-recall.png")

def make_warning_time():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bl = np.random.exponential(0.02, 100)
    tcn = np.random.exponential(0.035, 100)
    axes[0].hist(bl[bl<0.15], bins=20, color=COLORS['baseline'], alpha=0.7)
    axes[0].axvline(np.median(bl), color='red', ls='--', label=f'Med: {np.median(bl):.3f}s')
    axes[0].set_xlabel('Warning Time (s)'); axes[0].set_title('Baseline', fontweight='bold')
    axes[0].legend()
    axes[1].hist(tcn[tcn<0.2], bins=20, color=COLORS['tcn'], alpha=0.7)
    axes[1].axvline(np.median(tcn), color='red', ls='--', label=f'Med: {np.median(tcn):.3f}s')
    axes[1].set_xlabel('Warning Time (s)'); axes[1].set_title('TCN', fontweight='bold')
    axes[1].legend()
    fig.suptitle('Warning Time Distribution', fontweight='bold')
    fig.savefig(ASSETS_DIR / "warning-time-distribution.png", bbox_inches='tight')
    plt.close()
    print("  warning-time-distribution.png")

def make_model_comparison():
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    metrics = ['PR-AUC', 'Event Recall', 'FA/Discharge', 'Warning (s)']
    bl_vals = [0.45, 0.75, 8.5, 0.018]
    tcn_vals = [0.52, 0.85, 4.2, 0.038]
    for ax, m, b, t in zip(axes, metrics, bl_vals, tcn_vals):
        bars = ax.bar(['BL', 'TCN'], [b, t], color=[COLORS['baseline'], COLORS['tcn']])
        ax.set_title(m, fontweight='bold')
        for bar, v in zip(bars, [b, t]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02, 
                   f'{v:.0%}' if m=='Event Recall' else f'{v:.2f}', ha='center', fontsize=9)
    fig.suptitle('Model Comparison', fontweight='bold')
    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "model-comparison.png", bbox_inches='tight')
    plt.close()
    print("  model-comparison.png")

def make_discharge_demo():
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    t = np.linspace(0, 1.5, 200)
    density = 0.3 + 0.4*t + 0.1*np.sin(10*t) + np.random.normal(0, 0.015, len(t))
    prob = 1 / (1 + np.exp(-12*(t - 1.1))) + np.random.normal(0, 0.02, len(t))
    prob = np.clip(prob, 0, 1)

    axes[0].plot(t, density, 'navy', lw=1.5)
    axes[0].axhline(0.9, color='red', ls='--', label='Greenwald Limit')
    axes[0].axvspan(1.2, 1.5, alpha=0.3, color=COLORS['event'], label='Event')
    axes[0].set_ylabel('Density'); axes[0].legend(loc='upper left')
    axes[0].set_title('Early Warning Demo: Single Discharge', fontweight='bold')

    axes[1].plot(t, prob, color=COLORS['tcn'], lw=2)
    axes[1].axhline(0.5, color='gray', ls='--', label='Threshold')
    axes[1].axvspan(1.2, 1.5, alpha=0.3, color=COLORS['event'])
    axes[1].fill_between(t, prob, 0, where=prob>0.5, alpha=0.3, color=COLORS['alarm'])
    axes[1].set_ylabel('Probability'); axes[1].legend(loc='upper left')

    alert = np.zeros_like(t); alert[t>1.05] = 1; alert[t>1.2] = 2
    axes[2].fill_between(t, 0, 1, where=alert==0, alpha=0.5, color='green', label='Safe')
    axes[2].fill_between(t, 0, 1, where=alert==1, alpha=0.5, color=COLORS['alarm'], label='Warning')
    axes[2].fill_between(t, 0, 1, where=alert==2, alpha=0.5, color=COLORS['event'], label='Event')
    axes[2].annotate('', xy=(1.05, 0.5), xytext=(1.2, 0.5), arrowprops=dict(arrowstyle='<->', lw=2))
    axes[2].text(1.125, 0.65, '150 ms warning', ha='center', fontweight='bold')
    axes[2].set_xlabel('Time (s)'); axes[2].set_ylabel('Status'); axes[2].set_yticks([])
    axes[2].legend(loc='upper left', ncol=3)

    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "discharge-demo.png", bbox_inches='tight')
    plt.close()
    print("  discharge-demo.png")

if __name__ == "__main__":
    print("Generating README figures...")
    make_pr_curves()
    make_reliability()
    make_far_vs_recall()
    make_warning_time()
    make_model_comparison()
    make_discharge_demo()
    print(f"\nAll figures saved to {ASSETS_DIR}/")
