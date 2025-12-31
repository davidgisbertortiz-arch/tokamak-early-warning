"""
Event-level / Alarm-level evaluation for early warning systems.

This module provides comprehensive alarm evaluation with:
- Configurable alarm trigger policies (threshold, consecutive steps, EMA smoothing)
- Event-level metrics (event recall, warning time, false alarm rate)
- Trade-off curve computation (threshold vs metrics)
- Per-discharge analysis

Unlike sample-level metrics (PR-AUC), these metrics capture operational utility:
- Did we detect the event BEFORE it happened?
- How early did we detect it (warning time)?
- How many false alarms did we generate?
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import json


@dataclass
class AlarmConfig:
    """Configuration for alarm trigger policy."""
    # Threshold-based triggering
    threshold: float = 0.5  # Probability threshold for alarm
    
    # Consecutive steps requirement (reduces noise)
    n_consecutive: int = 1  # Number of consecutive threshold crossings to trigger
    
    # Exponential Moving Average smoothing
    use_ema: bool = False
    ema_alpha: float = 0.3  # EMA smoothing factor (higher = more weight on recent)
    
    # Alarm cooldown (prevent repeated alarms for same event)
    cooldown_steps: int = 0  # Steps to wait after alarm before new alarm


@dataclass 
class DischargeAlarmResult:
    """Alarm evaluation result for a single discharge."""
    discharge_id: int
    has_event: bool  # Whether discharge contains a density limit event
    
    # Event detection (only relevant if has_event)
    event_detected: bool = False  # Alarm triggered before event?
    event_time: Optional[float] = None  # Time of event start
    first_alarm_time: Optional[float] = None  # Time of first alarm
    warning_time: Optional[float] = None  # event_time - first_alarm_time
    n_alarms_before_event: int = 0  # Alarms in the warning window
    
    # False alarms
    n_false_alarms: int = 0  # Alarms on non-event discharges, or after event
    
    # Time series
    alarm_times: List[float] = field(default_factory=list)
    probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    smoothed_probs: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class AlarmMetrics:
    """Aggregate alarm metrics across all discharges."""
    # Event detection
    n_event_discharges: int  # Discharges containing events
    n_events_detected: int  # Events detected before they occurred
    event_recall: float  # n_events_detected / n_event_discharges
    miss_rate: float  # 1 - event_recall
    
    # Warning time statistics (only for detected events)
    warning_times: List[float]
    mean_warning_time: float
    median_warning_time: float
    std_warning_time: float
    p10_warning_time: float  # 10th percentile
    p25_warning_time: float  # 25th percentile
    p75_warning_time: float  # 75th percentile
    p90_warning_time: float  # 90th percentile
    min_warning_time: float
    max_warning_time: float
    
    # False alarm metrics
    n_false_alarms: int  # Total false alarms
    n_non_event_discharges: int  # Discharges without events
    n_discharges_with_false_alarms: int
    false_alarms_per_discharge: float  # Average across all discharges
    false_alarms_per_non_event_discharge: float  # Average on non-event discharges
    false_alarm_rate_per_hour: float  # Normalized by total observation time
    
    # Config used
    config: AlarmConfig
    
    # Per-discharge details
    discharge_results: List[DischargeAlarmResult] = field(default_factory=list)


def apply_ema(values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apply Exponential Moving Average smoothing.
    
    EMA(t) = alpha * value(t) + (1 - alpha) * EMA(t-1)
    
    Higher alpha = more weight on recent values (less smoothing).
    Lower alpha = more smoothing.
    """
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]
    
    return smoothed


def detect_alarm_triggers(
    probs: np.ndarray,
    config: AlarmConfig
) -> np.ndarray:
    """
    Detect alarm triggers based on config.
    
    Args:
        probs: Probability sequence
        config: Alarm configuration
        
    Returns:
        Boolean array indicating alarm triggers
    """
    # Apply EMA smoothing if configured
    if config.use_ema:
        probs = apply_ema(probs, config.ema_alpha)
    
    # Threshold crossing
    above_threshold = probs >= config.threshold
    
    # Consecutive steps requirement
    if config.n_consecutive > 1:
        triggers = np.zeros_like(above_threshold)
        consecutive_count = 0
        
        for i in range(len(above_threshold)):
            if above_threshold[i]:
                consecutive_count += 1
                if consecutive_count >= config.n_consecutive:
                    triggers[i] = True
            else:
                consecutive_count = 0
    else:
        triggers = above_threshold
    
    # Apply cooldown
    if config.cooldown_steps > 0:
        filtered = np.zeros_like(triggers)
        last_alarm_idx = -config.cooldown_steps - 1
        
        for i in range(len(triggers)):
            if triggers[i] and (i - last_alarm_idx) > config.cooldown_steps:
                filtered[i] = True
                last_alarm_idx = i
        
        triggers = filtered
    
    return triggers


def evaluate_discharge_alarms(
    discharge_df: pd.DataFrame,
    probs: np.ndarray,
    config: AlarmConfig,
    time_col: str = "time",
    target_col: str = "density_limit_phase"
) -> DischargeAlarmResult:
    """
    Evaluate alarms for a single discharge.
    
    Args:
        discharge_df: DataFrame for one discharge, sorted by time
        probs: Probability predictions aligned with discharge_df
        config: Alarm configuration
        time_col: Time column name
        target_col: Target column name
        
    Returns:
        DischargeAlarmResult with all metrics
    """
    discharge_id = discharge_df["discharge_ID"].iloc[0]
    times = discharge_df[time_col].values
    targets = discharge_df[target_col].values
    
    # Check if discharge has an event
    has_event = targets.max() > 0
    
    # Apply EMA and get smoothed probabilities
    if config.use_ema:
        smoothed = apply_ema(probs, config.ema_alpha)
    else:
        smoothed = probs
    
    # Detect alarm triggers
    triggers = detect_alarm_triggers(probs, config)
    alarm_times = times[triggers].tolist()
    
    result = DischargeAlarmResult(
        discharge_id=discharge_id,
        has_event=has_event,
        probabilities=probs,
        smoothed_probs=smoothed,
        alarm_times=alarm_times
    )
    
    if has_event:
        # Find event start time (first positive label)
        event_idx = np.argmax(targets > 0)
        event_time = times[event_idx]
        result.event_time = event_time
        
        # Check for alarms before event
        alarms_before = [t for t in alarm_times if t < event_time]
        result.n_alarms_before_event = len(alarms_before)
        result.event_detected = len(alarms_before) > 0
        
        if result.event_detected:
            result.first_alarm_time = alarms_before[0]
            result.warning_time = event_time - alarms_before[0]
        
        # False alarms after event (if any)
        result.n_false_alarms = len([t for t in alarm_times if t >= event_time])
    else:
        # All alarms on non-event discharge are false alarms
        result.n_false_alarms = len(alarm_times)
    
    return result


def compute_alarm_metrics(
    df: pd.DataFrame,
    probs: np.ndarray,
    config: AlarmConfig,
    discharge_col: str = "discharge_ID",
    time_col: str = "time",
    target_col: str = "density_limit_phase"
) -> AlarmMetrics:
    """
    Compute comprehensive alarm metrics across all discharges.
    
    Args:
        df: Full DataFrame with all discharges
        probs: Probability predictions aligned with df
        config: Alarm configuration
        discharge_col: Discharge ID column name
        time_col: Time column name
        target_col: Target column name
        
    Returns:
        AlarmMetrics with all computed metrics
    """
    df = df.copy()
    df["_prob"] = probs
    
    discharge_results = []
    total_time_hours = 0.0
    
    for discharge_id in df[discharge_col].unique():
        mask = df[discharge_col] == discharge_id
        discharge_df = df[mask].sort_values(time_col)
        discharge_probs = discharge_df["_prob"].values
        
        # Track total observation time
        if len(discharge_df) > 1:
            discharge_time = discharge_df[time_col].max() - discharge_df[time_col].min()
            total_time_hours += discharge_time / 3600
        
        result = evaluate_discharge_alarms(
            discharge_df, discharge_probs, config, time_col, target_col
        )
        discharge_results.append(result)
    
    # Aggregate metrics
    event_discharges = [r for r in discharge_results if r.has_event]
    non_event_discharges = [r for r in discharge_results if not r.has_event]
    
    n_event_discharges = len(event_discharges)
    n_events_detected = sum(1 for r in event_discharges if r.event_detected)
    
    # Warning times for detected events
    warning_times = [r.warning_time for r in event_discharges 
                     if r.warning_time is not None]
    
    # False alarms
    n_false_alarms = sum(r.n_false_alarms for r in discharge_results)
    n_discharges_with_fa = sum(1 for r in discharge_results if r.n_false_alarms > 0)
    
    # Compute percentiles
    if warning_times:
        p10 = np.percentile(warning_times, 10)
        p25 = np.percentile(warning_times, 25)
        p75 = np.percentile(warning_times, 75)
        p90 = np.percentile(warning_times, 90)
    else:
        p10 = p25 = p75 = p90 = 0.0
    
    return AlarmMetrics(
        n_event_discharges=n_event_discharges,
        n_events_detected=n_events_detected,
        event_recall=n_events_detected / n_event_discharges if n_event_discharges > 0 else 0.0,
        miss_rate=1 - (n_events_detected / n_event_discharges) if n_event_discharges > 0 else 1.0,
        
        warning_times=warning_times,
        mean_warning_time=np.mean(warning_times) if warning_times else 0.0,
        median_warning_time=np.median(warning_times) if warning_times else 0.0,
        std_warning_time=np.std(warning_times) if warning_times else 0.0,
        p10_warning_time=p10,
        p25_warning_time=p25,
        p75_warning_time=p75,
        p90_warning_time=p90,
        min_warning_time=min(warning_times) if warning_times else 0.0,
        max_warning_time=max(warning_times) if warning_times else 0.0,
        
        n_false_alarms=n_false_alarms,
        n_non_event_discharges=len(non_event_discharges),
        n_discharges_with_false_alarms=n_discharges_with_fa,
        false_alarms_per_discharge=n_false_alarms / len(discharge_results) if discharge_results else 0.0,
        false_alarms_per_non_event_discharge=(
            sum(r.n_false_alarms for r in non_event_discharges) / len(non_event_discharges)
            if non_event_discharges else 0.0
        ),
        false_alarm_rate_per_hour=n_false_alarms / total_time_hours if total_time_hours > 0 else 0.0,
        
        config=config,
        discharge_results=discharge_results
    )


def compute_tradeoff_curve(
    df: pd.DataFrame,
    probs: np.ndarray,
    thresholds: np.ndarray = None,
    n_consecutive_values: List[int] = None,
    use_ema: bool = False,
    ema_alpha: float = 0.3,
    discharge_col: str = "discharge_ID",
    time_col: str = "time",
    target_col: str = "density_limit_phase"
) -> pd.DataFrame:
    """
    Compute trade-off curves for different alarm configurations.
    
    Sweeps over thresholds and consecutive step requirements to show
    the trade-off between event recall, false alarm rate, and warning time.
    
    Args:
        df: Full DataFrame
        probs: Probability predictions
        thresholds: Thresholds to evaluate (default: linspace 0.1 to 0.9)
        n_consecutive_values: Consecutive step values to try
        use_ema: Whether to use EMA smoothing
        ema_alpha: EMA smoothing factor
        
    Returns:
        DataFrame with metrics for each configuration
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)
    
    if n_consecutive_values is None:
        n_consecutive_values = [1, 2, 3]
    
    results = []
    
    for threshold in thresholds:
        for n_consec in n_consecutive_values:
            config = AlarmConfig(
                threshold=threshold,
                n_consecutive=n_consec,
                use_ema=use_ema,
                ema_alpha=ema_alpha
            )
            
            metrics = compute_alarm_metrics(
                df, probs, config, discharge_col, time_col, target_col
            )
            
            results.append({
                "threshold": threshold,
                "n_consecutive": n_consec,
                "use_ema": use_ema,
                "ema_alpha": ema_alpha if use_ema else None,
                "event_recall": metrics.event_recall,
                "miss_rate": metrics.miss_rate,
                "n_events_detected": metrics.n_events_detected,
                "n_event_discharges": metrics.n_event_discharges,
                "false_alarms_per_discharge": metrics.false_alarms_per_discharge,
                "false_alarm_rate_per_hour": metrics.false_alarm_rate_per_hour,
                "n_false_alarms": metrics.n_false_alarms,
                "mean_warning_time": metrics.mean_warning_time,
                "median_warning_time": metrics.median_warning_time,
                "p25_warning_time": metrics.p25_warning_time,
                "p75_warning_time": metrics.p75_warning_time,
            })
    
    return pd.DataFrame(results)


def find_optimal_config(
    tradeoff_df: pd.DataFrame,
    min_event_recall: float = 0.8,
    max_false_alarms_per_discharge: float = 5.0,
    prefer: str = "warning_time"  # "warning_time" or "false_alarm"
) -> Dict[str, Any]:
    """
    Find optimal alarm configuration meeting constraints.
    
    Args:
        tradeoff_df: DataFrame from compute_tradeoff_curve
        min_event_recall: Minimum acceptable event recall
        max_false_alarms_per_discharge: Maximum acceptable FAR
        prefer: What to optimize - "warning_time" (maximize) or "false_alarm" (minimize)
        
    Returns:
        Dictionary with optimal config and metrics
    """
    # Filter by constraints
    valid = tradeoff_df[
        (tradeoff_df["event_recall"] >= min_event_recall) &
        (tradeoff_df["false_alarms_per_discharge"] <= max_false_alarms_per_discharge)
    ]
    
    if len(valid) == 0:
        # Relax constraints
        valid = tradeoff_df[tradeoff_df["event_recall"] >= min_event_recall]
        if len(valid) == 0:
            valid = tradeoff_df
    
    # Find optimal
    if prefer == "warning_time":
        best_idx = valid["median_warning_time"].idxmax()
    else:
        best_idx = valid["false_alarms_per_discharge"].idxmin()
    
    best_row = valid.loc[best_idx]
    
    return {
        "threshold": best_row["threshold"],
        "n_consecutive": int(best_row["n_consecutive"]),
        "use_ema": best_row["use_ema"],
        "ema_alpha": best_row.get("ema_alpha"),
        "metrics": {
            "event_recall": best_row["event_recall"],
            "false_alarms_per_discharge": best_row["false_alarms_per_discharge"],
            "median_warning_time": best_row["median_warning_time"],
        }
    }


def print_alarm_metrics_report(metrics: AlarmMetrics, title: str = "Alarm Evaluation") -> None:
    """Print formatted alarm metrics report."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")
    
    print(f"\n  Alarm Policy Configuration:")
    print(f"    Threshold:           {metrics.config.threshold:.2f}")
    print(f"    Consecutive steps:   {metrics.config.n_consecutive}")
    print(f"    EMA smoothing:       {metrics.config.use_ema}")
    if metrics.config.use_ema:
        print(f"    EMA alpha:           {metrics.config.ema_alpha:.2f}")
    
    print(f"\n  Event Detection:")
    print(f"    Event discharges:    {metrics.n_event_discharges}")
    print(f"    Events detected:     {metrics.n_events_detected}")
    print(f"    Event recall:        {metrics.event_recall:.1%}")
    print(f"    Miss rate:           {metrics.miss_rate:.1%}")
    
    if metrics.warning_times:
        print(f"\n  Warning Time Distribution (seconds):")
        print(f"    Mean:                {metrics.mean_warning_time:.4f}")
        print(f"    Median:              {metrics.median_warning_time:.4f}")
        print(f"    Std:                 {metrics.std_warning_time:.4f}")
        print(f"    Min:                 {metrics.min_warning_time:.4f}")
        print(f"    Max:                 {metrics.max_warning_time:.4f}")
        print(f"    Percentiles:")
        print(f"      P10:               {metrics.p10_warning_time:.4f}")
        print(f"      P25:               {metrics.p25_warning_time:.4f}")
        print(f"      P75:               {metrics.p75_warning_time:.4f}")
        print(f"      P90:               {metrics.p90_warning_time:.4f}")
    
    print(f"\n  False Alarm Statistics:")
    print(f"    Total false alarms:              {metrics.n_false_alarms}")
    print(f"    Non-event discharges:            {metrics.n_non_event_discharges}")
    print(f"    Discharges with false alarms:    {metrics.n_discharges_with_false_alarms}")
    print(f"    FA per discharge:                {metrics.false_alarms_per_discharge:.2f}")
    print(f"    FA per non-event discharge:      {metrics.false_alarms_per_non_event_discharge:.2f}")
    print(f"    FA per hour:                     {metrics.false_alarm_rate_per_hour:.2f}")


def metrics_to_dict(metrics: AlarmMetrics) -> Dict[str, Any]:
    """Convert AlarmMetrics to dictionary for JSON serialization."""
    return {
        "n_event_discharges": metrics.n_event_discharges,
        "n_events_detected": metrics.n_events_detected,
        "event_recall": metrics.event_recall,
        "miss_rate": metrics.miss_rate,
        "warning_times": {
            "mean": metrics.mean_warning_time,
            "median": metrics.median_warning_time,
            "std": metrics.std_warning_time,
            "min": metrics.min_warning_time,
            "max": metrics.max_warning_time,
            "p10": metrics.p10_warning_time,
            "p25": metrics.p25_warning_time,
            "p75": metrics.p75_warning_time,
            "p90": metrics.p90_warning_time,
        },
        "false_alarms": {
            "total": metrics.n_false_alarms,
            "per_discharge": metrics.false_alarms_per_discharge,
            "per_non_event_discharge": metrics.false_alarms_per_non_event_discharge,
            "per_hour": metrics.false_alarm_rate_per_hour,
        },
        "config": {
            "threshold": metrics.config.threshold,
            "n_consecutive": metrics.config.n_consecutive,
            "use_ema": metrics.config.use_ema,
            "ema_alpha": metrics.config.ema_alpha,
        }
    }
