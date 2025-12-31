"""
Event-based metrics for early warning evaluation.

Standard ML metrics (PR-AUC, ROC-AUC) evaluate sample-level performance.
For early warning systems, we need EVENT-level metrics:

- Event-level recall: % of actual events detected before they occur
- Lead time distribution: How early do we detect events?
- False alarm rate: False alarms per discharge / per hour

These metrics better reflect operational utility.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EventDetection:
    """Result of detecting a single event."""
    discharge_id: int
    event_time: float  # Time when event started
    detected: bool  # Was event detected before it started?
    first_alert_time: Optional[float]  # Time of first alert (if detected)
    lead_time: Optional[float]  # event_time - first_alert_time (if detected)
    n_alerts_before_event: int  # Number of alerts before event
    
    
@dataclass
class EventMetrics:
    """Container for all event-based metrics."""
    # Event detection
    n_events: int
    n_detected: int
    event_recall: float  # n_detected / n_events
    
    # Lead time statistics
    lead_times: List[float]  # Lead time for each detected event
    mean_lead_time: float
    median_lead_time: float
    min_lead_time: float
    max_lead_time: float
    
    # False alarms
    n_false_alarms: int  # Alerts on non-event discharges or too early
    n_discharges_with_false_alarms: int
    false_alarm_rate_per_discharge: float  # Per discharge with events
    total_time_hours: float
    false_alarms_per_hour: float
    
    # Per-discharge details
    event_detections: List[EventDetection]


def compute_event_metrics(
    df: pd.DataFrame,
    predictions: np.ndarray,
    max_lead_time: float = None,
    time_col: str = "time",
    discharge_col: str = "discharge_ID",
    target_col: str = "density_limit_phase"
) -> EventMetrics:
    """
    Compute event-based metrics for early warning evaluation.
    
    An "event" is defined as the first time point where target_col == 1
    within a discharge. Detection is successful if there's an alert
    (prediction == 1) before the event.
    
    Args:
        df: DataFrame with time, discharge_ID, and target columns
        predictions: Binary predictions (0/1) aligned with df
        max_lead_time: Maximum lead time to consider (None = unlimited)
        time_col: Name of time column
        discharge_col: Name of discharge ID column
        target_col: Name of target column
        
    Returns:
        EventMetrics with all computed metrics
    """
    df = df.copy()
    df["_pred"] = predictions
    
    event_detections = []
    false_alarms_by_discharge = []
    total_time = 0.0
    
    for discharge_id in df[discharge_col].unique():
        discharge_df = df[df[discharge_col] == discharge_id].sort_values(time_col)
        
        # Calculate discharge duration for false alarm rate
        discharge_duration = discharge_df[time_col].max() - discharge_df[time_col].min()
        total_time += discharge_duration
        
        # Find event time (first positive label)
        event_mask = discharge_df[target_col] == 1
        has_event = event_mask.any()
        
        if has_event:
            event_time = discharge_df.loc[event_mask, time_col].iloc[0]
            
            # Get predictions before event
            before_event = discharge_df[discharge_df[time_col] < event_time]
            
            if max_lead_time is not None:
                # Only consider alerts within max_lead_time of event
                before_event = before_event[
                    before_event[time_col] >= (event_time - max_lead_time)
                ]
            
            alerts_before = before_event[before_event["_pred"] == 1]
            n_alerts = len(alerts_before)
            detected = n_alerts > 0
            
            if detected:
                first_alert_time = alerts_before[time_col].iloc[0]
                lead_time = event_time - first_alert_time
            else:
                first_alert_time = None
                lead_time = None
            
            event_detections.append(EventDetection(
                discharge_id=discharge_id,
                event_time=event_time,
                detected=detected,
                first_alert_time=first_alert_time,
                lead_time=lead_time,
                n_alerts_before_event=n_alerts
            ))
            
            # Count false alarms (alerts way before event, if max_lead_time set)
            if max_lead_time is not None:
                too_early = discharge_df[
                    (discharge_df[time_col] < (event_time - max_lead_time)) &
                    (discharge_df["_pred"] == 1)
                ]
                false_alarms_by_discharge.append(len(too_early))
        else:
            # No event in this discharge - all alerts are false alarms
            n_false = (discharge_df["_pred"] == 1).sum()
            false_alarms_by_discharge.append(n_false)
    
    # Aggregate metrics
    n_events = len(event_detections)
    n_detected = sum(1 for e in event_detections if e.detected)
    lead_times = [e.lead_time for e in event_detections if e.lead_time is not None]
    
    n_false_alarms = sum(false_alarms_by_discharge)
    n_discharges_with_false_alarms = sum(1 for fa in false_alarms_by_discharge if fa > 0)
    
    total_time_hours = total_time / 3600 if total_time > 0 else 1e-6
    
    return EventMetrics(
        n_events=n_events,
        n_detected=n_detected,
        event_recall=n_detected / n_events if n_events > 0 else 0.0,
        lead_times=lead_times,
        mean_lead_time=np.mean(lead_times) if lead_times else 0.0,
        median_lead_time=np.median(lead_times) if lead_times else 0.0,
        min_lead_time=np.min(lead_times) if lead_times else 0.0,
        max_lead_time=np.max(lead_times) if lead_times else 0.0,
        n_false_alarms=n_false_alarms,
        n_discharges_with_false_alarms=n_discharges_with_false_alarms,
        false_alarm_rate_per_discharge=n_false_alarms / len(df[discharge_col].unique()),
        total_time_hours=total_time_hours,
        false_alarms_per_hour=n_false_alarms / total_time_hours,
        event_detections=event_detections
    )


def compute_lead_time_distribution(
    event_metrics: EventMetrics,
    percentiles: List[float] = [10, 25, 50, 75, 90]
) -> Dict[str, float]:
    """
    Compute lead time distribution statistics.
    
    Args:
        event_metrics: EventMetrics from compute_event_metrics
        percentiles: Percentiles to compute
        
    Returns:
        Dictionary with lead time statistics
    """
    lead_times = event_metrics.lead_times
    
    if not lead_times:
        return {f"p{p}": 0.0 for p in percentiles}
    
    result = {}
    for p in percentiles:
        result[f"p{p}"] = np.percentile(lead_times, p)
    
    return result


def print_event_metrics_report(metrics: EventMetrics, dataset_name: str = "Test") -> None:
    """Print formatted event-based metrics report."""
    print(f"\n{'='*60}")
    print(f" Event-Based Metrics Report - {dataset_name}")
    print(f"{'='*60}")
    
    print(f"\n  Event Detection:")
    print(f"    Total Events:       {metrics.n_events}")
    print(f"    Events Detected:    {metrics.n_detected}")
    print(f"    Event Recall:       {metrics.event_recall:.1%}")
    
    if metrics.lead_times:
        print(f"\n  Lead Time Distribution (seconds):")
        print(f"    Mean:               {metrics.mean_lead_time:.4f}")
        print(f"    Median:             {metrics.median_lead_time:.4f}")
        print(f"    Min:                {metrics.min_lead_time:.4f}")
        print(f"    Max:                {metrics.max_lead_time:.4f}")
        
        lead_pcts = compute_lead_time_distribution(metrics)
        print(f"    Percentiles:")
        for k, v in lead_pcts.items():
            print(f"      {k}: {v:.4f}s")
    
    print(f"\n  False Alarms:")
    print(f"    Total False Alarms:           {metrics.n_false_alarms}")
    print(f"    Discharges with False Alarms: {metrics.n_discharges_with_false_alarms}")
    print(f"    False Alarms / Discharge:     {metrics.false_alarm_rate_per_discharge:.2f}")
    print(f"    False Alarms / Hour:          {metrics.false_alarms_per_hour:.2f}")
