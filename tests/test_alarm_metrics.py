"""
Tests for alarm/event-level metrics.

Verifies that warning time computation and event detection work correctly
on synthetic examples with known outcomes.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.evaluation.alarm_metrics import (
    AlarmConfig,
    compute_alarm_metrics,
    detect_alarm_triggers,
    apply_ema,
    evaluate_discharge_alarms
)


@pytest.fixture
def simple_event_discharge():
    """
    Create a simple discharge with an event at t=0.8.
    
    Timeline:
    - t=0.0-0.7: normal operation (target=0)
    - t=0.8-1.0: event (target=1)
    """
    n_points = 100
    times = np.linspace(0, 1, n_points)
    
    df = pd.DataFrame({
        "discharge_ID": 1,
        "time": times,
        "density_limit_phase": (times >= 0.8).astype(int)
    })
    
    return df


@pytest.fixture
def non_event_discharge():
    """Create a discharge with no event."""
    n_points = 100
    times = np.linspace(0, 1, n_points)
    
    df = pd.DataFrame({
        "discharge_ID": 2,
        "time": times,
        "density_limit_phase": 0
    })
    
    return df


def test_warning_time_computation_perfect_detection(simple_event_discharge):
    """
    Test warning time when alarm triggers well before event.
    
    Setup: Alarm at t=0.5, event at t=0.8
    Expected warning time: 0.8 - 0.5 = 0.3s
    """
    df = simple_event_discharge.copy()
    n_points = len(df)
    
    # Probabilities: high around t=0.5, triggering alarm before event
    probs = np.zeros(n_points)
    probs[45:55] = 0.9  # High probs around t=0.5
    
    config = AlarmConfig(threshold=0.5, n_consecutive=1)
    result = evaluate_discharge_alarms(df, probs, config)
    
    assert result.has_event == True
    assert result.event_detected == True
    assert result.warning_time is not None
    
    # Warning time should be approximately 0.8 - 0.45 ≈ 0.35
    # (first high prob at index 45, which is t ≈ 0.45)
    assert 0.25 < result.warning_time < 0.45


def test_warning_time_late_detection(simple_event_discharge):
    """
    Test warning time when alarm triggers just before event.
    
    Setup: Alarm at t=0.75, event at t=0.8
    Expected warning time: ~0.05s
    """
    df = simple_event_discharge.copy()
    n_points = len(df)
    
    # Probabilities: only high just before event
    probs = np.zeros(n_points)
    probs[74:78] = 0.9  # High probs around t=0.75
    
    config = AlarmConfig(threshold=0.5, n_consecutive=1)
    result = evaluate_discharge_alarms(df, probs, config)
    
    assert result.event_detected == True
    assert result.warning_time is not None
    assert result.warning_time < 0.1  # Very short warning


def test_missed_event(simple_event_discharge):
    """
    Test when no alarm triggers before event (missed detection).
    """
    df = simple_event_discharge.copy()
    n_points = len(df)
    
    # No high probabilities before event
    probs = np.zeros(n_points)
    probs[85:95] = 0.9  # Only high after event at t=0.8
    
    config = AlarmConfig(threshold=0.5, n_consecutive=1)
    result = evaluate_discharge_alarms(df, probs, config)
    
    assert result.event_detected == False
    assert result.warning_time is None


def test_false_alarm_count_non_event_discharge(non_event_discharge):
    """
    Test false alarm counting on non-event discharge.
    
    All alarms on a non-event discharge are false alarms.
    """
    df = non_event_discharge.copy()
    n_points = len(df)
    
    # Multiple high probability regions
    probs = np.zeros(n_points)
    probs[20:25] = 0.9  # False alarm 1
    probs[50:55] = 0.9  # False alarm 2
    probs[70:75] = 0.9  # False alarm 3
    
    config = AlarmConfig(threshold=0.5, n_consecutive=1)
    result = evaluate_discharge_alarms(df, probs, config)
    
    assert result.has_event == False
    assert result.n_false_alarms > 0


def test_consecutive_steps_requirement():
    """
    Test that n_consecutive parameter filters out transient spikes.
    """
    probs = np.array([0.0, 0.8, 0.0, 0.8, 0.8, 0.8, 0.0])
    
    # n_consecutive=1: triggers on any threshold crossing
    config1 = AlarmConfig(threshold=0.5, n_consecutive=1)
    triggers1 = detect_alarm_triggers(probs, config1)
    assert triggers1.sum() >= 2  # Multiple triggers
    
    # n_consecutive=3: requires 3 consecutive above threshold
    config3 = AlarmConfig(threshold=0.5, n_consecutive=3)
    triggers3 = detect_alarm_triggers(probs, config3)
    # Only indices 3,4,5 form 3 consecutive, so trigger at index 5
    assert triggers3.sum() == 1


def test_ema_smoothing():
    """Test EMA smoothing reduces noise."""
    # Noisy signal
    probs = np.array([0.0, 0.8, 0.0, 0.8, 0.0, 0.8, 0.0])
    
    smoothed = apply_ema(probs, alpha=0.3)
    
    # Smoothed signal should have smaller variance
    assert np.std(smoothed) < np.std(probs)
    
    # Smoothed values should be bounded by original range
    assert smoothed.min() >= 0
    assert smoothed.max() <= 0.8


def test_aggregate_metrics_computation():
    """Test aggregate metrics across multiple discharges."""
    # Create mixed dataset
    data = []
    
    # Discharge 1: Event at t=0.8, alarm at t=0.5
    for t in np.linspace(0, 1, 50):
        data.append({
            "discharge_ID": 1,
            "time": t,
            "density_limit_phase": 1 if t >= 0.8 else 0
        })
    
    # Discharge 2: Event at t=0.9, no alarm
    for t in np.linspace(0, 1, 50):
        data.append({
            "discharge_ID": 2,
            "time": t,
            "density_limit_phase": 1 if t >= 0.9 else 0
        })
    
    # Discharge 3: No event
    for t in np.linspace(0, 1, 50):
        data.append({
            "discharge_ID": 3,
            "time": t,
            "density_limit_phase": 0
        })
    
    df = pd.DataFrame(data)
    
    # Probabilities: alarm only for discharge 1
    probs = np.zeros(len(df))
    probs[20:30] = 0.9  # Discharge 1: alarm around t=0.5
    probs[120:130] = 0.9  # Discharge 3: false alarm
    
    config = AlarmConfig(threshold=0.5, n_consecutive=1)
    metrics = compute_alarm_metrics(df, probs, config)
    
    # Should have 2 event discharges, 1 detected
    assert metrics.n_event_discharges == 2
    assert metrics.n_events_detected == 1
    assert 0.4 < metrics.event_recall < 0.6  # ~50% recall
    
    # Should have false alarms from discharge 3
    assert metrics.n_false_alarms > 0


def test_event_recall_100_percent():
    """Test perfect event detection scenario."""
    data = []
    
    # Create 5 discharges with events, all detected
    for discharge_id in range(5):
        for t in np.linspace(0, 1, 20):
            data.append({
                "discharge_ID": discharge_id,
                "time": t,
                "density_limit_phase": 1 if t >= 0.8 else 0
            })
    
    df = pd.DataFrame(data)
    
    # High probabilities before all events
    probs = np.tile(
        np.concatenate([np.zeros(10), np.ones(10) * 0.9]),
        5
    )
    
    config = AlarmConfig(threshold=0.5, n_consecutive=1)
    metrics = compute_alarm_metrics(df, probs, config)
    
    assert metrics.n_event_discharges == 5
    assert metrics.event_recall == 1.0  # 100% recall
