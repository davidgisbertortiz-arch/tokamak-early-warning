"""Tests for src/evaluation/event_metrics.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.evaluation.event_metrics import (
    compute_event_metrics,
    compute_lead_time_distribution,
    print_event_metrics_report,
    EventDetection,
    EventMetrics,
)


def _make_discharge(discharge_id, n=50, event_start_frac=0.8):
    """Helper: one discharge with an event starting at event_start_frac."""
    times = np.linspace(0, 1, n)
    labels = (times >= event_start_frac).astype(int)
    return pd.DataFrame({
        "discharge_ID": discharge_id,
        "time": times,
        "density_limit_phase": labels,
    })


def _make_no_event_discharge(discharge_id, n=50):
    return pd.DataFrame({
        "discharge_ID": discharge_id,
        "time": np.linspace(0, 1, n),
        "density_limit_phase": np.zeros(n, dtype=int),
    })


@pytest.fixture
def df_with_events():
    """Two event discharges + one non-event discharge."""
    return pd.concat([
        _make_discharge(1),
        _make_discharge(2, event_start_frac=0.6),
        _make_no_event_discharge(3),
    ], ignore_index=True)


# ── compute_event_metrics ────────────────────────────────────────────────────

def test_perfect_detection(df_with_events):
    """Predict 1 everywhere before event → should detect all events."""
    preds = np.ones(len(df_with_events), dtype=int)
    m = compute_event_metrics(df_with_events, preds)

    assert m.n_events == 2
    assert m.n_detected == 2
    assert m.event_recall == 1.0
    assert len(m.lead_times) == 2
    assert all(lt > 0 for lt in m.lead_times)
    # Non-event discharge causes false alarms
    assert m.n_false_alarms > 0


def test_no_detection(df_with_events):
    """Predict 0 everywhere → zero recall."""
    preds = np.zeros(len(df_with_events), dtype=int)
    m = compute_event_metrics(df_with_events, preds)

    assert m.n_events == 2
    assert m.n_detected == 0
    assert m.event_recall == 0.0
    assert m.lead_times == []
    assert m.mean_lead_time == 0.0
    assert m.n_false_alarms == 0


def test_partial_detection(df_with_events):
    """Alert only inside discharge 1 before its event."""
    preds = np.zeros(len(df_with_events), dtype=int)
    # Discharge 1: set prediction=1 at index 35 (time ~0.71, before event at 0.8)
    preds[35] = 1
    m = compute_event_metrics(df_with_events, preds)

    assert m.n_detected == 1  # only discharge 1 detected
    assert m.event_recall == pytest.approx(0.5)
    assert len(m.lead_times) == 1


def test_max_lead_time_filter(df_with_events):
    """With max_lead_time, very early alerts become false alarms."""
    preds = np.zeros(len(df_with_events), dtype=int)
    # Alert very early (index 5 ~ time 0.1) in discharge 1 (event at 0.8)
    preds[5] = 1
    m = compute_event_metrics(df_with_events, preds, max_lead_time=0.2)

    # The early alert is outside the 0.2s window before event so not a detection
    assert m.n_detected == 0


def test_no_events():
    """Dataset with no events at all."""
    df = _make_no_event_discharge(1)
    preds = np.zeros(len(df), dtype=int)
    m = compute_event_metrics(df, preds)

    assert m.n_events == 0
    assert m.event_recall == 0.0
    assert m.n_false_alarms == 0


def test_false_alarms_on_non_event_discharge():
    df = _make_no_event_discharge(1)
    preds = np.ones(len(df), dtype=int)
    m = compute_event_metrics(df, preds)

    assert m.n_false_alarms == len(df)
    assert m.n_discharges_with_false_alarms == 1


# ── compute_lead_time_distribution ───────────────────────────────────────────

def test_lead_time_distribution(df_with_events):
    preds = np.ones(len(df_with_events), dtype=int)
    m = compute_event_metrics(df_with_events, preds)
    dist = compute_lead_time_distribution(m)

    assert "p50" in dist
    assert all(isinstance(v, float) for v in dist.values())


def test_lead_time_distribution_empty():
    m = EventMetrics(
        n_events=0, n_detected=0, event_recall=0.0,
        lead_times=[], mean_lead_time=0.0, median_lead_time=0.0,
        min_lead_time=0.0, max_lead_time=0.0,
        n_false_alarms=0, n_discharges_with_false_alarms=0,
        false_alarm_rate_per_discharge=0.0,
        total_time_hours=0.0, false_alarms_per_hour=0.0,
        event_detections=[],
    )
    dist = compute_lead_time_distribution(m)
    assert all(v == 0.0 for v in dist.values())


# ── print_event_metrics_report ───────────────────────────────────────────────

def test_print_report_with_events(df_with_events, capsys):
    preds = np.ones(len(df_with_events), dtype=int)
    m = compute_event_metrics(df_with_events, preds)
    print_event_metrics_report(m, "Test")
    out = capsys.readouterr().out
    assert "Event Recall" in out
    assert "Lead Time" in out
    assert "False Alarms" in out


def test_print_report_no_detections(df_with_events, capsys):
    preds = np.zeros(len(df_with_events), dtype=int)
    m = compute_event_metrics(df_with_events, preds)
    print_event_metrics_report(m)
    out = capsys.readouterr().out
    assert "Event Recall" in out
