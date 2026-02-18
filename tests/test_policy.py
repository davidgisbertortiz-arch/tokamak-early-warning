"""Tests for src/alerting/policy.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.alerting.policy import (
    AlertLevel,
    AlertState,
    AlertResult,
    AlertPolicy,
    print_alert_policy_report,
)
# Exercise src/alerting/__init__
from src.alerting import AlertPolicy as AlertPolicyPkg


# ── AlertState ───────────────────────────────────────────────────────────────

def test_alert_state_defaults():
    s = AlertState(discharge_id=1)
    assert s.current_level == AlertLevel.GREEN
    assert s.consecutive_positive == 0
    assert s.consecutive_uncertain == 0


def test_alert_state_update_positive():
    s = AlertState(discharge_id=1)
    s.update({1}, 0.9, 0.0)
    assert s.consecutive_positive == 1
    assert s.consecutive_uncertain == 0
    s.update({1}, 0.95, 0.01)
    assert s.consecutive_positive == 2


def test_alert_state_update_uncertain():
    s = AlertState(discharge_id=1)
    s.update({0, 1}, 0.5, 0.0)
    assert s.consecutive_uncertain == 1
    assert s.consecutive_positive == 0


def test_alert_state_update_negative_resets():
    s = AlertState(discharge_id=1)
    s.update({1}, 0.9, 0.0)
    s.update({0}, 0.1, 0.01)
    assert s.consecutive_positive == 0
    assert s.consecutive_uncertain == 0


def test_alert_state_time_tracking():
    s = AlertState(discharge_id=1)
    s.current_level = AlertLevel.YELLOW
    s.update({0, 1}, 0.5, 0.0)
    s.update({0, 1}, 0.5, 0.1)
    assert s.time_in_yellow == pytest.approx(0.1)


def test_alert_state_red_time_tracking():
    s = AlertState(discharge_id=1)
    s.current_level = AlertLevel.RED
    s.update({1}, 0.9, 0.0)
    s.update({1}, 0.9, 0.05)
    assert s.time_in_red == pytest.approx(0.05)


# ── AlertPolicy transitions ─────────────────────────────────────────────────

@pytest.fixture
def policy():
    return AlertPolicy(
        prob_yellow_threshold=0.3,
        prob_red_threshold=0.7,
        n_consecutive_for_red=2,
        n_consecutive_clear=3,
    )


def test_green_to_yellow_uncertain(policy):
    state = AlertState(discharge_id=1)
    state.current_level = AlertLevel.GREEN
    level = policy._update_level(state, {0, 1}, 0.5)
    assert level == AlertLevel.YELLOW


def test_green_to_yellow_positive(policy):
    state = AlertState(discharge_id=1)
    state.current_level = AlertLevel.GREEN
    level = policy._update_level(state, {1}, 0.5)
    assert level == AlertLevel.YELLOW


def test_green_to_red_high_prob(policy):
    state = AlertState(discharge_id=1)
    state.current_level = AlertLevel.GREEN
    level = policy._update_level(state, {1}, 0.9)
    assert level == AlertLevel.RED


def test_green_stays_green(policy):
    state = AlertState(discharge_id=1)
    state.current_level = AlertLevel.GREEN
    level = policy._update_level(state, {0}, 0.1)
    assert level == AlertLevel.GREEN


def test_yellow_to_red_consecutive(policy):
    state = AlertState(discharge_id=1)
    state.current_level = AlertLevel.YELLOW
    state.consecutive_positive = 2
    level = policy._update_level(state, {1}, 0.5)
    assert level == AlertLevel.RED


def test_yellow_to_red_high_prob(policy):
    state = AlertState(discharge_id=1)
    state.current_level = AlertLevel.YELLOW
    level = policy._update_level(state, {1}, 0.9)
    assert level == AlertLevel.RED


def test_yellow_to_green(policy):
    state = AlertState(discharge_id=1)
    state.current_level = AlertLevel.YELLOW
    state.consecutive_uncertain = 0
    level = policy._update_level(state, {0}, 0.05)
    assert level == AlertLevel.GREEN


def test_yellow_stays_yellow(policy):
    state = AlertState(discharge_id=1)
    state.current_level = AlertLevel.YELLOW
    level = policy._update_level(state, {0, 1}, 0.4)
    assert level == AlertLevel.YELLOW


def test_red_to_yellow_negative(policy):
    state = AlertState(discharge_id=1)
    state.current_level = AlertLevel.RED
    level = policy._update_level(state, {0}, 0.1)
    assert level == AlertLevel.YELLOW


def test_red_to_yellow_uncertain(policy):
    state = AlertState(discharge_id=1)
    state.current_level = AlertLevel.RED
    level = policy._update_level(state, {0, 1}, 0.5)
    assert level == AlertLevel.YELLOW


def test_red_stays_red(policy):
    state = AlertState(discharge_id=1)
    state.current_level = AlertLevel.RED
    level = policy._update_level(state, {1}, 0.8)
    assert level == AlertLevel.RED


# ── process_discharge ────────────────────────────────────────────────────────

def test_process_discharge_all_green(policy):
    sets = [{0}] * 10
    probs = np.full(10, 0.05)
    times = np.linspace(0, 1, 10)
    result = policy.process_discharge(1, sets, probs, times)

    assert isinstance(result, AlertResult)
    assert all(a == AlertLevel.GREEN for a in result.alerts)
    assert result.n_yellow_alerts == 0
    assert result.n_red_alerts == 0
    assert result.time_to_first_yellow is None
    assert result.time_to_first_red is None


def test_process_discharge_escalation(policy):
    """GREEN -> YELLOW -> RED sequence."""
    sets = [{0}, {0}, {0, 1}, {1}, {1}, {1}]
    probs = np.array([0.05, 0.05, 0.4, 0.5, 0.6, 0.8])
    times = np.linspace(0, 0.5, 6)
    result = policy.process_discharge(1, sets, probs, times)

    assert result.time_to_first_yellow is not None
    assert result.n_yellow_alerts >= 1


def test_process_discharge_empty():
    pol = AlertPolicy()
    result = pol.process_discharge(1, [], np.array([]), np.array([]))
    assert result.alerts == []
    assert result.n_yellow_alerts == 0


# ── check_acceptance_criteria ────────────────────────────────────────────────

def test_acceptance_criteria_pass(policy):
    result = policy.check_acceptance_criteria(
        event_recall=0.95,
        false_alarms_per_discharge=0.5,
        mean_lead_time=0.1,
    )
    assert all(result.values())


def test_acceptance_criteria_fail(policy):
    result = policy.check_acceptance_criteria(
        event_recall=0.5,
        false_alarms_per_discharge=5.0,
        mean_lead_time=0.001,
    )
    assert not any(result.values())


# ── print_alert_policy_report ────────────────────────────────────────────────

def test_print_report(policy, capsys):
    sets = [{0}, {0, 1}, {1}, {1}]
    probs = np.array([0.05, 0.4, 0.6, 0.85])
    times = np.linspace(0, 0.3, 4)
    alert_result = policy.process_discharge(1, sets, probs, times)

    print_alert_policy_report(
        policy,
        [alert_result],
        event_recall=0.9,
        false_alarms_per_discharge=0.5,
        mean_lead_time=0.1,
    )
    out = capsys.readouterr().out
    assert "Alert Policy Evaluation" in out
    assert "Acceptance Criteria" in out
    assert "PASS" in out
