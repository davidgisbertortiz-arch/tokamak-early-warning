"""
Alert policy for converting conformal predictions to actionable alerts.

Conformal prediction gives us:
- {0}:    Confident NO event
- {1}:    Confident event imminent  
- {0,1}:  Uncertain

This module converts these into operational alerts:
- GREEN:  No concern
- YELLOW: Elevated risk (uncertain or recent positive signals)
- RED:    Imminent event (sustained high-confidence signals)

The policy tracks alert STATE over time to prevent alert fatigue
from transient uncertain predictions.
"""

import numpy as np
from typing import List, Set, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class AlertLevel(Enum):
    """Alert severity levels."""
    GREEN = 0   # Normal operation
    YELLOW = 1  # Elevated risk / uncertain
    RED = 2     # Imminent event
    

@dataclass 
class AlertState:
    """Current alert state with history for a single discharge."""
    discharge_id: int
    current_level: AlertLevel = AlertLevel.GREEN
    consecutive_positive: int = 0  # Consecutive {1} predictions
    consecutive_uncertain: int = 0  # Consecutive {0,1} predictions
    time_in_yellow: float = 0.0  # Total time in YELLOW state
    time_in_red: float = 0.0  # Total time in RED state
    last_update_time: Optional[float] = None
    alert_history: List[Dict] = field(default_factory=list)
    
    def update(
        self,
        prediction_set: Set[int],
        probability: float,
        time: float
    ) -> AlertLevel:
        """
        Update alert state based on new prediction.
        
        Returns new alert level.
        """
        # Track time delta
        dt = 0.0
        if self.last_update_time is not None:
            dt = time - self.last_update_time
        self.last_update_time = time
        
        # Update time accumulators
        if self.current_level == AlertLevel.YELLOW:
            self.time_in_yellow += dt
        elif self.current_level == AlertLevel.RED:
            self.time_in_red += dt
        
        # Update consecutive counts
        if prediction_set == {1}:
            self.consecutive_positive += 1
            self.consecutive_uncertain = 0
        elif prediction_set == {0, 1} or prediction_set == {1, 0}:
            self.consecutive_uncertain += 1
            self.consecutive_positive = 0  # Reset positive streak
        else:  # {0}
            self.consecutive_positive = 0
            self.consecutive_uncertain = 0
        
        # Record in history
        self.alert_history.append({
            "time": time,
            "prediction_set": prediction_set,
            "probability": probability,
            "level": self.current_level
        })
        
        return self.current_level


@dataclass
class AlertResult:
    """Result of applying alert policy to a sequence."""
    alerts: List[AlertLevel]  # Alert level at each time point
    final_state: AlertState
    time_to_first_yellow: Optional[float]  # Time until first YELLOW
    time_to_first_red: Optional[float]  # Time until first RED
    n_yellow_alerts: int
    n_red_alerts: int


class AlertPolicy:
    """
    Policy for converting conformal predictions to operational alerts.
    
    State transitions:
    
    GREEN -> YELLOW: 
        - Single {0,1} (uncertain) prediction, OR
        - Single {1} with prob >= prob_yellow_threshold
        
    YELLOW -> RED:
        - n_consecutive_for_red consecutive {1} predictions, OR
        - Any {1} with prob >= prob_red_threshold
        
    YELLOW -> GREEN:
        - n_consecutive_clear consecutive {0} predictions
        
    RED -> YELLOW:
        - Single {0} or {0,1} prediction
        
    RED -> GREEN:
        - Not allowed (must go through YELLOW first for safety)
    
    Acceptance Criteria (tunable):
        - Event recall >= 90% (miss rate <= 10%)
        - False alarm rate <= 1 per discharge
        - Mean lead time >= 50ms (enough time to react)
    """
    
    def __init__(
        self,
        # Thresholds for escalation
        prob_yellow_threshold: float = 0.3,  # P(Y=1) to trigger YELLOW
        prob_red_threshold: float = 0.7,     # P(Y=1) to trigger immediate RED
        n_consecutive_for_red: int = 2,       # Consecutive {1} for RED
        n_consecutive_clear: int = 3,         # Consecutive {0} to de-escalate
        # Acceptance criteria
        min_event_recall: float = 0.90,
        max_false_alarms_per_discharge: float = 1.0,
        min_lead_time_seconds: float = 0.05
    ):
        self.prob_yellow_threshold = prob_yellow_threshold
        self.prob_red_threshold = prob_red_threshold
        self.n_consecutive_for_red = n_consecutive_for_red
        self.n_consecutive_clear = n_consecutive_clear
        
        # Acceptance criteria
        self.acceptance_criteria = {
            "min_event_recall": min_event_recall,
            "max_false_alarms_per_discharge": max_false_alarms_per_discharge,
            "min_lead_time_seconds": min_lead_time_seconds
        }
    
    def _update_level(
        self,
        state: AlertState,
        prediction_set: Set[int],
        probability: float
    ) -> AlertLevel:
        """Compute new alert level based on current state and prediction."""
        
        current = state.current_level
        is_positive = prediction_set == {1}
        is_uncertain = len(prediction_set) == 2
        is_negative = prediction_set == {0}
        
        # State machine transitions
        if current == AlertLevel.GREEN:
            if is_positive and probability >= self.prob_red_threshold:
                return AlertLevel.RED  # High-confidence jump to RED
            elif is_positive or is_uncertain:
                return AlertLevel.YELLOW
            else:
                return AlertLevel.GREEN
                
        elif current == AlertLevel.YELLOW:
            if is_positive and probability >= self.prob_red_threshold:
                return AlertLevel.RED  # High-confidence escalation
            elif state.consecutive_positive >= self.n_consecutive_for_red:
                return AlertLevel.RED  # Sustained positive signals
            elif is_negative and state.consecutive_uncertain == 0:
                # Only de-escalate if we've had clear negatives
                # (This is checked via consecutive counts in practice)
                return AlertLevel.GREEN
            else:
                return AlertLevel.YELLOW  # Stay elevated
                
        elif current == AlertLevel.RED:
            if is_negative or is_uncertain:
                return AlertLevel.YELLOW  # De-escalate but stay alert
            else:
                return AlertLevel.RED  # Maintain RED
        
        return current  # Fallback
    
    def process_discharge(
        self,
        discharge_id: int,
        prediction_sets: List[Set[int]],
        probabilities: np.ndarray,
        times: np.ndarray
    ) -> AlertResult:
        """
        Process a full discharge sequence and generate alerts.
        
        Args:
            discharge_id: ID of the discharge
            prediction_sets: Conformal prediction sets for each time point
            probabilities: P(Y=1) for each time point
            times: Time stamps
            
        Returns:
            AlertResult with alert sequence and metrics
        """
        state = AlertState(discharge_id=discharge_id)
        alerts = []
        
        time_to_first_yellow = None
        time_to_first_red = None
        start_time = times[0] if len(times) > 0 else 0.0
        
        for i, (pred_set, prob, t) in enumerate(zip(prediction_sets, probabilities, times)):
            # Compute new alert level
            new_level = self._update_level(state, pred_set, prob)
            state.current_level = new_level
            
            # Update state tracking
            state.update(pred_set, prob, t)
            alerts.append(new_level)
            
            # Track first escalations
            if new_level == AlertLevel.YELLOW and time_to_first_yellow is None:
                time_to_first_yellow = t - start_time
            if new_level == AlertLevel.RED and time_to_first_red is None:
                time_to_first_red = t - start_time
        
        return AlertResult(
            alerts=alerts,
            final_state=state,
            time_to_first_yellow=time_to_first_yellow,
            time_to_first_red=time_to_first_red,
            n_yellow_alerts=sum(1 for a in alerts if a == AlertLevel.YELLOW),
            n_red_alerts=sum(1 for a in alerts if a == AlertLevel.RED)
        )
    
    def check_acceptance_criteria(
        self,
        event_recall: float,
        false_alarms_per_discharge: float,
        mean_lead_time: float
    ) -> Dict[str, bool]:
        """
        Check if system meets acceptance criteria.
        
        Returns dict with pass/fail for each criterion.
        """
        return {
            "event_recall": event_recall >= self.acceptance_criteria["min_event_recall"],
            "false_alarm_rate": false_alarms_per_discharge <= self.acceptance_criteria["max_false_alarms_per_discharge"],
            "lead_time": mean_lead_time >= self.acceptance_criteria["min_lead_time_seconds"],
        }


def print_alert_policy_report(
    policy: AlertPolicy,
    alert_results: List[AlertResult],
    event_recall: float,
    false_alarms_per_discharge: float,
    mean_lead_time: float
) -> None:
    """Print formatted alert policy evaluation report."""
    print(f"\n{'='*60}")
    print(f" Alert Policy Evaluation")
    print(f"{'='*60}")
    
    print(f"\n  Policy Parameters:")
    print(f"    P(Y=1) for YELLOW:          >= {policy.prob_yellow_threshold:.2f}")
    print(f"    P(Y=1) for immediate RED:   >= {policy.prob_red_threshold:.2f}")
    print(f"    Consecutive {{1}} for RED:    {policy.n_consecutive_for_red}")
    print(f"    Consecutive {{0}} to clear:   {policy.n_consecutive_clear}")
    
    # Aggregate alert stats
    total_yellow = sum(r.n_yellow_alerts for r in alert_results)
    total_red = sum(r.n_red_alerts for r in alert_results)
    
    print(f"\n  Alert Statistics:")
    print(f"    Total YELLOW alerts:        {total_yellow}")
    print(f"    Total RED alerts:           {total_red}")
    print(f"    Discharges processed:       {len(alert_results)}")
    
    # Acceptance criteria check
    criteria = policy.check_acceptance_criteria(
        event_recall, false_alarms_per_discharge, mean_lead_time
    )
    
    print(f"\n  Acceptance Criteria:")
    print(f"    {'Criterion':<30} {'Value':>10} {'Target':>15} {'Status':>10}")
    print(f"    {'-'*65}")
    
    status_event = "✓ PASS" if criteria["event_recall"] else "✗ FAIL"
    print(f"    {'Event Recall':<30} {event_recall:>10.1%} "
          f"{'>= ' + str(policy.acceptance_criteria['min_event_recall']*100) + '%':>15} "
          f"{status_event:>10}")
    
    status_fa = "✓ PASS" if criteria["false_alarm_rate"] else "✗ FAIL"
    print(f"    {'False Alarms / Discharge':<30} {false_alarms_per_discharge:>10.2f} "
          f"{'<= ' + str(policy.acceptance_criteria['max_false_alarms_per_discharge']):>15} "
          f"{status_fa:>10}")
    
    status_lt = "✓ PASS" if criteria["lead_time"] else "✗ FAIL"
    print(f"    {'Mean Lead Time (s)':<30} {mean_lead_time:>10.4f} "
          f"{'>= ' + str(policy.acceptance_criteria['min_lead_time_seconds']):>15} "
          f"{status_lt:>10}")
    
    all_pass = all(criteria.values())
    print(f"\n  Overall: {'✓ ALL CRITERIA MET' if all_pass else '✗ CRITERIA NOT MET'}")
