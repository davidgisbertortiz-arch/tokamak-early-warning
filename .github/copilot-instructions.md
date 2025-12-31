# Copilot Instructions for tokamak-early-warning

## Project Overview
ML pipeline for early warning of tokamak density limit events using MIT-PSFC dataset (~264k samples, ~1-2% positive class). **Core insight**: Standard ML metrics (accuracy, AUC) don't capture operational value—we evaluate at the *alarm level*: Did we warn operators before the event? How early? How many false alarms?

## Critical Rule: Prevent Data Leakage
**ALWAYS split by `discharge_ID`, never by individual samples.** Time points within a discharge are temporally correlated.

```python
# CORRECT - splits entire discharges together
from src.data.dataset import split_by_discharge_with_cal
train, cal, val, test = split_by_discharge_with_cal(df, random_state=42)

# WRONG - causes temporal leakage across train/test
from sklearn.model_selection import train_test_split
train, test = train_test_split(df)  # NEVER DO THIS
```

## Evaluation Philosophy
Sample-level metrics hide operational failures. Always evaluate both:

| Level | Metrics | Module |
|-------|---------|--------|
| **Sample** | PR-AUC, ROC-AUC | `src/evaluation/metrics.py` |
| **Alarm** | Event recall, warning time, false alarm rate | `src/evaluation/alarm_metrics.py` |
| **Calibration** | ECE (target < 0.05) | `src/uncertainty/calibration.py` |

```python
from src.evaluation.alarm_metrics import AlarmConfig, compute_alarm_metrics
config = AlarmConfig(threshold=0.5, n_consecutive=2, use_ema=True, ema_alpha=0.3)
metrics = compute_alarm_metrics(df, probs, config)
# metrics.event_recall, metrics.warning_times, metrics.false_alarms_per_discharge
```

## Key Patterns

### Feature Engineering (respects discharge boundaries)
```python
from src.features.temporal import engineer_features, get_engineered_feature_names, FeatureConfig
config = FeatureConfig(rolling_windows=(3, 5, 10), include_greenwald_proxy=True)
df_eng = engineer_features(df, config)  # Physics-informed: derivatives, rolling stats, Greenwald fraction
```

### TCN Temporal Modeling
```python
from src.models.tcn import TCNClassifier, TCNConfig, create_sequences
X_train, y_train, _ = create_sequences(train_df, feature_cols, TARGET_COLUMN, seq_len=20)
model = TCNClassifier(TCNConfig(seq_len=20, hidden_channels=32, n_blocks=3))
model.fit(X_train, y_train, X_val, y_val)
```

### Alert State Machine
`src/alerting/policy.py` provides GREEN/YELLOW/RED operational states based on conformal prediction sets.

## Quick Commands
```bash
./scripts/fetch_data.sh                          # Download dataset (idempotent)
python scripts/baseline.py                       # Logistic regression baseline
python scripts/train_tcn.py --epochs 30          # Full TCN pipeline
python scripts/evaluate_alarm_policy.py --model both --save-figures
pytest tests/ -v
```

## Key Constants
```python
from src.data.dataset import FEATURE_COLUMNS, TARGET_COLUMN
# FEATURE_COLUMNS = ["density", "elongation", "minor_radius", "plasma_current", "toroidal_B_field", "triangularity"]
# TARGET_COLUMN = "density_limit_phase"
```

## When Adding Models
1. Place in `src/models/` with sklearn-compatible `fit`/`predict_proba` interface
2. Use `class_weight="balanced"` or `pos_weight` for class imbalance
3. Evaluate with **all three** metric types (sample, alarm, calibration)
4. Use `random_state=42` for reproducibility

## Conventions
- PR-AUC is the primary sample-level metric (accuracy is meaningless with 1-2% positive rate)
- Results/figures save to `results/` directory
- Tests in `tests/` verify leakage prevention and metric correctness
