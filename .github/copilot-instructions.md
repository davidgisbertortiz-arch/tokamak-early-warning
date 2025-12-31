# Copilot Instructions for tokamak-early-warning

## Project Overview
Production-grade ML pipeline for early warning prediction of tokamak density limit events. Uses the MIT-PSFC Open Density Limit dataset (~264k samples, ~1-2% positive class). **Key differentiator**: Alarm-level evaluation (event recall, warning time, false alarm rate) not just sample-level metrics.

## Architecture
```
src/
├── data/dataset.py           # Data loading, discharge-based splitting
├── features/temporal.py      # Physics-informed feature engineering
├── models/
│   ├── baseline_lr.py        # StandardScaler + LogisticRegression
│   └── tcn.py                # Temporal Convolutional Network
├── evaluation/
│   ├── metrics.py            # PR-AUC, ROC-AUC
│   └── alarm_metrics.py      # Event recall, warning time, FAR
├── uncertainty/
│   ├── conformal.py          # Split conformal prediction
│   └── calibration.py        # Temperature scaling, ECE
└── alerting/policy.py        # GREEN/YELLOW/RED state machine

scripts/
├── fetch_data.sh             # Download dataset (idempotent)
├── baseline.py               # Basic LR training
├── train_tcn.py              # TCN training + full alarm evaluation
├── evaluate_alarm_policy.py  # Compare models, generate figures
└── make_early_warning_labels.py
```

## Critical Patterns

### 1. Data Leakage Prevention (MANDATORY)
**ALWAYS split by `discharge_ID`, never by samples.** Time points from same discharge are correlated.

```python
# CORRECT
from src.data.dataset import split_by_discharge_with_cal
train, cal, val, test = split_by_discharge_with_cal(df, random_state=42)

# WRONG - causes leakage
train, test = train_test_split(df)  # NEVER DO THIS
```

### 2. Alarm-Level Evaluation (Not Just PR-AUC!)
Standard ML metrics don't capture operational utility. Use alarm metrics:

```python
from src.evaluation.alarm_metrics import AlarmConfig, compute_alarm_metrics

config = AlarmConfig(
    threshold=0.5,
    n_consecutive=2,   # Require 2 consecutive predictions
    use_ema=True,
    ema_alpha=0.3      # EMA smoothing
)
metrics = compute_alarm_metrics(df, probs, config)
# metrics.event_recall: % events detected BEFORE they occur
# metrics.warning_times: distribution of (t_event - t_first_alarm)
# metrics.false_alarms_per_discharge
```

### 3. TCN Temporal Modeling
```python
from src.models.tcn import TCNClassifier, TCNConfig, create_sequences

config = TCNConfig(seq_len=20, hidden_channels=32, n_blocks=3)
X_train, y_train, _ = create_sequences(train_df, feature_cols, TARGET_COLUMN, seq_len=20)

model = TCNClassifier(config)
model.fit(X_train, y_train, X_val, y_val)
```

### 4. Physics-Informed Features
```python
from src.features.temporal import engineer_features, get_engineered_feature_names, FeatureConfig

config = FeatureConfig(rolling_windows=(3, 5, 10), include_greenwald_proxy=True)
df_eng = engineer_features(df, config)  # Computes per-discharge (no leakage)
feature_cols = get_engineered_feature_names(config)

# Key features:
# - greenwald_fraction: n/n_G, approaches 1 near density limit
# - d_density_dt: temporal derivatives
# - density_roll_std_*: fluctuation levels
```

### 5. Probability Calibration
```python
from src.uncertainty.calibration import TemperatureScaling, expected_calibration_error

temp_scaler = TemperatureScaling()
temp_scaler.fit(y_cal, logits_cal)
probs_calibrated = temp_scaler.calibrate(logits_test)

ece, _ = expected_calibration_error(y_test, probs_calibrated)  # Target: ECE < 0.05
```

## Quick Commands
```bash
pip install -r requirements.txt
./scripts/fetch_data.sh
python scripts/baseline.py                    # Basic LR baseline
python scripts/train_tcn.py --help            # TCN with alarm evaluation
python scripts/evaluate_alarm_policy.py --model both --save-figures
pytest tests/ -v                              # Run all tests
```

## Key APIs
```python
# Constants
FEATURE_COLUMNS = ["density", "elongation", "minor_radius", 
                   "plasma_current", "toroidal_B_field", "triangularity"]
TARGET_COLUMN = "density_limit_phase"

# Data loading
df = load_density_limit_data("data/raw/DL_DataFrame.h5")
train, cal, val, test = split_by_discharge_with_cal(df, random_state=42)

# Trade-off analysis
from src.evaluation.alarm_metrics import compute_tradeoff_curve, find_optimal_config
tradeoff_df = compute_tradeoff_curve(df, probs, thresholds=np.linspace(0.1, 0.9, 17))
optimal = find_optimal_config(tradeoff_df, min_event_recall=0.8)
```

## Testing
```bash
pytest tests/test_data_splitting.py   # Verify no discharge overlap
pytest tests/test_alarm_metrics.py    # Warning time sanity checks
pytest tests/test_features.py         # Feature engineering
```

## When Adding New Models
1. Create in `src/models/` with sklearn-compatible interface
2. Use `class_weight="balanced"` for imbalance
3. Evaluate with ALL metric types:
   - Sample: `evaluate_model()` → PR-AUC, ROC-AUC
   - Alarm: `compute_alarm_metrics()` → event recall, warning time, FAR
   - Calibration: `expected_calibration_error()` → ECE

## Conventions
- **random_state=42** everywhere for reproducibility
- **PR-AUC** is primary metric (class imbalance makes accuracy meaningless)
- Feature engineering respects discharge boundaries (no cross-discharge leakage)
- Results saved to `results/` directory
