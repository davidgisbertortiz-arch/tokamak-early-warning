# Copilot Instructions for tokamak-early-warning

## Project Overview
ML pipeline for early warning prediction of tokamak density limit events. Uses the MIT-PSFC Open Density Limit dataset (~264k samples, ~1-2% positive class). Includes uncertainty quantification via conformal prediction and event-based evaluation.

## Architecture
```
src/
├── data/dataset.py         # Data loading, discharge-based splitting
├── models/baseline_lr.py   # StandardScaler + LogisticRegression pipeline
├── evaluation/
│   ├── metrics.py          # PR-AUC, ROC-AUC, classification_report
│   └── event_metrics.py    # Event recall, lead time, false alarm rate
├── uncertainty/
│   ├── conformal.py        # Split conformal prediction with coverage
│   └── calibration.py      # Probability calibration, ECE
└── alerting/
    └── policy.py           # GREEN/YELLOW/RED alert state machine
scripts/
├── fetch_data.sh           # Download dataset (idempotent)
├── baseline.py             # Basic training script
├── evaluate_uncertainty.py # Full evaluation with conformal + events
└── make_early_warning_labels.py  # Transform labels for prediction
```

## Critical Patterns

### 1. Data Leakage Prevention (MANDATORY)
**ALWAYS split by `discharge_ID`, never by individual samples.** Time points from the same discharge are correlated.

```python
# CORRECT - standard 3-way split
from src.data.dataset import split_by_discharge
train_df, val_df, test_df = split_by_discharge(df, random_state=42)

# CORRECT - 4-way split with calibration set for conformal prediction
from src.data.dataset import split_by_discharge_with_cal
train_df, cal_df, val_df, test_df = split_by_discharge_with_cal(df, random_state=42)

# WRONG - causes leakage
from sklearn.model_selection import train_test_split
train, test = train_test_split(df)  # NEVER DO THIS
```

### 2. Class Imbalance (~1-2% positive)
- **Primary metric**: PR-AUC (`average_precision_score`) - always report this
- **Secondary metric**: ROC-AUC  
- Always use `class_weight="balanced"` in classifiers
- Never use accuracy as a metric

### 3. Uncertainty Quantification (Conformal Prediction)
```python
from src.uncertainty.conformal import ConformalClassifier

conformal = ConformalClassifier(model, alpha=0.10)  # 90% coverage target
conformal.calibrate(X_cal, y_cal)  # MUST be separate from training data
result = conformal.predict(X_test)
# result.prediction_sets: {0}, {1}, or {0,1} (uncertain)

# Evaluate per-discharge coverage (critical for temporal data)
metrics = conformal.evaluate_coverage(X_test, y_test, discharge_ids=test_df["discharge_ID"])
```

### 4. Event-Based Evaluation
Standard ML metrics don't capture operational utility. Use event-based metrics:
```python
from src.evaluation.event_metrics import compute_event_metrics

metrics = compute_event_metrics(df, predictions)
# metrics.event_recall: % of events detected before they occur
# metrics.lead_times: how early each event was detected
# metrics.false_alarms_per_hour: operational false alarm rate
```

### 5. Alert Policy
Convert predictions to actionable alerts via state machine:
```python
from src.alerting.policy import AlertPolicy

policy = AlertPolicy(
    prob_yellow_threshold=0.2,  # Elevate to YELLOW
    prob_red_threshold=0.5,     # Escalate to RED
    n_consecutive_for_red=1,    # Consecutive {1} for RED
)
# Acceptance criteria: event_recall >= 80%, false_alarms <= 5/discharge
```

## Quick Commands
```bash
pip install -r requirements.txt        # pandas, scikit-learn, tables
./scripts/fetch_data.sh                # Download ~264k samples (idempotent)
python scripts/baseline.py             # Basic train/eval with LogisticRegression
python scripts/evaluate_uncertainty.py # Full uncertainty + event eval
python scripts/make_early_warning_labels.py --dt 0.1  # Create predictive labels
```

## Key APIs
```python
# Data
FEATURE_COLUMNS = ["density", "elongation", "minor_radius", 
                   "plasma_current", "toroidal_B_field", "triangularity"]
TARGET_COLUMN = "density_limit_phase"
df = load_density_limit_data("data/raw/DL_DataFrame.h5")

# Split (use _with_cal for conformal prediction)
train, cal, val, test = split_by_discharge_with_cal(df, random_state=42)

# Features
X, y = get_features_target(df)
```

## When Adding New Models
1. Create in `src/models/` with `create_*_pipeline()` + `train_*()` functions
2. Use sklearn Pipeline: `StandardScaler` → classifier
3. Evaluate with all three metric types:
   - Standard: `evaluate_model()` → PR-AUC, ROC-AUC
   - Event-based: `compute_event_metrics()` → event recall, lead time
   - Uncertainty: `ConformalClassifier.evaluate_coverage()` → per-discharge coverage

## Project-Specific Conventions

### Import Pattern
Scripts add `src/` to path for imports:
```python
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import load_density_limit_data
```

### Reproducibility
Always use `random_state=42` for consistency across splits and models.

### Pipeline Pattern
All models follow the sklearn Pipeline pattern:
```python
Pipeline([
    ("scaler", StandardScaler()),  # Always normalize features first
    ("classifier", LogisticRegression(class_weight="balanced", random_state=42))
])
```

### Data Format
- Dataset is HDF5 format (`.h5` files) using pandas `read_hdf()` / `to_hdf()`
- Sorting by `["discharge_ID", "time"]` is standard after loading
- CSV format available but HDF5 preferred for performance

## Data Files (gitignored except .gitkeep)
- `data/raw/DL_DataFrame.h5` - Main dataset (HDF5, ~264k rows)
- `data/raw/DL_DataFrame.csv` - Alternative CSV format (slower to load)
- `data/processed/` - Transformed data (e.g., early warning labels from make_early_warning_labels.py)
