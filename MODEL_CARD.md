# Model Card — Tokamak Early Warning System

## Model Details

| Field | Value |
|-------|-------|
| **Models** | Logistic Regression (baseline) · Temporal Convolutional Network (TCN) |
| **Task** | Binary classification of `density_limit_phase` for early warning of tokamak density-limit disruptions |
| **Framework** | scikit-learn (LR) · PyTorch (TCN) |
| **License** | MIT |
| **Version** | 1.0 |
| **Contact** | See repository issues |

Both models expose an sklearn-compatible `fit` / `predict_proba` interface and use `class_weight="balanced"` or equivalent `pos_weight` to handle class imbalance.

---

## Dataset

**Source:** [MIT-PSFC Open Density Limit Database](https://github.com/MIT-PSFC/open_density_limit_database)

| Property | Value |
|----------|-------|
| Samples | ~264 000 time points |
| Discharges | ~2 300 unique plasma experiments |
| Positive rate | ~1–2 % |
| Features | `density`, `plasma_current`, `toroidal_B_field`, `minor_radius`, `elongation`, `triangularity` |
| Label | `density_limit_phase` (1 = plasma in or approaching the density-limit regime) |

### Engineered Features

`src/features/temporal.py` adds physics-informed features per discharge: rolling statistics (windows 3/5/10), temporal derivatives, and a Greenwald-fraction proxy (`density / plasma_current`).

---

## Train / Val / Test Split Policy

**Critical constraint — no data leakage.** Time points within a single discharge are temporally correlated, so we always split by `discharge_ID`, never by individual samples.

| Split | Fraction | Purpose |
|-------|----------|---------|
| Train | 60 % | Model fitting |
| Calibration | 10 % | Post-hoc probability calibration |
| Validation | 15 % | Hyper-parameter tuning / early stopping |
| Test | 15 % | Final held-out evaluation |

The split is implemented in `src/data/dataset.split_by_discharge_with_cal` with a fixed random seed (`42`) for reproducibility. A three-way variant (`split_by_discharge`) is also available.

---

## Evaluation Metrics

### Sample-level

| Metric | Why |
|--------|-----|
| **PR-AUC** (primary) | Appropriate for extreme class imbalance (~1 % positive) where accuracy is meaningless |
| **ROC-AUC** | Complements PR-AUC; measures discrimination across all thresholds |

### Alarm-level (operational)

| Metric | Definition |
|--------|------------|
| **Event Recall** | Fraction of density-limit events for which an alarm fired *before* event onset |
| **Warning Time** | Seconds between alarm trigger and event start |
| **False Alarm Rate** | Spurious alarms per discharge |

These are computed by `src/evaluation/alarm_metrics.py` with an `AlarmConfig` that controls threshold, consecutive-sample count, and optional EMA smoothing.

### Typical Results

| Model | PR-AUC | Event Recall | FA / Discharge | Median Warning |
|-------|--------|--------------|----------------|----------------|
| Baseline (LR) | ~0.45 | ~75 % | ~8.5 | ~0.02 s |
| **TCN** | **~0.52** | **~85 %** | **~4.2** | **~0.04 s** |

---

## Calibration Approach

Well-calibrated probabilities are essential for setting operationally meaningful alarm thresholds. Two post-hoc methods are available (`src/uncertainty/calibration.py`):

1. **Temperature Scaling** — learns a single scalar *T* that divides the model's logits before the sigmoid, minimising NLL on the calibration split. Used by default for the TCN.
2. **Isotonic Regression / Platt Scaling** — non-parametric and parametric alternatives for the LR baseline.

Calibration quality is measured by **Expected Calibration Error (ECE)**; the target is ECE < 0.05.

Additionally, `src/uncertainty/conformal.py` provides conformal prediction sets for distribution-free coverage guarantees.

---

## Limitations

* **Single-machine scope** — trained and evaluated exclusively on the MIT-PSFC C-Mod dataset. Performance may not transfer to other tokamaks (JET, DIII-D, ITER) without domain adaptation.
* **Short warning times** — median warnings are in the tens-of-milliseconds range, which may be insufficient for some actuator response times.
* **Label quality** — `density_limit_phase` is derived from post-hoc analysis; real-time ground truth may differ.
* **CPU-only default** — training is feasible on CPU but will be slower for larger models or datasets.

## Ethical and Safety Considerations

* **Not a standalone safety system.** Predictions must be used as *decision support* alongside human expert oversight and engineering safeguards.
* **False negatives carry high cost.** A missed event can lead to physical damage; alarm-threshold selection should be performed jointly with domain specialists.
* **False alarms erode trust.** Excessive false alarms in an operational context may cause operators to ignore warnings ("alarm fatigue").
* **Validate before deployment.** Any online use requires additional latency, reliability, and fail-safe validation beyond what this research pipeline provides.

