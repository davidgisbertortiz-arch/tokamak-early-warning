# 🔬 Tokamak Early Warning System

[![Tests](https://github.com/davidgisbertortiz-arch/tokamak-early-warning/actions/workflows/tests.yml/badge.svg)](https://github.com/davidgisbertortiz-arch/tokamak-early-warning/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A production-grade ML pipeline for predicting tokamak density limit disruptions *before* they occur.**

<p align="center">
  <img src="results/discharge_demo.png" alt="Early Warning Demo" width="800"/>
</p>

---

## 🌟 Why This Matters

### The Problem: Tokamak Disruptions are Dangerous

A **tokamak** is a donut-shaped device that confines plasma at 100+ million °C to achieve nuclear fusion—our best shot at clean, limitless energy. But there's a catch: if you push the plasma density too high, it suddenly collapses in a **disruption**. These violent events:

- Dump gigajoules of energy onto reactor walls in milliseconds
- Generate runaway electrons that can melt components
- Can cost millions in repairs and months of downtime

The **density limit** is a fundamental boundary beyond which disruptions become inevitable. Crossing it without warning is catastrophic.

### The Solution: Early Warning

What if we could predict these events **before** they happen? Even a fraction of a second of warning allows operators to:
- Safely ramp down the plasma
- Inject impurities to radiate energy
- Avoid the worst damage

This project builds a machine learning system that watches plasma diagnostics in real-time and raises an alarm when trouble is coming.

---

## 📊 What Makes This Project Different

Most ML classification projects stop at **sample-level metrics** like accuracy or AUC. But for early warning, those metrics are misleading! What actually matters:

| Metric | What it Measures | Why it Matters |
|--------|-----------------|----------------|
| **Event Recall** | % of events detected *before* they occur | A missed event = disaster |
| **Warning Time** | Seconds between alarm and event | More time = safer response |
| **False Alarm Rate** | Spurious alarms per discharge | Too many = operators ignore them |

This system evaluates models the way operators would: *"Did you warn me in time?"*

<p align="center">
  <img src="results/model_comparison.png" alt="Model Comparison" width="700"/>
</p>

---

## 🧬 The Dataset

We use the [MIT-PSFC Open Density Limit Database](https://github.com/MIT-PSFC/open_density_limit_database):

| Property | Value |
|----------|-------|
| **Samples** | ~264,000 time points |
| **Discharges** | ~2,300 unique plasma experiments |
| **Positive Rate** | ~1-2% (extreme class imbalance!) |
| **Features** | Density, plasma current, B-field, shape parameters |

### What is `density_limit_phase`?

The label (`density_limit_phase = 1`) marks time points where the plasma is in the **danger zone**—approaching or at the density limit. The challenge: predict this *before* entering the phase.

<details>
<summary><b>📋 Feature Descriptions (click to expand)</b></summary>

| Column | Description | Physical Meaning |
|--------|-------------|-----------------|
| `discharge_ID` | Unique identifier | Groups time points from same experiment |
| `time` | Time in discharge (s) | Temporal coordinate |
| `density` | Plasma density | Higher = closer to limit |
| `plasma_current` | Plasma current (A) | Determines Greenwald limit |
| `toroidal_B_field` | Magnetic field (T) | Confinement strength |
| `minor_radius` | Plasma size (m) | Geometry parameter |
| `elongation` | Plasma shape | How "stretched" the plasma is |
| `triangularity` | Plasma shape | D-shaped vs circular |

</details>

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- ~2GB disk space for dataset
- Works on CPU (no GPU required)

### Installation & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset (~264k samples)
./scripts/fetch_data.sh

# 3. Train baseline & generate all figures (~5-10 min)
python scripts/make_figures.py

# 4. (Optional) Full TCN training with detailed evaluation
python scripts/train_tcn.py --epochs 30
```

### Explore Interactively

```bash
jupyter notebook notebooks/02_demo.ipynb
```

## 📁 Project Structure

```
tokamak-early-warning/
├── scripts/
│   ├── fetch_data.sh              # Download dataset
│   ├── baseline.py                # Baseline logistic regression
│   ├── train_tcn.py               # Train TCN model
│   ├── make_figures.py            # Generate all visualizations
│   └── evaluate_alarm_policy.py   # Full model comparison
├── src/
│   ├── data/dataset.py            # Data loading + splitting
│   ├── features/temporal.py       # Physics-informed features
│   ├── models/
│   │   ├── baseline_lr.py         # Logistic regression
│   │   └── tcn.py                 # Temporal Convolutional Network
│   ├── evaluation/
│   │   ├── metrics.py             # PR-AUC, ROC-AUC
│   │   └── alarm_metrics.py       # Event recall, warning time, FAR
│   └── uncertainty/calibration.py # Temperature scaling
├── notebooks/
│   └── 02_demo.ipynb              # Interactive demonstration
├── tests/                         # pytest test suite
├── results/                       # Generated figures
└── data/                          # Dataset (gitignored)
```

---

## 🔄 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                  │
│  ┌──────────┐    ┌─────────────────┐    ┌──────────────────────────┐   │
│  │   HDF5   │───▶│  Split by       │───▶│   Feature Engineering    │   │
│  │  Dataset │    │  discharge_ID   │    │  (physics + temporal)    │   │
│  └──────────┘    │  (no leakage!)  │    └──────────────────────────┘   │
│                  └─────────────────┘                │                   │
│                     │         │         │           │                   │
│                     ▼         ▼         ▼           │                   │
│               ┌─────────┐ ┌───────┐ ┌───────┐       │                   │
│               │ Train   │ │  Cal  │ │ Test  │       │                   │
│               │  60%    │ │  10%  │ │  15%  │       │                   │
│               └────┬────┘ └───┬───┘ └───┬───┘       │                   │
└────────────────────┼──────────┼─────────┼───────────┼───────────────────┘
                     │          │         │           │
                     ▼          ▼         ▼           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          MODEL PIPELINE                                  │
│  ┌────────────────────┐         ┌─────────────────────┐                 │
│  │   TCN Model        │         │   Calibration       │                 │
│  │   • Windowing      │────────▶│   • Temperature     │                 │
│  │   • 3 TCN blocks   │  probs  │     scaling         │                 │
│  │   • Dilated conv   │         │   • ECE < 0.05      │                 │
│  └────────────────────┘         └──────────┬──────────┘                 │
└─────────────────────────────────────────────┼───────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        ALARM EVALUATION                                  │
│  ┌─────────────────────┐    ┌─────────────────────────────────────────┐ │
│  │   Alarm Policy      │    │   Event-Level Metrics                   │ │
│  │   • Threshold       │───▶│   • Event Recall: detected before?      │ │
│  │   • Consecutive     │    │   • Warning Time: t_event - t_alarm     │ │
│  │   • EMA smoothing   │    │   • False Alarm Rate per discharge      │ │
│  └─────────────────────┘    └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📈 Results

### Model Comparison

| Model | PR-AUC | Event Recall | FA/Discharge | Median Warning |
|-------|--------|--------------|--------------|----------------|
| Baseline (LR) | ~0.45 | ~75% | ~8.5 | ~0.02s |
| **TCN** | **~0.52** | **~85%** | **~4.2** | **~0.04s** |

*PR-AUC (Precision-Recall Area Under Curve) is the primary metric due to extreme class imbalance.*

### Precision-Recall Curves

<p align="center">
  <img src="results/pr_curves.png" alt="PR Curves" width="600"/>
</p>

### Calibration Reliability

Well-calibrated probabilities are essential for setting alarm thresholds:

<p align="center">
  <img src="results/reliability_diagram.png" alt="Reliability Diagram" width="700"/>
</p>

### Event Recall vs False Alarm Trade-off

<p align="center">
  <img src="results/far_vs_recall.png" alt="FAR vs Recall" width="600"/>
</p>

### Warning Time Distribution

<p align="center">
  <img src="results/warning_time_distribution.png" alt="Warning Times" width="700"/>
</p>

---

## 🧪 ML Pitfalls We Avoid

### 1. Data Leakage via Random Splitting ⚠️

**Wrong:**
```python
train, test = train_test_split(df)  # LEAKS temporal info!
```

**Right:**
```python
from src.data.dataset import split_by_discharge
train, val, test = split_by_discharge(df, random_state=42)
```

### 2. Misleading Metrics with Class Imbalance

With only ~1% positive samples, a model predicting "always negative" gets 99% accuracy! We use **PR-AUC** as the primary metric.

### 3. Sample-Level vs Event-Level Evaluation

A model might have great AUC but still miss 50% of events or trigger alarms too late. We compute **alarm-level metrics** that capture operational utility.

---

## 🧠 Model Interpretation

The model learns physically meaningful patterns:

| Pattern | What it Means |
|---------|---------------|
| **Greenwald fraction → 1.0** | Strongest predictor. n/n_G > 0.8 = danger |
| **Rising density derivatives** | Rapid increases precede events |
| **High fluctuation levels** | Instability signature |
| **Shape-current interactions** | Certain geometries more prone |

---

## 🔮 Roadmap

- [ ] Transfer learning to other tokamaks (JET, DIII-D)
- [ ] Add soft X-ray and Mirnov coil signals
- [ ] Survival analysis: predict time-to-event
- [ ] ONNX export for real-time deployment
- [ ] SHAP per-prediction explanations

---

## 🧪 Running Tests

```bash
pytest tests/ -v                    # All tests
pytest tests/test_data_splitting.py # Verify no leakage
pytest tests/ --cov=src             # With coverage
```

---

## 📚 References

1. [MIT-PSFC Open Density Limit Database](https://github.com/MIT-PSFC/open_density_limit_database)
2. Greenwald, M. (2002). "Density limits in toroidal plasmas." *PPCF* 44, R27.
3. Bai et al. (2018). "An Empirical Evaluation of Generic Convolutional and Recurrent Networks."
4. Guo et al. (2017). "On Calibration of Modern Neural Networks." *ICML*.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Built for fusion energy research</i>
</p>
