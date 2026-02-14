# Model Card

## Model Details
- **Project**: tokamak-early-warning
- **Primary models**: Logistic Regression baseline and TCN (Temporal Convolutional Network)
- **Task**: binary classification of `density_limit_phase` for early warning of density limit events
- **Intended domain**: research and offline evaluation on MIT-PSFC open dataset

## Intended Use
- Support research on alarm-policy trade-offs (event recall, warning time, false alarms).
- Compare model classes under leakage-safe split protocols (`discharge_ID`-based).
- Not intended as a standalone real-time safety controller without additional validation.

## Data and Splits
- Dataset: MIT-PSFC Open Density Limit Database.
- Class imbalance: ~1-2% positive samples.
- **Critical anti-leakage constraint**: split by `discharge_ID`, never by individual samples.

## Training and Reproducibility
- Centralized runtime defaults in `tokamak_early_warning/config.py`.
- Fixed default seed: `42`.
- One-command deterministic workflow: `bash scripts/reproduce.sh`.
- Output artifacts written under `reports/<timestamp>/` with config and metrics JSON.

## Evaluation
- Sample-level: PR-AUC, ROC-AUC.
- Alarm-level: event recall, warning time, false alarms/discharge.
- Calibration: ECE and reliability curves.

## Risks and Limitations
- Dataset-specific behavior may not transfer to other machines/tokamaks without adaptation.
- Operational deployment requires additional controls, fail-safe logic, and human factors validation.
- Alarm threshold selection is context-dependent and should be tuned with domain experts.

## Ethical and Safety Considerations
- Predictions can influence operational decisions in high-energy systems.
- Use only as decision support with expert oversight.
- Validate latency and reliability constraints before any online use.
