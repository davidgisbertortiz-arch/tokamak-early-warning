# tokamak-early-warning

Early-warning ML pipeline for predicting tokamak density limit events using the MIT-PSFC Open Density Limit dataset.

## Quick Start (Codespaces)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
./scripts/fetch_data.sh

# 3. Run baseline model
python scripts/baseline.py
```

## Dataset

We use the [MIT-PSFC Open Density Limit Database](https://github.com/MIT-PSFC/open_density_limit_database):

| Column | Description |
|--------|-------------|
| `discharge_ID` | Unique identifier for each plasma discharge |
| `time` | Time point within the discharge (seconds) |
| `density_limit_phase` | Binary label: 1 = near/at density limit, 0 = normal |
| `density` | Plasma density |
| `elongation` | Plasma elongation |
| `minor_radius` | Plasma minor radius |
| `plasma_current` | Plasma current |
| `toroidal_B_field` | Toroidal magnetic field |
| `triangularity` | Plasma triangularity |

**Dataset Stats**: ~264k samples, ~1-2% positive class (severe imbalance)

## Why Split by Discharge ID?

⚠️ **Critical for avoiding data leakage**

Time series data from the same discharge are highly correlated. If we randomly split samples, consecutive time points from the same discharge could end up in both train and test sets, leading to:
- Artificially inflated metrics
- Models that memorize discharge patterns instead of learning generalizable features

**Solution**: We split by `discharge_ID`, ensuring all time points from a discharge stay in the same split.

## Metrics

Due to severe class imbalance (~1-2% positive):

- **Primary**: PR-AUC (Precision-Recall AUC) - more informative for imbalanced data
- **Secondary**: ROC-AUC - complementary view

## Project Structure

```
tokamak-early-warning/
├── scripts/
│   ├── fetch_data.sh              # Download dataset
│   ├── baseline.py                # Train & evaluate baseline model
│   └── make_early_warning_labels.py  # Create predictive labels
├── src/
│   ├── data/
│   │   └── dataset.py             # Data loading & splitting utilities
│   ├── models/
│   │   └── baseline_lr.py         # Baseline logistic regression
│   └── evaluation/
│       └── metrics.py             # Evaluation utilities
├── data/                          # Downloaded data (gitignored)
│   └── .gitkeep
├── requirements.txt
└── README.md
```

## Early Warning Labels (Optional)

Transform the `density_limit_phase` label into a predictive "early warning" label:

```bash
# Create labels that predict events 0.1s ahead
python scripts/make_early_warning_labels.py --dt 0.1
```

This creates `data/processed/DL_early_warning.h5` with an `early_warning` column that is 1 for samples within `dt` seconds *before* an event.

## Development

### Adding New Models

1. Create model in `src/models/`
2. Follow the pattern in `baseline_lr.py`: function to create pipeline, function to train
3. Update `scripts/baseline.py` or create new training script

### Key Design Decisions

- **No data in repo**: Data is downloaded on-demand via `fetch_data.sh`
- **Reproducibility**: Fixed random seeds (42) throughout
- **Class imbalance**: Use `class_weight="balanced"` in classifiers
- **Leakage prevention**: Always split by `discharge_ID`, never by individual samples

## License

MIT
