# Dataset Documentation

## Source
- **Name**: MIT-PSFC Open Density Limit Database
- **URL**: https://github.com/MIT-PSFC/open_density_limit_database
- **Primary file used here**: `data/raw/DL_DataFrame.h5`

## Schema Used in This Repository
- Grouping key: `discharge_ID`
- Temporal column: `time`
- Base features:
  - `density`
  - `elongation`
  - `minor_radius`
  - `plasma_current`
  - `toroidal_B_field`
  - `triangularity`
- Target label: `density_limit_phase`

## Data Characteristics
- ~264k samples across ~2.3k discharges.
- Positive class rate around 1-2%.
- Time points inside one discharge are strongly correlated.

## Split Policy (Leakage Prevention)
- **Mandatory**: split by `discharge_ID`.
- Implemented in `src/data/dataset.py` with:
  - `split_by_discharge`
  - `split_by_discharge_with_cal`

## Preprocessing and Feature Engineering
- Temporal derivatives and rolling windows in `src/features/temporal.py`.
- Optional Greenwald proxy and interaction features.
- All temporal operations respect discharge boundaries.

## Reproducibility Notes
- Use fixed seeds via `tokamak_early_warning/config.py`.
- Use `bash scripts/reproduce.sh` for end-to-end deterministic runs that archive results in `reports/`.

## Known Limitations
- Potentially limited representativeness outside the covered operating regimes.
- Label quality and event boundary definitions may vary by discharge context.
