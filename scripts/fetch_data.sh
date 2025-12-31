#!/usr/bin/env bash
# Script to download MIT-PSFC Open Density Limit dataset
# Idempotent: only downloads if files don't exist

set -euo pipefail

mkdir -p data/raw

H5="data/raw/DL_DataFrame.h5"
CSV="data/raw/DL_DataFrame.csv"   # optional

URL_H5="https://raw.githubusercontent.com/MIT-PSFC/open_density_limit_database/main/data/DL_DataFrame.h5"
URL_CSV="https://raw.githubusercontent.com/MIT-PSFC/open_density_limit_database/main/data/DL_DataFrame.csv"

if [ ! -f "$H5" ]; then
  echo "Downloading H5..."
  curl -L "$URL_H5" -o "$H5"
else
  echo "H5 already exists: $H5"
fi

# Optional: CSV download (comment out to save time/space)
if [ ! -f "$CSV" ]; then
  echo "Downloading CSV..."
  curl -L "$URL_CSV" -o "$CSV"
else
  echo "CSV already exists: $CSV"
fi

echo "Data download complete!"
ls -lh data/raw/
