#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${SEED:-42}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%SZ)"
RUN_DIR="reports/${TIMESTAMP}"
TRAIN_DIR="${RUN_DIR}/train"
EVAL_DIR="${RUN_DIR}/evaluation"

mkdir -p "$TRAIN_DIR" "$EVAL_DIR"

echo "[reproduce] Installing dependencies"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

echo "[reproduce] Fetching dataset (idempotent)"
./scripts/fetch_data.sh

echo "[reproduce] Training TCN with fixed seed=${SEED}"
python scripts/train_tcn.py --seed "$SEED" --output-dir "$TRAIN_DIR"

echo "[reproduce] Evaluating alarm policy with fixed seed=${SEED}"
python scripts/evaluate_alarm_policy.py --model both --seed "$SEED" --output-dir "$EVAL_DIR" --save-figures

cat > "${RUN_DIR}/config.json" <<EOF
{
  "timestamp": "${TIMESTAMP}",
  "seed": ${SEED},
  "train_output": "${TRAIN_DIR}",
  "evaluation_output": "${EVAL_DIR}",
  "metrics": {
    "train": "${TRAIN_DIR}/tcn_results.json",
    "evaluation": "${EVAL_DIR}/alarm_evaluation_results.json"
  }
}
EOF

echo "[reproduce] Completed run in ${RUN_DIR}"
