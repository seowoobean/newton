#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="phystwin/data"
CASE_NAME="double_lift_cloth_1"
CONFIG="phystwin/config/cloth.yaml"

SRC_PKL="${BASE_PATH}/${CASE_NAME}/final_data.pkl"
OUT_PKL="phystwin/experiments/${CASE_NAME}/train/model.pkl"
OPTIMAL_PARAMS="phystwin/experiments_optimization/${CASE_NAME}/optimal_params.pkl"
BEST_PARAMS="phystwin/experiments/${CASE_NAME}/train/best_params.pkl"

# Optional: add "--apply-transform" if you want to bake scale/reverse_z/z_offset.
EXTRA_ARGS=""

exec uv run python phystwin/sim/export_model.py \
  --pkl "$SRC_PKL" \
  --config "$CONFIG" \
  --out-pkl "$OUT_PKL" \
  --optimal-params "$OPTIMAL_PARAMS" \
  --best-params "$BEST_PARAMS" \
  ${EXTRA_ARGS}
