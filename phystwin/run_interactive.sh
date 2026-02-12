#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="phystwin/data"
CASE_NAME="double_lift_cloth_1"
CONFIG="phystwin/config/cloth.yaml"
PARAMS="phystwin/experiments/double_lift_cloth_1/train/best_params.pkl"
EXTRA_ARGS="--enable-controller-drag --controller-mass 1.0 --controller-drag-stiffness 2.0e4 --controller-drag-damping 1.0e2"

PKL_PATH="${BASE_PATH}/${CASE_NAME}/final_data.pkl"

ARGS=(
  --pkl "$PKL_PATH"
  --config "$CONFIG"
)

if [[ -n "${PARAMS}" ]]; then
  ARGS+=(--params "$PARAMS")
fi

exec uv run python phystwin/sim/spring_mass_interactive.py \
  "${ARGS[@]}" \
  ${EXTRA_ARGS}
