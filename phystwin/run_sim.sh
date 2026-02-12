#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="phystwin/data"
CASE_NAME="double_lift_cloth_1"
CONFIG="phystwin/config/cloth.yaml"
EXTRA_ARGS=""

PKL_PATH="${BASE_PATH}/${CASE_NAME}/final_data.pkl"

exec uv run python phystwin/sim/spring_mass_from_pkl.py \
  --pkl "$PKL_PATH" \
  --config "$CONFIG" \
  ${EXTRA_ARGS}
