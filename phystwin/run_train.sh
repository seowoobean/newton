#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="phystwin/data"
CASE_NAME="double_lift_cloth_1"
TRAIN_FRAME="116"
CONFIG="phystwin/config/cloth.yaml"
EXTRA_ARGS="--max-iter 200 --monitor --monitor-every 1"

exec uv run python phystwin/train/train_warp.py \
  --base_path "$BASE_PATH" \
  --case_name "$CASE_NAME" \
  --train_frame "$TRAIN_FRAME" \
  --config "$CONFIG" \
  ${EXTRA_ARGS}
