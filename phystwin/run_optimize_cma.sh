#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="phystwin/data"
CASE_NAME="double_lift_cloth_1"
TRAIN_FRAME="116"
CONFIG="phystwin/config/cloth.yaml"
# Enable the ViewerGL monitor to pop up a simulator window.
# Set MONITOR_FRAMES to limit how many frames are shown per evaluation (0 = use train_frame).
MONITOR_ARGS="--monitor --monitor-frames 116 --monitor-every 1"
EXTRA_ARGS="${MONITOR_ARGS}"

exec uv run --extra examples --with cma --with scipy python phystwin/optimize/optimize_cma.py \
  --base_path "$BASE_PATH" \
  --case_name "$CASE_NAME" \
  --train_frame "$TRAIN_FRAME" \
  --config "$CONFIG" \
  ${EXTRA_ARGS}
