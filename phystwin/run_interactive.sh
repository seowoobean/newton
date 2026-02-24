#!/usr/bin/env bash
set -euo pipefail

PKL="phystwin/experiments/double_lift_cloth_1/train/model.pkl"
URDF="newton/examples/ai_worker/ai_worker.urdf"

# Simulation integration / timing
ARGS_SIM="--substeps 10"

# Spring-vs-robot contact response tuning
ARGS_CONTACT="--spring-soft-contact-ke 1.0e5 --spring-soft-contact-kd 5.0e2 --spring-soft-contact-kf 3.0e3 --spring-soft-contact-margin 0.03"

# Robot joint hold gains
ARGS_ROBOT_PD="--robot-joint-ke 1.0e3 --robot-joint-kd 1.0e1"

# Scene placement / rendering
ARGS_SCENE="--urdf-offset -0.6 0.0 0.0 --visual-z-offset 0.0 --mesh-show-points"

# Interactive controls
ARGS_INTERACTIVE="--enable-controller-drag"

# Compose all optional args
EXTRA_ARGS="${ARGS_SIM} ${ARGS_CONTACT} ${ARGS_ROBOT_PD} ${ARGS_SCENE} ${ARGS_INTERACTIVE}"

AI_WORKER_ROOT="/home/roro/git/ai_worker"
FFW_DESC="${AI_WORKER_ROOT}/ffw_description"
if [[ -d "${FFW_DESC}" ]]; then
  export ROS_PACKAGE_PATH="${FFW_DESC}${ROS_PACKAGE_PATH:+:${ROS_PACKAGE_PATH}}"
  export GZ_SIM_RESOURCE_PATH="${FFW_DESC}${GZ_SIM_RESOURCE_PATH:+:${GZ_SIM_RESOURCE_PATH}}"
fi

ARGS=()
if [[ -n "${PKL}" ]]; then
  ARGS+=(--pkl "$PKL")
fi
if [[ -n "${URDF}" ]]; then
  ARGS+=(--urdf "$URDF")
fi

exec uv run python phystwin/sim/spring_mass_interactive.py \
  "${ARGS[@]}" \
  ${EXTRA_ARGS}
