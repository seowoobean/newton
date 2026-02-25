#!/usr/bin/env bash
set -euo pipefail

CASE_NAME="double_lift_cloth_1"
PKL="phystwin/experiments/${CASE_NAME}/train/model.pkl"
URDF="newton/examples/ai_worker/ai_worker.urdf"

# Simulation integration / timing
ARGS_SIM="--substeps 150"

# Spring-vs-robot contact response tuning
ARGS_CONTACT="--spring-soft-contact-ke 9.0e4 --spring-soft-contact-kd 1.0e3 --spring-soft-contact-kf 5.0e7 --spring-soft-contact-margin 0.01"

# Robot joint hold gains
ARGS_ROBOT_PD="--robot-joint-ke 1.0e6 --robot-joint-kd 1.0e3"

# Robot initial joint pose preset
ARGS_ROBOT_POSE="--robot-pose-preset forward_reach"

# Single-arm joint overrides (only active arm is modified; the other arm stays as-is)
ARGS_ARM="--active-arm right --lift-joint 0.0 --arm-j1 -1.570796 --arm-j2 -0.174533 --arm-j3 -0.872665 --arm-j4 0.552665 --arm-j5 0.523599 --arm-j6 1.047198 --arm-j7 1.047198 --print-arm-joint-ranges"

# Right gripper auto-close sequence
ARGS_GRIPPER="--auto-close-right-gripper --gripper-close-start-time 0.5 --gripper-close-duration 2.0 --gripper-close-target 0.87 --auto-sweep-arm-j1-after-grip --arm-j1-sweep-target -3.0 --arm-j1-sweep-duration 15.0"

# Spring-mass placement
ARGS_SPRING="--spring-offset -0.095 0.08 0.0 --particle-mass-override 1.0"

# Scene placement / rendering
ARGS_SCENE="--urdf-offset -0.6 0.0 0.0 --visual-z-offset 0.0 --mesh-show-points"

# Table placement / size
ARGS_TABLE="--table-center-x 0.15 --table-center-y 0.0 --table-center-z 0.6 --table-half-x 0.45 --table-half-y 0.35 --table-half-z 0.6 --table-cloth-clearance 0.04"

# Interactive controls
ARGS_INTERACTIVE="--enable-controller-drag"

# Compose all optional args
EXTRA_ARGS="${ARGS_SIM} ${ARGS_CONTACT} ${ARGS_ROBOT_PD} ${ARGS_ROBOT_POSE} ${ARGS_ARM} ${ARGS_GRIPPER} ${ARGS_SPRING} ${ARGS_SCENE} ${ARGS_TABLE} ${ARGS_INTERACTIVE}"

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
