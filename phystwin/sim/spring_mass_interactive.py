"""Simulate a robot and a spring-mass system with separate solvers."""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from phystwin.mapping.pkl_mapping import SpringMassPKL, SpringMassPKLPair, load_pkl, map_pkl_to_newton
from phystwin.sim.mesh_utils import triangulate_points


LOGGER = logging.getLogger("phystwin.robot_spring_dual")

TABLE_CENTER_X = -0.25
TABLE_CENTER_Y = 0.0
TABLE_CENTER_Z = 0.75
TABLE_HALF_X = 0.45
TABLE_HALF_Y = 0.35
TABLE_HALF_Z = 0.25
TABLE_CLOTH_CLEARANCE = 0.03

INTERACTIVE_PARAM_KEYS = {
    "fps",
    "FPS",
    "substeps",
    "num_substeps",
    "scale",
    "z_offset",
    "reverse_z",
    "particle_radius",
    "mass",
    "particle_mass",
    "spring_ke",
    "init_spring_Y",
    "spring_kd",
    "dashpot_damping",
    "k_neighbors",
    "object_max_neighbours",
    "use_controllers",
    "controller_k",
    "controller_ke",
    "controller_kd",
    "controller_mass",
    "filter_visibility",
    "filter_motion_valid",
    "spring_neighbor_mode",
    "object_radius",
    "spring_stiffness",
    "spring_damping",
}


@wp.kernel
def apply_controller_force_kernel(
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    controller_indices: wp.array(dtype=int),
    controller_targets: wp.array(dtype=wp.vec3),
    stiffness: float,
    damping: float,
):
    tid = wp.tid()
    idx = controller_indices[tid]
    target = controller_targets[tid]
    x = q[idx]
    v = qd[idx]
    f[idx] += stiffness * (target - x) - damping * v


@wp.kernel
def set_controller_targets_kernel(
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    controller_indices: wp.array(dtype=int),
    controller_targets: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    idx = controller_indices[tid]
    q[idx] = controller_targets[tid]
    qd[idx] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def gather_patch_contact_candidates_kernel(
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    shape_patch_mask: wp.array(dtype=wp.int32),
    max_candidates: int,
    candidate_count: wp.array(dtype=int),
    candidate_particle: wp.array(dtype=int),
    candidate_shape: wp.array(dtype=int),
    candidate_body_pos: wp.array(dtype=wp.vec3),
    candidate_normal: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    count = contact_count[0]
    if tid >= count:
        return

    particle_idx = contact_particle[tid]
    shape_idx = contact_shape[tid]
    if particle_idx < 0 or shape_idx < 0:
        return
    if shape_patch_mask[shape_idx] == 0:
        return

    out_idx = wp.atomic_add(candidate_count, 0, 1)
    if out_idx >= max_candidates:
        return
    candidate_particle[out_idx] = particle_idx
    candidate_shape[out_idx] = shape_idx
    candidate_body_pos[out_idx] = contact_body_pos[tid]
    candidate_normal[out_idx] = contact_normal[tid]


@wp.kernel
def filter_patch_contact_candidates_by_touch_kernel(
    candidate_particle: wp.array(dtype=int),
    candidate_shape: wp.array(dtype=int),
    candidate_body_pos: wp.array(dtype=wp.vec3),
    candidate_normal: wp.array(dtype=wp.vec3),
    shape_body_indices: wp.array(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    particle_radius: wp.array(dtype=float),
    max_gap: float,
    max_filtered: int,
    filtered_count: wp.array(dtype=wp.int32),
    filtered_particle: wp.array(dtype=int),
    filtered_shape: wp.array(dtype=int),
    filtered_normal: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    particle_idx = candidate_particle[tid]
    shape_idx = candidate_shape[tid]
    if particle_idx < 0 or shape_idx < 0:
        return
    if particle_idx >= particle_q.shape[0] or particle_idx >= particle_radius.shape[0]:
        return
    if shape_idx >= shape_body_indices.shape[0]:
        return

    body_idx = shape_body_indices[shape_idx]
    if body_idx < 0 or body_idx >= body_q.shape[0]:
        return

    n_world = candidate_normal[tid]
    n_len = wp.length(n_world)
    if n_len <= 1.0e-8:
        return
    n_world = n_world / n_len

    X_wb = body_q[body_idx]
    particle_pos = particle_q[particle_idx]
    contact_pos_world = wp.transform_point(X_wb, candidate_body_pos[tid])
    center_gap = wp.dot(particle_pos - contact_pos_world, n_world)
    surface_gap = center_gap - particle_radius[particle_idx]
    if surface_gap > max_gap:
        return

    out_idx = wp.atomic_add(filtered_count, 0, 1)
    if out_idx >= max_filtered:
        return
    filtered_particle[out_idx] = particle_idx
    filtered_shape[out_idx] = shape_idx
    filtered_normal[out_idx] = n_world


@wp.kernel
def apply_contact_patch_force_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    patch_particle_indices: wp.array(dtype=int),
    patch_body_indices: wp.array(dtype=int),
    patch_anchor_local: wp.array(dtype=wp.vec3),
    patch_normal_local: wp.array(dtype=wp.vec3),
    normal_ke: float,
    normal_kd: float,
    tangent_ke: float,
    tangent_kd: float,
    mu: float,
    max_force: float,
):
    tid = wp.tid()
    particle_idx = patch_particle_indices[tid]
    body_idx = patch_body_indices[tid]
    if particle_idx < 0 or body_idx < 0:
        return

    x = particle_q[particle_idx]
    v = particle_qd[particle_idx]

    X_wb = body_q[body_idx]
    body_v_s = body_qd[body_idx]
    body_v = wp.spatial_top(body_v_s)
    body_w = wp.spatial_bottom(body_v_s)

    body_origin = wp.transform_point(X_wb, wp.vec3(0.0, 0.0, 0.0))
    x_target = wp.transform_point(X_wb, patch_anchor_local[tid])
    v_target = body_v + wp.cross(body_w, x_target - body_origin)

    n_world = wp.transform_vector(X_wb, patch_normal_local[tid])
    n_len = wp.length(n_world)
    if n_len > 1.0e-8:
        n_world = n_world / n_len
    else:
        n_world = wp.vec3(0.0, 0.0, 1.0)

    err = x_target - x
    rel_v = v_target - v
    err_n = wp.dot(err, n_world)
    vel_n = wp.dot(rel_v, n_world)
    err_t = err - n_world * err_n
    vel_t = rel_v - n_world * vel_n

    f_n = n_world * (normal_ke * err_n + normal_kd * vel_n)
    f_t = tangent_ke * err_t + tangent_kd * vel_t
    f_t_len = wp.length(f_t)
    max_tangent = mu * (wp.length(f_n) + 1.0e-6)
    if f_t_len > max_tangent and f_t_len > 1.0e-8:
        f_t = f_t * (max_tangent / f_t_len)

    f_total = f_n + f_t
    f_len = wp.length(f_total)
    if max_force > 0.0 and f_len > max_force and f_len > 1.0e-8:
        f_total = f_total * (max_force / f_len)

    wp.atomic_add(particle_f, particle_idx, f_total)


@wp.kernel
def eval_patch_break_mask_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    patch_particle_indices: wp.array(dtype=int),
    patch_body_indices: wp.array(dtype=int),
    patch_anchor_local: wp.array(dtype=wp.vec3),
    break_distance: float,
    keep_mask: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    particle_idx = patch_particle_indices[tid]
    body_idx = patch_body_indices[tid]
    if particle_idx < 0 or body_idx < 0:
        keep_mask[tid] = 0
        return

    x = particle_q[particle_idx]
    X_wb = body_q[body_idx]
    x_target = wp.transform_point(X_wb, patch_anchor_local[tid])
    dist = wp.length(x_target - x)
    if dist <= break_distance:
        keep_mask[tid] = 1
    else:
        keep_mask[tid] = 0


@wp.kernel
def copy_indexed_transform_kernel(
    src: wp.array(dtype=wp.transform),
    dst: wp.array(dtype=wp.transform),
    src_indices: wp.array(dtype=wp.int32),
    dst_indices: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    dst_idx = dst_indices[tid]
    src_idx = src_indices[tid]
    dst[dst_idx] = src[src_idx]


@wp.kernel
def copy_indexed_spatial_vector_kernel(
    src: wp.array(dtype=wp.spatial_vector),
    dst: wp.array(dtype=wp.spatial_vector),
    src_indices: wp.array(dtype=wp.int32),
    dst_indices: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    dst_idx = dst_indices[tid]
    src_idx = src_indices[tid]
    dst[dst_idx] = src[src_idx]


def _apply_scale(points: np.ndarray, scale: float, reverse_z: bool) -> np.ndarray:
    scaled = points.astype(np.float32) * scale
    if reverse_z and scaled.size:
        scaled[..., 2] *= -1.0
    return scaled


def _compute_z_shift(points: np.ndarray, z_offset: float) -> float:
    if points.size == 0:
        return z_offset
    min_z = float(points[:, 2].min())
    if min_z <= 0.0:
        return -min_z + z_offset
    return z_offset


def _object_mask(data: SpringMassPKL, frame: int, filter_visibility: bool, filter_motion_valid: bool) -> np.ndarray:
    mask = np.ones(data.object_points.shape[1], dtype=bool)
    if filter_visibility and data.object_visibilities.size:
        mask &= data.object_visibilities[frame].astype(bool)
    if filter_motion_valid and data.object_motions_valid.size:
        mask &= data.object_motions_valid[frame].astype(bool)
    return mask


def _build_solver(
    name: str,
    model: newton.Model,
    mujoco_nconmax: int | None = None,
    mujoco_njmax: int | None = None,
):
    solver = str(name).lower()
    if solver in ("semi_implicit", "semi-implicit", "semi"):
        return newton.solvers.SolverSemiImplicit(model)
    if solver == "mujoco":
        # Use Newton collision pipeline as the single source of contact generation.
        if mujoco_nconmax is None:
            # Ensure MuJoCo can accept converted Newton contacts for dense contact scenes.
            base_nconmax = int(getattr(model, "rigid_contact_max", 0) or 0)
            mujoco_nconmax = max(256, base_nconmax * 2)
        if mujoco_njmax is None:
            # Keep enough constraint slots for contacts + joints in dense scenes.
            base_dof = int(getattr(model, "joint_dof_count", 0) or 0)
            mujoco_njmax = max(1024, base_dof * 8, int(mujoco_nconmax) * 8)
        return newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_contacts=False,
            nconmax=int(mujoco_nconmax),
            njmax=int(mujoco_njmax),
        )
    if solver == "vbd":
        return newton.solvers.SolverVBD(model)
    raise ValueError(f"Unsupported solver: {name}")


def _requires_newton_contacts(solver_name: str) -> bool:
    solver = str(solver_name).lower()
    return solver in ("semi_implicit", "semi-implicit", "semi", "mujoco", "vbd")


def _model_body_ids(model) -> list | None:
    for attr in ("body_key", "body_label", "body_name"):
        if hasattr(model, attr):
            values = getattr(model, attr)
            if values is not None:
                return list(values)
    return None


def _configure_joint_hold(model: newton.Model, control: newton.Control, target_ke: float, target_kd: float) -> None:
    """Configure a simple global joint PD hold toward the model's reference pose."""
    if model.joint_count == 0 or model.joint_dof_count == 0:
        return
    if model.joint_target_ke is not None:
        model.joint_target_ke.fill_(float(target_ke))
    if model.joint_target_kd is not None:
        model.joint_target_kd.fill_(float(target_kd))
    if control is not None and control.joint_target_pos is not None and model.joint_q is not None:
        target_pos = control.joint_target_pos.numpy()
        q = model.joint_q.numpy()
        n = min(target_pos.shape[0], q.shape[0])
        if n > 0:
            target_pos[:n] = q[:n]
            control.joint_target_pos.assign(target_pos)
    if control is not None and control.joint_target_vel is not None:
        target_vel = control.joint_target_vel.numpy()
        target_vel.fill(0.0)
        control.joint_target_vel.assign(target_vel)


def _apply_robot_pose_preset(model: newton.Model, control: newton.Control, preset: str) -> None:
    """Apply a named joint-space pose preset to the robot model/control."""
    preset_name = str(preset).lower()
    if preset_name in ("", "none"):
        return

    if preset_name != "forward_reach":
        LOGGER.warning("Unknown robot pose preset: %s", preset)
        return

    # Heuristic arm pose that extends both manipulators forward.
    targets = {
        "lift_joint": 0.0,
        "arm_l_joint1": 0.0,
        "arm_l_joint2": 1.05,
        "arm_l_joint3": 1.55,
        "arm_l_joint4": 0.0,
        "arm_l_joint5": 1.10,
        "arm_l_joint6": 0.0,
        "arm_l_joint7": 0.0,
        "arm_r_joint1": 0.0,
        "arm_r_joint2": -1.05,
        "arm_r_joint3": 1.55,
        "arm_r_joint4": 0.0,
        "arm_r_joint5": 1.10,
        "arm_r_joint6": 0.0,
        "arm_r_joint7": 0.0,
    }
    _apply_named_joint_targets(model, control, targets)


def _joint_limits_by_name(model: newton.Model) -> dict[str, tuple[int, float, float]]:
    limits: dict[str, tuple[int, float, float]] = {}
    if model.joint_q_start is None:
        return limits
    joint_ids = None
    for attr in ("joint_key", "joint_label", "joint_name"):
        if hasattr(model, attr):
            values = getattr(model, attr)
            if values is not None:
                joint_ids = list(values)
                break
    if not joint_ids:
        return limits

    q_start = model.joint_q_start.numpy()
    lower = model.joint_limit_lower.numpy() if model.joint_limit_lower is not None else None
    upper = model.joint_limit_upper.numpy() if model.joint_limit_upper is not None else None

    full_names: list[str] = [str(name) for name in joint_ids]
    suffix_counts: dict[str, int] = {}
    for full_name in full_names:
        suffix = full_name.split("/")[-1]
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1

    for i, name in enumerate(full_names):
        qi = int(q_start[i])
        lo = float(lower[qi]) if lower is not None and 0 <= qi < lower.shape[0] else -np.inf
        hi = float(upper[qi]) if upper is not None and 0 <= qi < upper.shape[0] else np.inf
        limits[name] = (qi, lo, hi)
        suffix = name.split("/")[-1]
        if suffix_counts.get(suffix, 0) == 1:
            limits[suffix] = (qi, lo, hi)
    return limits


def _apply_named_joint_targets(model: newton.Model, control: newton.Control, targets: dict[str, float]) -> None:
    if not targets:
        return
    limits = _joint_limits_by_name(model)
    if not limits:
        return
    q = model.joint_q.numpy()
    changed = 0
    for joint_name, value in targets.items():
        info = limits.get(joint_name)
        if info is None:
            LOGGER.warning("Joint not found for target override: %s", joint_name)
            continue
        qi, lo, hi = info
        v = float(value)
        if np.isfinite(lo):
            v = max(v, lo)
        if np.isfinite(hi):
            v = min(v, hi)
        q[qi] = v
        changed += 1
    if changed == 0:
        return
    model.joint_q.assign(q)
    if control is not None and control.joint_target_pos is not None:
        target_pos = control.joint_target_pos.numpy()
        n = min(target_pos.shape[0], q.shape[0])
        if n > 0:
            target_pos[:n] = q[:n]
            control.joint_target_pos.assign(target_pos)


def _apply_single_arm_overrides(model: newton.Model, control: newton.Control, args: argparse.Namespace) -> None:
    side = "r" if str(args.active_arm).lower() == "right" else "l"
    arm_targets: dict[str, float] = {}
    vals = [
        args.arm_j1,
        args.arm_j2,
        args.arm_j3,
        args.arm_j4,
        args.arm_j5,
        args.arm_j6,
        args.arm_j7,
    ]
    for i, value in enumerate(vals, start=1):
        if value is None:
            continue
        arm_targets[f"arm_{side}_joint{i}"] = float(value)
    if args.lift_joint is not None:
        arm_targets["lift_joint"] = float(args.lift_joint)
    _apply_named_joint_targets(model, control, arm_targets)


def _log_arm_joint_ranges(model: newton.Model, active_arm: str) -> None:
    limits = _joint_limits_by_name(model)
    side = "r" if str(active_arm).lower() == "right" else "l"
    joint_names = ["lift_joint"] + [f"arm_{side}_joint{i}" for i in range(1, 8)]
    LOGGER.info("Joint ranges for active arm '%s':", active_arm)
    for name in joint_names:
        info = limits.get(name)
        if info is None:
            LOGGER.info("  %s: not found", name)
            continue
        _, lo, hi = info
        LOGGER.info("  %s: [%.6f, %.6f]", name, lo, hi)


def _resolve_joint_q_indices(model: newton.Model, joint_names: list[str]) -> list[int]:
    limits = _joint_limits_by_name(model)
    q_indices: list[int] = []
    for name in joint_names:
        info = limits.get(name)
        if info is None:
            continue
        qi = int(info[0])
        if qi not in q_indices:
            q_indices.append(qi)
    return q_indices


def _quat_rotate_np(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a vector by a quaternion in [x, y, z, w] format."""
    q = np.asarray(quat_xyzw, dtype=np.float32)
    v = np.asarray(vec, dtype=np.float32)
    q_xyz = q[:3]
    q_w = float(q[3])
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def _quat_inv_rotate_np(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a quaternion in [x, y, z, w] format."""
    q = np.asarray(quat_xyzw, dtype=np.float32)
    q_inv = np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)
    return _quat_rotate_np(q_inv, vec)


def _normalize_np(vec: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros(3, dtype=np.float32)
    return v / n


def _add_front_table(
    builder: newton.ModelBuilder,
    center_x: float,
    center_y: float,
    center_z: float,
    half_x: float,
    half_y: float,
    half_z: float,
) -> None:
    """Add a static tabletop box in front of the robot."""
    table_xform = wp.transform(wp.vec3(center_x, center_y, center_z), wp.quat_identity())
    builder.add_shape_box(
        body=-1,
        xform=table_xform,
        hx=half_x,
        hy=half_y,
        hz=half_z,
    )


def _load_data_and_params(path: str) -> tuple[SpringMassPKL | SpringMassPKLPair, dict]:
    """Load spring-mass data and optional scalar params from a PKL payload."""
    data = load_pkl(path)
    params: dict = {}
    with open(path, "rb") as f:
        raw = pickle.load(f)

    if isinstance(raw, dict):
        raw_params = raw.get("params")
        if isinstance(raw_params, dict):
            params.update(raw_params)

        # Handle simple top-level scalar params while ignoring structured data payload fields.
        data_keys = {
            "controller_mask",
            "controller_points",
            "object_points",
            "object_colors",
            "object_visibilities",
            "object_motions_valid",
            "surface_points",
            "interior_points",
            "gt",
            "predict",
            "pred",
            "params",
        }
        for k, v in raw.items():
            if k in data_keys:
                continue
            if isinstance(v, (bool, int, float, str)):
                params[k] = v

    params = {k: v for k, v in params.items() if k in INTERACTIVE_PARAM_KEYS and v is not None}
    return data, params


def _pick_param(pkl_params: dict, args: argparse.Namespace, key: str):
    if key in pkl_params and pkl_params[key] is not None:
        return pkl_params[key]
    return getattr(args, key, None)


def _pick_param_alias(pkl_params: dict, args: argparse.Namespace, keys: tuple[str, ...], fallback_key: str):
    for key in keys:
        if key in pkl_params and pkl_params[key] is not None:
            return pkl_params[key]
    return getattr(args, fallback_key, None)


def _apply_learned_spring_arrays(model: newton.Model, pkl_params: dict) -> None:
    stiffness = pkl_params.get("spring_stiffness")
    damping = pkl_params.get("spring_damping")

    if stiffness is not None and model.spring_stiffness is not None:
        arr = np.asarray(stiffness, dtype=np.float32).reshape(-1)
        dst = model.spring_stiffness.numpy()
        if arr.shape == dst.shape:
            model.spring_stiffness.assign(arr)
            LOGGER.info("Applied spring_stiffness from PKL params (count=%d).", arr.size)
        else:
            LOGGER.warning(
                "Ignoring spring_stiffness from PKL params due to shape mismatch: got %s expected %s.",
                arr.shape,
                dst.shape,
            )

    if damping is not None and model.spring_damping is not None:
        arr = np.asarray(damping, dtype=np.float32).reshape(-1)
        dst = model.spring_damping.numpy()
        if arr.shape == dst.shape:
            model.spring_damping.assign(arr)
            LOGGER.info("Applied spring_damping from PKL params (count=%d).", arr.size)
        else:
            LOGGER.warning(
                "Ignoring spring_damping from PKL params due to shape mismatch: got %s expected %s.",
                arr.shape,
                dst.shape,
            )


class Example:
    def __init__(self, viewer, args: argparse.Namespace):
        self.viewer = viewer
        data, pkl_params = _load_data_and_params(args.pkl)
        if isinstance(data, SpringMassPKLPair):
            spring_data = data.predict
        else:
            spring_data = data

        self.fps = float(_pick_param_alias(pkl_params, args, ("fps", "FPS"), "fps"))
        self.frame_dt = 1.0 / self.fps
        if args.substeps is not None:
            # CLI should override PKL-provided substeps when explicitly set.
            self.sim_substeps = max(1, int(args.substeps))
        else:
            pkl_substeps = _pick_param_alias(pkl_params, args, ("substeps", "num_substeps"), "substeps")
            self.sim_substeps = max(1, int(pkl_substeps)) if pkl_substeps is not None else 20
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        scale = float(_pick_param(pkl_params, args, "scale"))
        z_offset = float(_pick_param(pkl_params, args, "z_offset"))
        reverse_z = bool(_pick_param(pkl_params, args, "reverse_z"))
        particle_radius = float(_pick_param(pkl_params, args, "particle_radius"))
        particle_mass = float(_pick_param_alias(pkl_params, args, ("mass", "particle_mass"), "mass"))
        if args.particle_mass_override is not None:
            particle_mass = float(args.particle_mass_override)
        spring_ke = float(
            _pick_param_alias(
                pkl_params,
                args,
                ("spring_ke", "init_spring_Y"),
                "spring_ke",
            )
        )
        spring_kd = float(
            _pick_param_alias(
                pkl_params,
                args,
                ("spring_kd", "dashpot_damping"),
                "spring_kd",
            )
        )
        k_neighbors = int(
            _pick_param_alias(
                pkl_params,
                args,
                ("k_neighbors", "object_max_neighbours"),
                "k_neighbors",
            )
        )
        use_controllers_raw = _pick_param(pkl_params, args, "use_controllers")
        if bool(getattr(args, "disable_pkl_controllers", False)):
            use_controllers_raw = False
        # If drag is enabled from CLI, ensure controller particles exist even when PKL omits this flag.
        use_controllers = bool(use_controllers_raw) or bool(getattr(args, "enable_controller_drag", False))
        controller_k_val = _pick_param(pkl_params, args, "controller_k")
        controller_ke_val = _pick_param(pkl_params, args, "controller_ke")
        controller_kd_val = _pick_param(pkl_params, args, "controller_kd")
        controller_k = int(controller_k_val) if controller_k_val is not None else 1
        controller_ke = float(controller_ke_val) if controller_ke_val is not None else 1.0e4
        controller_kd = float(controller_kd_val) if controller_kd_val is not None else 1.0e1
        filter_visibility = bool(_pick_param(pkl_params, args, "filter_visibility"))
        filter_motion_valid = bool(_pick_param(pkl_params, args, "filter_motion_valid"))
        spring_neighbor_mode = str(_pick_param(pkl_params, args, "spring_neighbor_mode"))
        object_radius = _pick_param(pkl_params, args, "object_radius")
        object_max_neighbours = _pick_param(pkl_params, args, "object_max_neighbours")
        controller_mass = _pick_param(pkl_params, args, "controller_mass")
        if controller_mass is None and bool(getattr(args, "enable_controller_drag", False)):
            # Force-drag needs dynamic controller particles; default to non-zero mass.
            controller_mass = 1.0
        table_center_x = float(args.table_center_x)
        table_center_y = float(args.table_center_y)
        table_center_z = float(args.table_center_z)
        table_half_x = float(args.table_half_x)
        table_half_y = float(args.table_half_y)
        table_half_z = float(args.table_half_z)
        table_cloth_clearance = float(args.table_cloth_clearance)

        urdf_xform = None
        if args.urdf_offset is not None:
            offset = np.asarray(args.urdf_offset, dtype=np.float32)
            if offset.size == 3 and np.linalg.norm(offset) > 0.0:
                urdf_xform = wp.transform(wp.vec3(*offset.tolist()), wp.quat_identity())

        # Build spring model with robot colliders duplicated inside the same model so
        # spring particle contacts can be generated against robot geometry.
        # Keep the same world convention as robot_model to avoid collider frame mismatch.
        spring_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        spring_builder.add_urdf(
            args.urdf,
            xform=urdf_xform,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
        )
        _add_front_table(
            spring_builder,
            table_center_x,
            table_center_y,
            table_center_z,
            table_half_x,
            table_half_y,
            table_half_z,
        )

        predict_mask = _object_mask(
            spring_data,
            frame=0,
            filter_visibility=filter_visibility,
            filter_motion_valid=filter_motion_valid,
        )
        predict_points = spring_data.object_points[0][predict_mask]
        scaled_predict = _apply_scale(predict_points, scale, reverse_z)
        z_shift_ground = _compute_z_shift(scaled_predict, z_offset)
        if scaled_predict.size:
            min_scaled_z = float(scaled_predict[:, 2].min())
            table_top_z = table_center_z + table_half_z
            z_shift_table = (table_top_z + table_cloth_clearance) - min_scaled_z
            z_shift = max(z_shift_ground, z_shift_table)
        else:
            z_shift = z_shift_ground

        spring_mapping = map_pkl_to_newton(
            data=spring_data,
            frame=0,
            scale=scale,
            z_offset=z_shift,
            reverse_z=reverse_z,
            particle_radius=particle_radius,
            mass=particle_mass,
            spring_ke=spring_ke,
            spring_kd=spring_kd,
            k_neighbors=k_neighbors,
            add_ground=not args.no_ground,
            use_controllers=use_controllers,
            controller_k=controller_k,
            controller_ke=controller_ke,
            controller_kd=controller_kd,
            filter_visibility=filter_visibility,
            filter_motion_valid=filter_motion_valid,
            spring_neighbor_mode=spring_neighbor_mode,
            object_radius=object_radius,
            object_max_neighbours=object_max_neighbours,
            builder=spring_builder,
        )
        self.spring_model = spring_mapping.model
        self.spring_object_indices = spring_mapping.object_particle_indices.astype(np.int32)
        self.spring_controller_indices = spring_mapping.controller_particle_indices.astype(np.int32)
        _apply_learned_spring_arrays(self.spring_model, pkl_params)
        # Strengthen particle-vs-robot contact response for spring simulation.
        self.spring_model.soft_contact_ke = float(args.spring_soft_contact_ke)
        self.spring_model.soft_contact_kd = float(args.spring_soft_contact_kd)
        self.spring_model.soft_contact_kf = float(args.spring_soft_contact_kf)
        self.spring_model.soft_contact_mu = float(args.spring_soft_contact_mu)
        default_margin = max(0.01, particle_radius * 0.75)
        self.spring_soft_contact_margin = (
            float(args.spring_soft_contact_margin)
            if args.spring_soft_contact_margin is not None
            else default_margin
        )

        if controller_mass is not None and controller_mass > 0.0:
            ctrl_indices = self.spring_controller_indices
            if ctrl_indices.size and self.spring_model.particle_mass is not None and self.spring_model.particle_inv_mass is not None:
                mass_np = self.spring_model.particle_mass.numpy()
                inv_np = self.spring_model.particle_inv_mass.numpy()
                mass_np[ctrl_indices] = float(controller_mass)
                inv_np[ctrl_indices] = 1.0 / float(controller_mass)
                self.spring_model.particle_mass.assign(mass_np)
                self.spring_model.particle_inv_mass.assign(inv_np)

        self.spring_solver = _build_solver(
            args.spring_solver,
            self.spring_model,
            mujoco_nconmax=args.mujoco_nconmax,
            mujoco_njmax=args.mujoco_njmax,
        )
        self.spring_state_0 = self.spring_model.state()
        self.spring_state_1 = self.spring_model.state()
        self.spring_control = self.spring_model.control()
        if args.spring_offset is not None:
            spring_offset = np.asarray(args.spring_offset, dtype=np.float32)
            if spring_offset.size == 3 and np.linalg.norm(spring_offset) > 0.0:
                q0 = self.spring_state_0.particle_q.numpy()
                q1 = self.spring_state_1.particle_q.numpy()
                q0[:, :3] += spring_offset[None, :]
                q1[:, :3] += spring_offset[None, :]
                self.spring_state_0.particle_q.assign(q0)
                self.spring_state_1.particle_q.assign(q1)
        self.spring_contacts = None
        self.spring_collisions_enabled = bool(args.enable_spring_collisions) or _requires_newton_contacts(args.spring_solver)
        if self.spring_collisions_enabled:
            self.spring_collision_pipeline = newton.examples.create_collision_pipeline(
                self.spring_model,
                args=args,
                soft_contact_margin=self.spring_soft_contact_margin,
            )
            self.spring_contacts = self.spring_model.collide(
                self.spring_state_0,
                collision_pipeline=self.spring_collision_pipeline,
            )
        else:
            self.spring_collision_pipeline = None

        # Keep duplicated robot bodies in spring model kinematic (collision-only).
        if self.spring_model.body_count > 0:
            self.spring_model.body_mass.zero_()
            self.spring_model.body_inv_mass.zero_()
            self.spring_model.body_inertia.zero_()
            self.spring_model.body_inv_inertia.zero_()

        # Build robot model (dynamics source of truth).
        robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        robot_builder.add_urdf(
            args.urdf,
            xform=urdf_xform,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
        )
        _add_front_table(
            robot_builder,
            table_center_x,
            table_center_y,
            table_center_z,
            table_half_x,
            table_half_y,
            table_half_z,
        )
        if not args.no_ground:
            robot_builder.add_ground_plane()
        self.robot_model = robot_builder.finalize()
        self.robot_state_0 = self.robot_model.state()
        self.robot_state_1 = self.robot_model.state()
        self.robot_control = self.robot_model.control()
        _configure_joint_hold(
            self.robot_model,
            self.robot_control,
            target_ke=float(args.robot_joint_ke),
            target_kd=float(args.robot_joint_kd),
        )
        _apply_robot_pose_preset(self.robot_model, self.robot_control, args.robot_pose_preset)
        _apply_single_arm_overrides(self.robot_model, self.robot_control, args)
        if args.print_arm_joint_ranges:
            _log_arm_joint_ranges(self.robot_model, args.active_arm)
        self._init_right_gripper_autoclose(args)
        self.robot_solver = _build_solver(
            args.robot_solver,
            self.robot_model,
            mujoco_nconmax=args.mujoco_nconmax,
            mujoco_njmax=args.mujoco_njmax,
        )
        # Ensure backends that cache model properties (e.g., MuJoCo) see updated gains.
        if hasattr(self.robot_solver, "notify_model_changed"):
            self.robot_solver.notify_model_changed(newton.solvers.SolverNotifyFlags.JOINT_DOF_PROPERTIES)
            self.robot_solver.notify_model_changed(newton.solvers.SolverNotifyFlags.ACTUATOR_PROPERTIES)
        newton.eval_fk(self.robot_model, self.robot_model.joint_q, self.robot_model.joint_qd, self.robot_state_0)
        self._init_robot_spring_collider_sync()
        self.robot_contacts = None
        self.robot_collisions_enabled = bool(args.enable_robot_collisions) or _requires_newton_contacts(args.robot_solver)
        if self.robot_collisions_enabled:
            self.robot_collision_pipeline = newton.examples.create_collision_pipeline(self.robot_model, args=args)
            self.robot_contacts = self.robot_model.collide(
                self.robot_state_0, collision_pipeline=self.robot_collision_pipeline
            )
        else:
            self.robot_collision_pipeline = None

        # Viewer: attach robot model, render spring-mass manually.
        self.viewer.set_model(self.robot_model)
        self.viewer.show_visual = True
        self.viewer.show_collision = True
        self.viewer.show_static = True
        self.viewer.show_triangles = True

        self.mesh_enabled = bool(args.visualize_mesh)
        self.mesh_show_points = bool(args.mesh_show_points or not self.mesh_enabled)
        self.visual_z_offset = float(args.visual_z_offset)
        self._spring_particle_radius_host = (
            self.spring_model.particle_radius.numpy().copy()
            if self.spring_model.particle_radius is not None
            else None
        )
        self._spring_particle_radius_wp = (
            self.spring_model.particle_radius
            if self.spring_model.particle_radius is not None
            else wp.full(
                self.spring_model.particle_count,
                float(particle_radius),
                dtype=float,
                device=self.spring_model.device,
            )
        )
        self.object_radius_value = (
            float(self._spring_particle_radius_host[0])
            if self._spring_particle_radius_host is not None
            else particle_radius
        )
        self.mesh_indices_wp = None
        self.mesh_point_count = 0

        self._init_spring_mesh(args)
        self._frame_camera(args.frame_on)
        self._init_controller_drag(args)
        self._spring_robot_body_set = set(self._robot_to_spring_spring_body_idx.tolist())
        self._spring_robot_shape_mask = None
        if self.spring_model.shape_body is not None and self.spring_model.shape_count > 0:
            shape_body = self.spring_model.shape_body.numpy()
            self._spring_robot_shape_mask = np.isin(shape_body, np.array(sorted(self._spring_robot_body_set), dtype=np.int32))
        self._init_contact_patch(args)

    def _init_right_gripper_autoclose(self, args: argparse.Namespace) -> None:
        self.auto_close_right_gripper = bool(args.auto_close_right_gripper)
        self.gripper_close_start_time = float(args.gripper_close_start_time)
        self.gripper_close_duration = max(float(args.gripper_close_duration), 1.0e-6)
        self.gripper_close_target = float(np.clip(args.gripper_close_target, 0.0, 1.0))
        self._robot_joint_target_pos_host = self.robot_control.joint_target_pos.numpy().copy()
        self._robot_joint_target_pos_dirty = False
        self._last_gripper_alpha: float | None = None
        self._last_arm_j1_alpha: float | None = None
        self._right_gripper_q_indices: np.ndarray | None = None
        self._right_gripper_open_targets: np.ndarray | None = None
        self._right_gripper_close_targets: np.ndarray | None = None
        self.auto_sweep_arm_j1_after_grip = bool(args.auto_sweep_arm_j1_after_grip)
        self.arm_j1_sweep_target = float(args.arm_j1_sweep_target)
        self.arm_j1_sweep_duration = max(float(args.arm_j1_sweep_duration), 1.0e-6)
        self._arm_j1_q_index: int | None = None
        self._arm_j1_start_target: float | None = None
        self._arm_j1_sweep_start_time: float | None = None
        self._arm_j1_sweep_clamped_target: float | None = None
        if not self.auto_close_right_gripper:
            return
        joint_names = [
            "gripper_r_joint",
            "gripper_r_joint2",
            "gripper_r_joint3",
            "gripper_r_joint4",
        ]
        q_indices = _resolve_joint_q_indices(self.robot_model, joint_names)
        if not q_indices:
            LOGGER.warning("Auto right gripper close enabled but gripper joints were not found.")
            self.auto_close_right_gripper = False
            return
        q_indices_np = np.asarray(q_indices, dtype=np.int32)
        lower = self.robot_model.joint_limit_lower.numpy()[q_indices_np]
        upper = self.robot_model.joint_limit_upper.numpy()[q_indices_np]
        open_targets = self.robot_control.joint_target_pos.numpy()[q_indices_np].copy()
        close_targets = lower + self.gripper_close_target * (upper - lower)
        close_targets = np.clip(close_targets, lower, upper)
        self._right_gripper_q_indices = q_indices_np
        self._right_gripper_open_targets = open_targets
        self._right_gripper_close_targets = close_targets
        if self.auto_sweep_arm_j1_after_grip:
            side = "r" if str(args.active_arm).lower() == "right" else "l"
            joint_name = f"arm_{side}_joint1"
            limits = _joint_limits_by_name(self.robot_model)
            info = limits.get(joint_name)
            if info is None:
                LOGGER.warning("Auto arm-j1 sweep enabled but joint '%s' not found.", joint_name)
                self.auto_sweep_arm_j1_after_grip = False
            else:
                qi, lo, hi = info
                self._arm_j1_q_index = int(qi)
                start_targets = self.robot_control.joint_target_pos.numpy()
                self._arm_j1_start_target = float(start_targets[self._arm_j1_q_index])
                target = self.arm_j1_sweep_target
                if np.isfinite(lo):
                    target = max(target, float(lo))
                if np.isfinite(hi):
                    target = min(target, float(hi))
                self._arm_j1_sweep_clamped_target = float(target)
                self._arm_j1_sweep_start_time = self.gripper_close_start_time + self.gripper_close_duration

    def _update_right_gripper_autoclose(self, current_time: float) -> None:
        if not self.auto_close_right_gripper:
            return
        if self._right_gripper_q_indices is None:
            return
        if self._right_gripper_open_targets is None or self._right_gripper_close_targets is None:
            return
        alpha = (current_time - self.gripper_close_start_time) / self.gripper_close_duration
        alpha = float(np.clip(alpha, 0.0, 1.0))
        if self._last_gripper_alpha is not None and abs(alpha - self._last_gripper_alpha) <= 1.0e-6:
            return
        targets = (1.0 - alpha) * self._right_gripper_open_targets + alpha * self._right_gripper_close_targets
        self._robot_joint_target_pos_host[self._right_gripper_q_indices] = targets
        self._robot_joint_target_pos_dirty = True
        self._last_gripper_alpha = alpha

    def _update_post_grip_arm_j1_sweep(self, current_time: float) -> None:
        if not self.auto_sweep_arm_j1_after_grip:
            return
        if self._arm_j1_q_index is None:
            return
        if self._arm_j1_start_target is None or self._arm_j1_sweep_clamped_target is None:
            return
        if self._arm_j1_sweep_start_time is None:
            return
        alpha = (current_time - self._arm_j1_sweep_start_time) / self.arm_j1_sweep_duration
        alpha = float(np.clip(alpha, 0.0, 1.0))
        if self._last_arm_j1_alpha is not None and abs(alpha - self._last_arm_j1_alpha) <= 1.0e-6:
            return
        value = (1.0 - alpha) * self._arm_j1_start_target + alpha * self._arm_j1_sweep_clamped_target
        self._robot_joint_target_pos_host[self._arm_j1_q_index] = value
        self._robot_joint_target_pos_dirty = True
        self._last_arm_j1_alpha = alpha

    def _init_contact_patch(self, args: argparse.Namespace) -> None:
        self.contact_patch_enabled = bool(args.enable_contact_patch)
        self.contact_patch_arm = str(args.contact_patch_arm).lower()
        self.contact_patch_max_particles = max(1, int(args.contact_patch_max_particles))
        self.contact_patch_min_contacts = max(1, int(args.contact_patch_min_contacts))
        self.contact_patch_normal_ke = float(args.contact_patch_normal_ke)
        self.contact_patch_normal_kd = float(args.contact_patch_normal_kd)
        self.contact_patch_tangent_ke = float(args.contact_patch_tangent_ke)
        self.contact_patch_tangent_kd = float(args.contact_patch_tangent_kd)
        self.contact_patch_mu = max(0.0, float(args.contact_patch_mu))
        self.contact_patch_max_force = max(0.0, float(args.contact_patch_max_force))
        self.contact_patch_max_gap = float(args.contact_patch_max_gap)
        self.contact_patch_break_distance = max(0.0, float(args.contact_patch_break_distance))
        self.contact_patch_release_missed_steps = max(1, int(args.contact_patch_release_missed_steps))
        self.contact_patch_refresh_interval = max(1, int(args.contact_patch_refresh_interval))
        self.contact_patch_log_interval = max(0, int(args.contact_patch_log_interval))
        self._patch_refresh_counter = 0

        self._patch_particles = np.zeros(0, dtype=np.int32)
        self._patch_body_indices = np.zeros(0, dtype=np.int32)
        self._patch_anchor_local = np.zeros((0, 3), dtype=np.float32)
        self._patch_normal_local = np.zeros((0, 3), dtype=np.float32)
        self._patch_missed_steps = np.zeros(0, dtype=np.int32)
        self._patch_step_counter = 0

        self._patch_shape_mask = None
        self._patch_shape_mask_wp = None
        self._shape_body_indices = None
        self._shape_body_indices_wp = None
        self._patch_particles_wp = None
        self._patch_body_indices_wp = None
        self._patch_anchor_local_wp = None
        self._patch_normal_local_wp = None
        self._patch_break_keep_wp = None
        self._patch_candidate_capacity = max(256, self.contact_patch_max_particles * 8)
        self._patch_candidate_count_wp = wp.zeros(1, dtype=wp.int32, device=self.spring_model.device)
        self._patch_candidate_particle_wp = wp.full(
            self._patch_candidate_capacity,
            -1,
            dtype=wp.int32,
            device=self.spring_model.device,
        )
        self._patch_candidate_shape_wp = wp.full(
            self._patch_candidate_capacity,
            -1,
            dtype=wp.int32,
            device=self.spring_model.device,
        )
        self._patch_candidate_body_pos_wp = wp.zeros(
            self._patch_candidate_capacity,
            dtype=wp.vec3,
            device=self.spring_model.device,
        )
        self._patch_candidate_normal_wp = wp.zeros(
            self._patch_candidate_capacity,
            dtype=wp.vec3,
            device=self.spring_model.device,
        )
        self._patch_filtered_count_wp = wp.zeros(1, dtype=wp.int32, device=self.spring_model.device)
        self._patch_filtered_particle_wp = wp.full(
            self._patch_candidate_capacity,
            -1,
            dtype=wp.int32,
            device=self.spring_model.device,
        )
        self._patch_filtered_shape_wp = wp.full(
            self._patch_candidate_capacity,
            -1,
            dtype=wp.int32,
            device=self.spring_model.device,
        )
        self._patch_filtered_normal_wp = wp.zeros(
            self._patch_candidate_capacity,
            dtype=wp.vec3,
            device=self.spring_model.device,
        )
        if self.spring_model.shape_body is not None and self.spring_model.shape_count > 0:
            self._shape_body_indices = self.spring_model.shape_body.numpy().astype(np.int32, copy=True)
            self._shape_body_indices_wp = wp.array(
                self._shape_body_indices.tolist(),
                dtype=wp.int32,
                device=self.spring_model.device,
            )
        if not self.contact_patch_enabled:
            return
        if self._shape_body_indices is None:
            LOGGER.warning("Contact patch enabled but spring model has no shape-body mapping.")
            self.contact_patch_enabled = False
            return

        body_ids = _model_body_ids(self.spring_model)
        if body_ids is None:
            LOGGER.warning("Contact patch enabled but spring model has no body identifiers.")
            self.contact_patch_enabled = False
            return
        body_names = [str(k).lower() for k in body_ids]
        side_token = "r" if self.contact_patch_arm == "right" else "l"
        preferred = [i for i, name in enumerate(body_names) if f"gripper_{side_token}_" in name]
        if not preferred:
            LOGGER.warning(
                "Contact patch enabled but no matching '%s' gripper bodies were found.",
                self.contact_patch_arm,
            )
            self.contact_patch_enabled = False
            return

        body_mask = np.zeros(self.spring_model.body_count, dtype=bool)
        body_mask[np.asarray(preferred, dtype=np.int32)] = True
        shape_mask = np.zeros(self.spring_model.shape_count, dtype=bool)
        valid_shape = self._shape_body_indices >= 0
        shape_mask[valid_shape] = body_mask[self._shape_body_indices[valid_shape]]
        if not np.any(shape_mask):
            LOGGER.warning("Contact patch enabled but no shapes were found for selected gripper bodies.")
            self.contact_patch_enabled = False
            return
        self._patch_shape_mask = shape_mask
        self._patch_shape_mask_wp = wp.array(
            shape_mask.astype(np.int32).tolist(),
            dtype=wp.int32,
            device=self.spring_model.device,
        )
        LOGGER.info(
            (
                "Contact patch enabled: arm=%s, candidate_shapes=%d, max_particles=%d, "
                "refresh_interval=%d, max_gap=%.4f."
            ),
            self.contact_patch_arm,
            int(np.count_nonzero(shape_mask)),
            self.contact_patch_max_particles,
            self.contact_patch_refresh_interval,
            self.contact_patch_max_gap,
        )

    def _sync_patch_device_buffers(self) -> None:
        count = int(self._patch_particles.shape[0])
        if count == 0:
            self._patch_particles_wp = None
            self._patch_body_indices_wp = None
            self._patch_anchor_local_wp = None
            self._patch_normal_local_wp = None
            self._patch_break_keep_wp = None
            return

        self._patch_particles_wp = wp.array(
            self._patch_particles.astype(np.int32).tolist(),
            dtype=wp.int32,
            device=self.spring_model.device,
        )
        self._patch_body_indices_wp = wp.array(
            self._patch_body_indices.astype(np.int32).tolist(),
            dtype=wp.int32,
            device=self.spring_model.device,
        )
        self._patch_anchor_local_wp = wp.array(
            [wp.vec3(*v.tolist()) for v in self._patch_anchor_local.astype(np.float32)],
            dtype=wp.vec3,
            device=self.spring_model.device,
        )
        self._patch_normal_local_wp = wp.array(
            [wp.vec3(*v.tolist()) for v in self._patch_normal_local.astype(np.float32)],
            dtype=wp.vec3,
            device=self.spring_model.device,
        )
        self._patch_break_keep_wp = wp.zeros(count, dtype=wp.int32, device=self.spring_model.device)

    def _gather_patch_contact_candidates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.spring_contacts is None or self._patch_shape_mask_wp is None:
            return (
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
            )

        soft_count_arr = self.spring_contacts.soft_contact_count.numpy()
        soft_count = int(soft_count_arr[0]) if soft_count_arr.size else 0
        if soft_count <= 0:
            return (
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
            )

        self._patch_candidate_count_wp.zero_()
        wp.launch(
            gather_patch_contact_candidates_kernel,
            dim=soft_count,
            inputs=[
                self.spring_contacts.soft_contact_count,
                self.spring_contacts.soft_contact_particle,
                self.spring_contacts.soft_contact_shape,
                self.spring_contacts.soft_contact_body_pos,
                self.spring_contacts.soft_contact_normal,
                self._patch_shape_mask_wp,
                self._patch_candidate_capacity,
                self._patch_candidate_count_wp,
                self._patch_candidate_particle_wp,
                self._patch_candidate_shape_wp,
                self._patch_candidate_body_pos_wp,
                self._patch_candidate_normal_wp,
            ],
            device=self.spring_model.device,
        )

        candidate_count = int(self._patch_candidate_count_wp.numpy()[0])
        candidate_count = min(candidate_count, self._patch_candidate_capacity)
        if candidate_count <= 0:
            return (
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
            )
        seen_particles = self._patch_candidate_particle_wp.numpy()[:candidate_count].astype(np.int32, copy=True)
        if self._shape_body_indices_wp is None:
            return (
                seen_particles,
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
            )

        self._patch_filtered_count_wp.zero_()
        wp.launch(
            filter_patch_contact_candidates_by_touch_kernel,
            dim=candidate_count,
            inputs=[
                self._patch_candidate_particle_wp,
                self._patch_candidate_shape_wp,
                self._patch_candidate_body_pos_wp,
                self._patch_candidate_normal_wp,
                self._shape_body_indices_wp,
                self.spring_state_0.particle_q,
                self.spring_state_0.body_q,
                self._spring_particle_radius_wp,
                self.contact_patch_max_gap,
                self._patch_candidate_capacity,
                self._patch_filtered_count_wp,
                self._patch_filtered_particle_wp,
                self._patch_filtered_shape_wp,
                self._patch_filtered_normal_wp,
            ],
            device=self.spring_model.device,
        )

        filtered_count = int(self._patch_filtered_count_wp.numpy()[0])
        filtered_count = min(filtered_count, self._patch_candidate_capacity)
        if filtered_count <= 0:
            return (
                seen_particles,
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
            )

        acquire_particles = self._patch_filtered_particle_wp.numpy()[:filtered_count].astype(np.int32, copy=False)
        acquire_shapes = self._patch_filtered_shape_wp.numpy()[:filtered_count].astype(np.int32, copy=False)
        acquire_normals = self._patch_filtered_normal_wp.numpy()[:filtered_count].astype(np.float32, copy=False)
        return seen_particles, acquire_particles, acquire_shapes, acquire_normals

    def _append_patch_anchor(
        self,
        particle_idx: int,
        body_idx: int,
        normal_world: np.ndarray,
        q_particle: np.ndarray,
        q_body: np.ndarray,
    ) -> None:
        if self._patch_particles.shape[0] >= self.contact_patch_max_particles:
            return
        if body_idx < 0 or body_idx >= self.spring_model.body_count:
            return
        if particle_idx < 0 or particle_idx >= q_particle.shape[0]:
            return

        body_pos = q_body[body_idx, :3].astype(np.float32, copy=False)
        body_quat = q_body[body_idx, 3:7].astype(np.float32, copy=False)
        particle_pos = q_particle[particle_idx, :3].astype(np.float32, copy=False)
        anchor_local = _quat_inv_rotate_np(body_quat, particle_pos - body_pos)
        normal_local = _quat_inv_rotate_np(body_quat, _normalize_np(normal_world))

        self._patch_particles = np.concatenate([self._patch_particles, np.array([particle_idx], dtype=np.int32)])
        self._patch_body_indices = np.concatenate([self._patch_body_indices, np.array([body_idx], dtype=np.int32)])
        self._patch_anchor_local = np.vstack([self._patch_anchor_local, anchor_local[None, :]])
        self._patch_normal_local = np.vstack([self._patch_normal_local, normal_local[None, :]])
        self._patch_missed_steps = np.concatenate([self._patch_missed_steps, np.array([0], dtype=np.int32)])

    def _prune_patch(self, keep_mask: np.ndarray) -> None:
        if keep_mask.size != self._patch_particles.shape[0]:
            return
        self._patch_particles = self._patch_particles[keep_mask]
        self._patch_body_indices = self._patch_body_indices[keep_mask]
        self._patch_anchor_local = self._patch_anchor_local[keep_mask]
        self._patch_normal_local = self._patch_normal_local[keep_mask]
        self._patch_missed_steps = self._patch_missed_steps[keep_mask]

    def _refresh_contact_patch(self) -> None:
        if not self.contact_patch_enabled:
            return
        if self._patch_shape_mask is None or self._shape_body_indices is None:
            return
        seen_particles, acquire_particles, acquire_shapes, acquire_normals = self._gather_patch_contact_candidates()
        patch_changed = False

        if self._patch_particles.size > 0:
            if seen_particles.size > 0:
                seen = np.isin(self._patch_particles, seen_particles)
                self._patch_missed_steps = np.where(seen, 0, self._patch_missed_steps + 1)
            else:
                self._patch_missed_steps = self._patch_missed_steps + 1

        existing = set(self._patch_particles.tolist())
        q_particle = None
        q_body = None
        if acquire_particles.size > 0:
            unique_particles, first_idx = np.unique(acquire_particles, return_index=True)
            order = np.argsort(first_idx)
            unique_particles = unique_particles[order]
            first_idx = first_idx[order]
            can_acquire = self._patch_particles.size > 0 or unique_particles.size >= self.contact_patch_min_contacts
            if can_acquire:
                for i, particle_idx in enumerate(unique_particles.tolist()):
                    if self._patch_particles.shape[0] >= self.contact_patch_max_particles:
                        break
                    if particle_idx in existing:
                        continue
                    if q_particle is None or q_body is None:
                        # Only pull transforms to host when creating new anchors.
                        q_particle = self.spring_state_0.particle_q.numpy()
                        q_body = self.spring_state_0.body_q.numpy()
                    src_idx = int(first_idx[i])
                    shape_idx = int(acquire_shapes[src_idx])
                    body_idx = int(self._shape_body_indices[shape_idx])
                    normal_world = acquire_normals[src_idx]
                    self._append_patch_anchor(particle_idx, body_idx, normal_world, q_particle, q_body)
                    existing.add(particle_idx)
                    patch_changed = True

        if patch_changed:
            self._sync_patch_device_buffers()
            patch_changed = False

        if self._patch_particles.size == 0:
            return

        keep = self._patch_missed_steps <= self.contact_patch_release_missed_steps
        if (
            self.contact_patch_break_distance > 0.0
            and self._patch_particles_wp is not None
            and self._patch_break_keep_wp is not None
        ):
            wp.launch(
                eval_patch_break_mask_kernel,
                dim=int(self._patch_particles.shape[0]),
                inputs=[
                    self.spring_state_0.particle_q,
                    self.spring_state_0.body_q,
                    self._patch_particles_wp,
                    self._patch_body_indices_wp,
                    self._patch_anchor_local_wp,
                    self.contact_patch_break_distance,
                    self._patch_break_keep_wp,
                ],
                device=self.spring_model.device,
            )
            keep_break = self._patch_break_keep_wp.numpy().astype(bool, copy=False)
            keep &= keep_break

        prev_count = int(self._patch_particles.shape[0])
        self._prune_patch(keep)
        if int(self._patch_particles.shape[0]) != prev_count:
            patch_changed = True

        if patch_changed:
            self._sync_patch_device_buffers()

    def _apply_contact_patch_constraints(self) -> None:
        if not self.contact_patch_enabled:
            return
        patch_count = int(self._patch_particles.shape[0])
        if patch_count == 0:
            return
        if (
            self._patch_particles_wp is None
            or self._patch_body_indices_wp is None
            or self._patch_anchor_local_wp is None
            or self._patch_normal_local_wp is None
        ):
            self._sync_patch_device_buffers()
            if self._patch_particles_wp is None:
                return

        wp.launch(
            apply_contact_patch_force_kernel,
            dim=patch_count,
            inputs=[
                self.spring_state_0.particle_q,
                self.spring_state_0.particle_qd,
                self.spring_state_0.particle_f,
                self.spring_state_0.body_q,
                self.spring_state_0.body_qd,
                self._patch_particles_wp,
                self._patch_body_indices_wp,
                self._patch_anchor_local_wp,
                self._patch_normal_local_wp,
                self.contact_patch_normal_ke,
                self.contact_patch_normal_kd,
                self.contact_patch_tangent_ke,
                self.contact_patch_tangent_kd,
                self.contact_patch_mu,
                self.contact_patch_max_force,
            ],
            device=self.spring_model.device,
        )

        self._patch_step_counter += 1
        if self.contact_patch_log_interval > 0 and (self._patch_step_counter % self.contact_patch_log_interval == 0):
            LOGGER.info("Contact patch active particles: %d", self._patch_particles.shape[0])

    def _init_robot_spring_collider_sync(self) -> None:
        """Create body-index mapping for syncing robot poses into spring colliders."""
        robot_ids = _model_body_ids(self.robot_model)
        spring_ids = _model_body_ids(self.spring_model)

        robot_indices: list[int] = []
        spring_indices: list[int] = []
        if robot_ids is not None and spring_ids is not None:
            spring_by_id = {body_id: i for i, body_id in enumerate(spring_ids)}
            for r_idx, body_id in enumerate(robot_ids):
                s_idx = spring_by_id.get(body_id)
                if s_idx is None:
                    continue
                robot_indices.append(r_idx)
                spring_indices.append(s_idx)

        self._robot_to_spring_robot_body_idx = np.asarray(robot_indices, dtype=np.int32)
        self._robot_to_spring_spring_body_idx = np.asarray(spring_indices, dtype=np.int32)
        if self._robot_to_spring_robot_body_idx.size == 0:
            if self.robot_model.body_count == self.spring_model.body_count:
                # Fallback for exporters that rewrite body keys but preserve order.
                self._robot_to_spring_robot_body_idx = np.arange(self.robot_model.body_count, dtype=np.int32)
                self._robot_to_spring_spring_body_idx = np.arange(self.spring_model.body_count, dtype=np.int32)
                LOGGER.warning(
                    "No overlapping body identifiers found. Falling back to index-based robot->spring collider sync."
                )
            else:
                LOGGER.warning("No overlapping body identifiers found between robot and spring collider models.")

        self._robot_to_spring_robot_body_idx_wp = None
        self._robot_to_spring_spring_body_idx_wp = None
        if self._robot_to_spring_robot_body_idx.size > 0:
            self._robot_to_spring_robot_body_idx_wp = wp.array(
                self._robot_to_spring_robot_body_idx.tolist(),
                dtype=wp.int32,
                device=self.spring_model.device,
            )
            self._robot_to_spring_spring_body_idx_wp = wp.array(
                self._robot_to_spring_spring_body_idx.tolist(),
                dtype=wp.int32,
                device=self.spring_model.device,
            )

    def _sync_robot_to_spring_colliders(self) -> None:
        """Copy robot body transforms/velocities into spring-model kinematic colliders."""
        if self._robot_to_spring_robot_body_idx.size == 0:
            return

        if self._robot_to_spring_robot_body_idx_wp is None or self._robot_to_spring_spring_body_idx_wp is None:
            return
        count = int(self._robot_to_spring_robot_body_idx.size)
        if count <= 0:
            return

        wp.launch(
            copy_indexed_transform_kernel,
            dim=count,
            inputs=[
                self.robot_state_0.body_q,
                self.spring_state_0.body_q,
                self._robot_to_spring_robot_body_idx_wp,
                self._robot_to_spring_spring_body_idx_wp,
            ],
            device=self.spring_model.device,
        )
        wp.launch(
            copy_indexed_spatial_vector_kernel,
            dim=count,
            inputs=[
                self.robot_state_0.body_qd,
                self.spring_state_0.body_qd,
                self._robot_to_spring_robot_body_idx_wp,
                self._robot_to_spring_spring_body_idx_wp,
            ],
            device=self.spring_model.device,
        )

    def _init_spring_mesh(self, args: argparse.Namespace) -> None:
        if not self.mesh_enabled:
            return
        points = self.spring_state_0.particle_q.numpy()[self.spring_object_indices]
        self.mesh_point_count = int(points.shape[0])
        try:
            tri = triangulate_points(
                points,
                method=str(args.mesh_method),
                max_edge=args.mesh_max_edge,
                edge_factor=float(args.mesh_edge_factor),
            )
        except Exception as exc:
            LOGGER.warning("Mesh triangulation failed (%s). Falling back to points.", exc)
            self.mesh_enabled = False
            return
        if tri.size == 0:
            LOGGER.warning("Mesh triangulation produced no faces. Falling back to points.")
            self.mesh_enabled = False
            return
        tri_flat = tri.reshape(-1).astype(np.int32)
        self.mesh_indices_wp = wp.array(tri_flat.tolist(), dtype=wp.int32, device=self.spring_model.device)

    def _frame_camera(self, frame_on: str) -> None:
        if not hasattr(self.viewer, "set_camera"):
            return
        positions = None
        mode = (frame_on or "auto").lower()
        if mode in ("robot", "auto"):
            body_np = self.robot_state_0.body_q.numpy()
            if body_np.size:
                positions = body_np[:, :3]
        if positions is None and mode in ("particles", "auto"):
            positions = self.spring_model.particle_q.numpy()
        if positions is None or positions.size == 0:
            return
        center = positions.mean(axis=0)
        extents = positions.max(axis=0) - positions.min(axis=0)
        radius = float(np.linalg.norm(extents) * 0.5)
        dist = max(radius * 3.0, 1.0)
        pos = center + np.array([0.0, -dist, dist], dtype=np.float32)
        front = center - pos
        norm = np.linalg.norm(front)
        if norm <= 1e-6:
            return
        front /= norm
        pitch = float(np.degrees(np.arcsin(front[2])))
        yaw = float(np.degrees(np.arctan2(front[1], front[0])))
        self.viewer.set_camera(wp.vec3(*pos.tolist()), pitch, yaw)

    def _init_controller_drag(self, args: argparse.Namespace) -> None:
        self.controller_drag_enabled = bool(args.enable_controller_drag)
        self.controller_drag_stiffness = float(args.controller_drag_stiffness)
        self.controller_drag_damping = float(args.controller_drag_damping)
        self.controller_drag_mode = str(args.controller_drag_mode).lower()
        if self.controller_drag_mode not in ("force", "kinematic"):
            raise ValueError(f"Unsupported controller drag mode: {self.controller_drag_mode}")

        self.controller_count = int(self.spring_controller_indices.size)
        self.controller_indices_wp = None
        self.controller_targets_wp = None
        if self.controller_count > 0:
            self.controller_indices_wp = wp.array(
                self.spring_controller_indices.tolist(),
                dtype=int,
                device=self.spring_model.device,
            )
            self.controller_targets_wp = wp.array(
                [wp.vec3(0.0, 0.0, 0.0)] * self.controller_count,
                dtype=wp.vec3,
                device=self.spring_model.device,
            )

        self.controller_dragging = False
        self.controller_drag_depth = None
        self.controller_drag_anchor = None
        self.controller_drag_base = None
        self.controller_drag_targets = None

        self.drag_target_color = wp.vec3(1.0, 0.3, 0.9)
        self.drag_line_color = wp.vec3(1.0, 0.6, 0.2)
        self._empty_vec3 = wp.array([], dtype=wp.vec3, device=self.spring_model.device)
        self._empty_float = wp.array([], dtype=float, device=self.spring_model.device)

        if not self.controller_drag_enabled:
            return
        if self.controller_count == 0:
            LOGGER.warning("Controller drag enabled but no controller particles exist.")
            return
        if not hasattr(self.viewer, "renderer") or not hasattr(self.viewer, "camera"):
            return
        renderer = self.viewer.renderer
        if not hasattr(renderer, "register_mouse_press"):
            return
        renderer.register_mouse_press(self._on_mouse_press)
        renderer.register_mouse_release(self._on_mouse_release)
        renderer.register_mouse_drag(self._on_mouse_drag)

    def _to_framebuffer_coords(self, x: float, y: float) -> tuple[float, float]:
        fb_w, fb_h = self.viewer.renderer.window.get_framebuffer_size()
        win_w, win_h = self.viewer.renderer.window.get_size()
        if win_w <= 0 or win_h <= 0:
            return float(x), float(y)
        scale_x = fb_w / win_w
        scale_y = fb_h / win_h
        return float(x) * scale_x, float(y) * scale_y

    def _start_controller_drag(self, ray_start_np: np.ndarray, ray_dir_np: np.ndarray) -> None:
        if self.controller_count == 0:
            return
        positions = self.spring_state_0.particle_q.numpy()[self.spring_controller_indices]
        if positions.size == 0:
            return
        center = positions.mean(axis=0)
        d = ray_dir_np / max(np.linalg.norm(ray_dir_np), 1.0e-8)
        depth = float(np.dot(center - ray_start_np, d))
        if depth < 0.0:
            depth = 0.0
        anchor = ray_start_np + d * depth
        self.controller_drag_depth = depth
        self.controller_drag_anchor = anchor
        self.controller_drag_base = positions.copy()
        self.controller_dragging = True
        self._update_controller_drag(ray_start_np, ray_dir_np)

    def _update_controller_drag(self, ray_start_np: np.ndarray, ray_dir_np: np.ndarray) -> None:
        if not self.controller_dragging:
            return
        if self.controller_drag_depth is None or self.controller_drag_anchor is None:
            return
        if self.controller_drag_base is None:
            return
        d = ray_dir_np / max(np.linalg.norm(ray_dir_np), 1.0e-8)
        target = ray_start_np + d * float(self.controller_drag_depth)
        delta = target - self.controller_drag_anchor
        self.controller_drag_targets = self.controller_drag_base + delta
        if self.controller_targets_wp is not None:
            self.controller_targets_wp.assign([wp.vec3(*pt.tolist()) for pt in self.controller_drag_targets])

    def _on_mouse_press(self, x, y, button, modifiers):
        if not self.controller_drag_enabled:
            return
        try:
            import pyglet  # noqa: PLC0415
        except Exception:
            return
        if button == pyglet.window.mouse.RIGHT and (modifiers & pyglet.window.key.MOD_SHIFT):
            fb_x, fb_y = self._to_framebuffer_coords(x, y)
            ray_start, ray_dir = self.viewer.camera.get_world_ray(fb_x, fb_y)
            ray_start_np = np.array([ray_start.x, ray_start.y, ray_start.z], dtype=np.float32)
            ray_dir_np = np.array([ray_dir.x, ray_dir.y, ray_dir.z], dtype=np.float32)
            self._start_controller_drag(ray_start_np, ray_dir_np)

    def _on_mouse_release(self, x, y, button, modifiers):
        if self.controller_dragging:
            self.controller_dragging = False

    def _on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if not self.controller_drag_enabled or not self.controller_dragging:
            return
        try:
            import pyglet  # noqa: PLC0415
        except Exception:
            return
        if not (buttons & pyglet.window.mouse.RIGHT):
            return
        fb_x, fb_y = self._to_framebuffer_coords(x, y)
        ray_start, ray_dir = self.viewer.camera.get_world_ray(fb_x, fb_y)
        ray_start_np = np.array([ray_start.x, ray_start.y, ray_start.z], dtype=np.float32)
        ray_dir_np = np.array([ray_dir.x, ray_dir.y, ray_dir.z], dtype=np.float32)
        self._update_controller_drag(ray_start_np, ray_dir_np)

    def _update_controller_targets(self) -> None:
        if self.controller_indices_wp is None or self.controller_targets_wp is None:
            return
        if self.controller_count == 0:
            return
        if self.controller_drag_targets is None:
            return
        self.controller_targets_wp.assign([wp.vec3(*pt.tolist()) for pt in self.controller_drag_targets])
        if self.controller_drag_mode == "kinematic":
            wp.launch(
                set_controller_targets_kernel,
                dim=self.controller_count,
                inputs=[
                    self.spring_state_0.particle_q,
                    self.spring_state_0.particle_qd,
                    self.controller_indices_wp,
                    self.controller_targets_wp,
                ],
                device=self.spring_model.device,
            )
            return
        wp.launch(
            apply_controller_force_kernel,
            dim=self.controller_count,
            inputs=[
                self.spring_state_0.particle_q,
                self.spring_state_0.particle_qd,
                self.spring_state_0.particle_f,
                self.controller_indices_wp,
                self.controller_targets_wp,
                self.controller_drag_stiffness,
                self.controller_drag_damping,
            ],
            device=self.spring_model.device,
        )

    def _log_drag_marker(self) -> None:
        if not self.controller_drag_enabled or not self.controller_dragging:
            self.viewer.log_lines("/drag/line", self._empty_vec3, self._empty_vec3, self._empty_vec3)
            self.viewer.log_points("/drag/target", self._empty_vec3, self._empty_float, self._empty_vec3)
            return
        if self.controller_drag_targets is None or self.controller_drag_anchor is None:
            return
        center = self.controller_drag_targets.mean(axis=0)
        starts = wp.array([wp.vec3(*self.controller_drag_anchor.tolist())], dtype=wp.vec3, device=self.spring_model.device)
        ends = wp.array([wp.vec3(*center.tolist())], dtype=wp.vec3, device=self.spring_model.device)
        colors = wp.array([self.drag_line_color], dtype=wp.vec3, device=self.spring_model.device)
        self.viewer.log_lines("/drag/line", starts, ends, colors, hidden=False)

        point = wp.array([wp.vec3(*center.tolist())], dtype=wp.vec3, device=self.spring_model.device)
        radii = wp.full(1, float(self.object_radius_value) * 1.5, dtype=float, device=self.spring_model.device)
        point_colors = wp.array([self.drag_target_color], dtype=wp.vec3, device=self.spring_model.device)
        self.viewer.log_points("/drag/target", point, radii, point_colors, hidden=False)

    def step(self):
        for substep_idx in range(self.sim_substeps):
            # Robot step.
            self.robot_state_0.clear_forces()
            self.viewer.apply_forces(self.robot_state_0)
            current_time = self.sim_time + substep_idx * self.sim_dt
            self._robot_joint_target_pos_dirty = False
            self._update_right_gripper_autoclose(current_time)
            self._update_post_grip_arm_j1_sweep(current_time)
            if self._robot_joint_target_pos_dirty:
                self.robot_control.joint_target_pos.assign(self._robot_joint_target_pos_host)
            if self.robot_collisions_enabled:
                self.robot_contacts = self.robot_model.collide(
                    self.robot_state_0, collision_pipeline=self.robot_collision_pipeline
                )
            self.robot_solver.step(
                self.robot_state_0,
                self.robot_state_1,
                self.robot_control,
                self.robot_contacts,
                self.sim_dt,
            )
            self.robot_state_0, self.robot_state_1 = self.robot_state_1, self.robot_state_0

            # Spring-mass step.
            self.spring_state_0.clear_forces()
            self._update_controller_targets()
            self._sync_robot_to_spring_colliders()
            if self.spring_collisions_enabled:
                self.spring_contacts = self.spring_model.collide(
                    self.spring_state_0,
                    collision_pipeline=self.spring_collision_pipeline,
                )
            if self.contact_patch_enabled:
                if self._patch_refresh_counter % self.contact_patch_refresh_interval == 0:
                    self._refresh_contact_patch()
                self._apply_contact_patch_constraints()
                self._patch_refresh_counter += 1
            self.spring_solver.step(
                self.spring_state_0,
                self.spring_state_1,
                self.spring_control,
                self.spring_contacts,
                self.sim_dt,
            )
            self.spring_state_0, self.spring_state_1 = self.spring_state_1, self.spring_state_0

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.robot_state_0)
        if self.robot_contacts is not None:
            self.viewer.log_contacts(self.robot_contacts, self.robot_state_0)
        self._log_spring_points()
        self._log_drag_marker()
        self.viewer.end_frame()

    def _log_spring_points(self) -> None:
        positions = self.spring_state_0.particle_q.numpy()
        if positions.size == 0:
            return
        obj_points = positions[self.spring_object_indices]
        if self.visual_z_offset != 0.0:
            obj_points = obj_points.copy()
            obj_points[:, 2] += self.visual_z_offset
        points_wp = wp.array(obj_points.tolist(), dtype=wp.vec3, device=self.spring_model.device)
        if self.mesh_indices_wp is not None and obj_points.shape[0] == self.mesh_point_count:
            self.viewer.log_mesh(
                "/spring/mesh",
                points_wp,
                self.mesh_indices_wp,
                hidden=not self.mesh_enabled,
                backface_culling=False,
            )
        show_points = (not self.mesh_enabled) or self.mesh_show_points
        radii_wp = wp.full(
            obj_points.shape[0],
            self.object_radius_value,
            dtype=float,
            device=self.spring_model.device,
        )
        colors_wp = wp.full(obj_points.shape[0], wp.vec3(0.8, 0.7, 0.2), dtype=wp.vec3, device=self.spring_model.device)
        self.viewer.log_points("/spring/points", points_wp, radii_wp, colors_wp, hidden=not show_points)

        if self.spring_controller_indices.size:
            ctrl_points = positions[self.spring_controller_indices]
            if self.visual_z_offset != 0.0:
                ctrl_points = ctrl_points.copy()
                ctrl_points[:, 2] += self.visual_z_offset
            ctrl_wp = wp.array(ctrl_points.tolist(), dtype=wp.vec3, device=self.spring_model.device)
            ctrl_radii = wp.full(
                ctrl_points.shape[0],
                float(self.spring_model.particle_radius.numpy()[0]) * 1.25,
                dtype=float,
                device=self.spring_model.device,
            )
            ctrl_colors = wp.full(
                ctrl_points.shape[0],
                wp.vec3(0.2, 1.0, 0.2),
                dtype=wp.vec3,
                device=self.spring_model.device,
            )
            self.viewer.log_points("/spring/controllers", ctrl_wp, ctrl_radii, ctrl_colors, hidden=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = newton.examples.create_parser()
    parser.add_argument("--pkl", type=str, required=True, help="Path to spring-mass PKL file.")
    parser.add_argument("--urdf", type=str, required=True, help="Path to robot URDF file.")
    parser.add_argument(
        "--substeps",
        type=int,
        default=None,
        help="Simulation substeps (overrides PKL substeps when set).",
    )
    parser.add_argument(
        "--spring-soft-contact-ke",
        type=float,
        default=5.0e4,
        help="Soft-contact stiffness for spring particles against colliders.",
    )
    parser.add_argument(
        "--spring-soft-contact-kd",
        type=float,
        default=2.0e2,
        help="Soft-contact damping for spring particles against colliders.",
    )
    parser.add_argument(
        "--spring-soft-contact-kf",
        type=float,
        default=2.0e3,
        help="Soft-contact tangential friction stiffness for spring particles against colliders.",
    )
    parser.add_argument(
        "--spring-soft-contact-margin",
        type=float,
        default=None,
        help=(
            "Soft-contact detection margin for spring collide(). "
            "If unset, uses max(0.01, 0.75 * particle_radius)."
        ),
    )
    parser.add_argument("--visual-z-offset", type=float, default=0.001, help="Render-only Z offset.")
    parser.add_argument(
        "--robot-solver",
        type=str,
        default="mujoco",
        choices=("semi_implicit", "mujoco", "vbd"),
        help="Solver for the robot model.",
    )
    parser.add_argument(
        "--spring-solver",
        type=str,
        default="semi_implicit",
        choices=("semi_implicit", "mujoco", "vbd"),
        help="Solver for the spring-mass model.",
    )
    parser.add_argument(
        "--mujoco-nconmax",
        type=int,
        default=None,
        help="Override MuJoCo contact capacity (nconmax) for SolverMuJoCo.",
    )
    parser.add_argument(
        "--mujoco-njmax",
        type=int,
        default=None,
        help="Override MuJoCo constraint capacity (njmax) for SolverMuJoCo.",
    )
    parser.add_argument(
        "--robot-joint-ke",
        type=float,
        default=2.0e4,
        help="Global robot joint position stiffness (PD P gain) for holding pose against gravity.",
    )
    parser.add_argument(
        "--robot-joint-kd",
        type=float,
        default=5.0e2,
        help="Global robot joint damping (PD D gain) for holding pose against gravity.",
    )
    parser.add_argument(
        "--urdf-offset",
        type=float,
        nargs=3,
        default=None,
        help="XYZ offset applied to the robot root body.",
    )
    parser.add_argument(
        "--enable-controller-drag",
        action="store_true",
        help="Enable Shift+Right drag to move spring controller points as a group.",
    )
    parser.add_argument(
        "--disable-pkl-controllers",
        action="store_true",
        help="Ignore controller springs specified by PKL params (unless controller drag explicitly needs them).",
    )
    parser.add_argument(
        "--mesh-show-points",
        action="store_true",
        help="Keep rendering object points when mesh visualization is enabled.",
    )
    parser.add_argument(
        "--spring-offset",
        type=float,
        nargs=3,
        default=None,
        help="XYZ offset applied to spring-mass particles at initialization.",
    )
    parser.add_argument(
        "--particle-mass-override",
        type=float,
        default=None,
        help="Override object particle mass at runtime (takes precedence over PKL params).",
    )
    parser.add_argument(
        "--robot-pose-preset",
        type=str,
        default="none",
        choices=("none", "forward_reach"),
        help="Named robot initial pose preset.",
    )
    parser.add_argument(
        "--active-arm",
        type=str,
        default="right",
        choices=("left", "right"),
        help="Which arm to control with --arm-j* values.",
    )
    parser.add_argument("--lift-joint", type=float, default=None, help="Optional lift_joint target.")
    parser.add_argument("--arm-j1", type=float, default=None, help="Active-arm joint1 target.")
    parser.add_argument("--arm-j2", type=float, default=None, help="Active-arm joint2 target.")
    parser.add_argument("--arm-j3", type=float, default=None, help="Active-arm joint3 target.")
    parser.add_argument("--arm-j4", type=float, default=None, help="Active-arm joint4 target.")
    parser.add_argument("--arm-j5", type=float, default=None, help="Active-arm joint5 target.")
    parser.add_argument("--arm-j6", type=float, default=None, help="Active-arm joint6 target.")
    parser.add_argument("--arm-j7", type=float, default=None, help="Active-arm joint7 target.")
    parser.add_argument(
        "--print-arm-joint-ranges",
        action="store_true",
        help="Print active-arm joint limits from the loaded URDF/model.",
    )
    parser.add_argument(
        "--auto-close-right-gripper",
        action="store_true",
        help="Automatically close right gripper after spawn.",
    )
    parser.add_argument(
        "--gripper-close-start-time",
        type=float,
        default=0.5,
        help="Seconds after start before right gripper starts closing.",
    )
    parser.add_argument(
        "--gripper-close-duration",
        type=float,
        default=2.0,
        help="Seconds spent closing right gripper.",
    )
    parser.add_argument(
        "--gripper-close-target",
        type=float,
        default=0.9,
        help="Normalized close target in [0,1] for right gripper joints.",
    )
    parser.add_argument(
        "--auto-sweep-arm-j1-after-grip",
        action="store_true",
        help="After gripper closes, sweep active arm joint1 toward target.",
    )
    parser.add_argument(
        "--arm-j1-sweep-target",
        type=float,
        default=-3.0,
        help="Target value for active arm joint1 after gripper close.",
    )
    parser.add_argument(
        "--arm-j1-sweep-duration",
        type=float,
        default=2.0,
        help="Seconds to sweep active arm joint1 after gripper close.",
    )
    parser.add_argument(
        "--enable-contact-patch",
        action="store_true",
        help="Enable constraint-based contact patch between selected gripper and spring particles.",
    )
    parser.add_argument(
        "--contact-patch-arm",
        type=str,
        default="right",
        choices=("left", "right"),
        help="Gripper side used to acquire and maintain contact patch constraints.",
    )
    parser.add_argument(
        "--contact-patch-max-particles",
        type=int,
        default=48,
        help="Maximum number of spring particles kept in a contact patch.",
    )
    parser.add_argument(
        "--contact-patch-min-contacts",
        type=int,
        default=6,
        help="Minimum unique contacts needed to start a new patch.",
    )
    parser.add_argument(
        "--contact-patch-normal-ke",
        type=float,
        default=1.2e5,
        help="Normal-direction patch stiffness.",
    )
    parser.add_argument(
        "--contact-patch-normal-kd",
        type=float,
        default=2.0e3,
        help="Normal-direction patch damping.",
    )
    parser.add_argument(
        "--contact-patch-tangent-ke",
        type=float,
        default=6.0e4,
        help="Tangential stick stiffness before slip limiting.",
    )
    parser.add_argument(
        "--contact-patch-tangent-kd",
        type=float,
        default=1.0e3,
        help="Tangential stick damping before slip limiting.",
    )
    parser.add_argument(
        "--contact-patch-mu",
        type=float,
        default=2.5,
        help="Patch tangential force limit scale (larger keeps sticking stronger).",
    )
    parser.add_argument(
        "--contact-patch-max-force",
        type=float,
        default=4.0e3,
        help="Per-particle clamp for total patch force.",
    )
    parser.add_argument(
        "--contact-patch-max-gap",
        type=float,
        default=0.0,
        help="Max particle-surface gap for patch acquisition (0 means touching/penetrating only).",
    )
    parser.add_argument(
        "--contact-patch-break-distance",
        type=float,
        default=0.04,
        help="Release patch particle when anchor error exceeds this distance.",
    )
    parser.add_argument(
        "--contact-patch-release-missed-steps",
        type=int,
        default=12,
        help="Release patch particle after this many substeps without supporting contacts.",
    )
    parser.add_argument(
        "--contact-patch-refresh-interval",
        type=int,
        default=8,
        help="Run patch contact acquisition every N substeps.",
    )
    parser.add_argument(
        "--contact-patch-log-interval",
        type=int,
        default=0,
        help="Substep interval for logging patch size (0 disables).",
    )
    parser.add_argument("--table-center-x", type=float, default=TABLE_CENTER_X, help="Table center X.")
    parser.add_argument("--table-center-y", type=float, default=TABLE_CENTER_Y, help="Table center Y.")
    parser.add_argument("--table-center-z", type=float, default=TABLE_CENTER_Z, help="Table center Z (height).")
    parser.add_argument("--table-half-x", type=float, default=TABLE_HALF_X, help="Table half size X.")
    parser.add_argument("--table-half-y", type=float, default=TABLE_HALF_Y, help="Table half size Y.")
    parser.add_argument("--table-half-z", type=float, default=TABLE_HALF_Z, help="Table half size Z (thickness/height).")
    parser.add_argument(
        "--table-cloth-clearance",
        type=float,
        default=TABLE_CLOTH_CLEARANCE,
        help="Minimum initial cloth clearance above table top.",
    )

    viewer, args = newton.examples.init(parser)
    # Internal defaults for options no longer exposed in CLI.
    hidden_defaults = {
        "solver": None,
        "robot_solver": "mujoco",
        "spring_solver": "semi_implicit",
        "no_ground": False,
        "enable_robot_collisions": False,
        "enable_spring_collisions": True,
        "spring_soft_contact_mu": 0.985,
        "controller_drag_mode": "force",
        "controller_drag_stiffness": 2.0e4,
        "controller_drag_damping": 1.0e2,
        "frame_on": "robot",
        "visualize_mesh": False,
        "mesh_method": "pca_delaunay",
        "mesh_max_edge": None,
        "mesh_edge_factor": 2.5,
        "mujoco_nconmax": None,
        "mujoco_njmax": None,
    }
    for key, value in hidden_defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if args.solver:
        args.robot_solver = args.solver
    example = Example(viewer, args)
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
