from __future__ import annotations

"""Newton-based trainer for spring-mass inverse physics (gradient-based)."""

import csv
import logging
import os
import pickle
import time
from typing import Any

import numpy as np
import warp as wp
import yaml

import newton
import newton.examples

from phystwin.mapping.pkl_mapping import SpringMassPKL, SpringMassPKLPair, load_pkl, map_pkl_to_newton


LOGGER = logging.getLogger("phystwin.trainer")


@wp.func
def smooth_l1(x: float) -> float:
    ax = wp.abs(x)
    return wp.where(ax < 1.0, 0.5 * x * x, ax - 0.5)


@wp.kernel
def track_loss_kernel(
    pred_q: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    valid: wp.array(dtype=int),
    frame_offset: int,
    denom: float,
    loss: wp.array(dtype=float),
):
    tid = wp.tid()
    idx = frame_offset + tid
    if valid[idx] == 0:
        return
    diff = pred_q[tid] - gt[idx]
    val = smooth_l1(diff[0]) + smooth_l1(diff[1]) + smooth_l1(diff[2])
    wp.atomic_add(loss, 0, val / denom)


@wp.kernel(enable_backward=False)
def set_controller_points_kernel(
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    controller_indices: wp.array(dtype=int),
    controller_targets: wp.array(dtype=wp.vec3),
    dt: float,
):
    tid = wp.tid()
    idx = controller_indices[tid]
    target = controller_targets[tid]
    prev = q[idx]
    q[idx] = target
    qd[idx] = (target - prev) / dt


@wp.kernel
def apply_drag_kernel(qd: wp.array(dtype=wp.vec3), factor: float):
    tid = wp.tid()
    qd[tid] = qd[tid] * factor


@wp.kernel
def apply_ground_penalty_kernel(
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    count: int,
    ground_height: float,
    k: float,
    kd: float,
):
    tid = wp.tid()
    if tid >= count:
        return
    x = q[tid]
    if x[2] < ground_height:
        penetration = ground_height - x[2]
        f[tid] += wp.vec3(0.0, 0.0, k * penetration - kd * qd[tid][2])


@wp.kernel
def adam_update_kernel(
    param: wp.array(dtype=float),
    grad: wp.array(dtype=float),
    m: wp.array(dtype=float),
    v: wp.array(dtype=float),
    lr: float,
    beta1: float,
    beta2: float,
    bias_correction1: float,
    bias_correction2: float,
    eps: float,
    grad_clip: float,
    min_val: float,
    max_val: float,
):
    tid = wp.tid()
    g = grad[tid]
    if grad_clip > 0.0:
        g = wp.clamp(g, -grad_clip, grad_clip)
    m_new = beta1 * m[tid] + (1.0 - beta1) * g
    v_new = beta2 * v[tid] + (1.0 - beta2) * g * g
    m_hat = m_new / bias_correction1
    v_hat = v_new / bias_correction2
    val = param[tid] - lr * m_hat / (wp.sqrt(v_hat) + eps)
    if val < min_val:
        val = min_val
    if val > max_val:
        val = max_val
    param[tid] = val
    m[tid] = m_new
    v[tid] = v_new


@wp.kernel
def exp_param_kernel(
    log_param: wp.array(dtype=float),
    min_val: float,
    max_val: float,
    out: wp.array(dtype=float),
):
    """Convert log parameters to actual parameters with clamping.

    This kernel supports automatic differentiation because it uses
    Warp's built-in wp.exp() and wp.clamp() functions.
    """
    tid = wp.tid()
    val = wp.exp(log_param[tid])
    val = wp.clamp(val, min_val, max_val)
    out[tid] = val


class InvPhyTrainerWarp:
    """Newton-based inverse physics trainer using gradient descent."""

    def __init__(
        self,
        data_path: str,
        base_dir: str,
        train_frame: int,
        config: dict[str, Any],
        device: str = "cuda:0",
        monitor: bool = False,
        monitor_frames: int = 0,
        monitor_every: int = 1,
        initial_params: dict[str, Any] | None = None,
    ) -> None:
        self.data_path = data_path
        self.base_dir = base_dir
        self.train_frame = train_frame
        self.config = config
        self.device = device
        self.monitor = monitor
        self.monitor_frames = monitor_frames
        self.monitor_every = max(1, monitor_every)
        self.initial_params = initial_params or {}
        if self.initial_params:
            self.config.update(self.initial_params)

        wp.init()
        try:
            wp.set_device(device)
        except Exception:
            LOGGER.warning("Failed to set Warp device %s; using default", device)

        os.makedirs(os.path.join(self.base_dir, "train"), exist_ok=True)

        data = load_pkl(data_path)
        if isinstance(data, SpringMassPKLPair):
            self.predict_data = data.predict
            self.gt_data = data.gt
        else:
            self.predict_data = data
            self.gt_data = data

        self.predict_frames = int(self.predict_data.object_points.shape[0])
        self.gt_frames = int(self.gt_data.object_points.shape[0])
        self.num_frames = min(self.predict_frames, self.gt_frames) if self.gt_frames else self.predict_frames
        self.train_frame = min(self.train_frame, self.num_frames)

        self.eval_count = 0
        self.log_every = int(self.config.get("log_every", 1))
        self.monitor_viewer = None

        self.compare_axis = self.config.get("compare_axis", "x")
        self.compare_offset = self.config.get("compare_offset", None)
        self.compare_symmetric = bool(self.config.get("compare_symmetric", False))
        self.gt_color = self._parse_color(self.config.get("gt_color"), (0.2, 0.6, 1.0))
        self.gt_controller_color = self._parse_color(self.config.get("gt_controller_color"), (1.0, 0.4, 0.4))
        self.predict_color = self._parse_color(self.config.get("predict_color"), (0.8, 0.7, 0.2))
        self.predict_controller_color = self._parse_color(
            self.config.get("predict_controller_color"), (0.2, 1.0, 0.2)
        )

        self._setup_sim()
        LOGGER.info(
            "Setup complete | frames=%d | object_count=%d | substeps=%d | dt=%.6e | train_params=%s",
            self.train_frame,
            self.object_count,
            self.substeps,
            self.dt,
            sorted(self.train_params),
        )
        self._init_logger()

    def _init_logger(self) -> None:
        self.log_path = os.path.join(self.base_dir, "train", "train_log.csv")
        self._log_file = open(self.log_path, "w", encoding="utf-8", newline="")
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow(
            [
                "iter",
                "loss",
                "elapsed",
                "q_bad_pre",
                "qd_bad_pre",
                "q_max_abs",
                "q_max_norm",
                "q_min",
                "q_max",
                "qd_max_abs",
                "qd_max_norm",
                "qd_min",
                "qd_max",
                "q_bad",
                "qd_bad",
                "spring_ke_bad",
                "spring_ke_gbad",
                "spring_kd_bad",
                "spring_kd_gbad",
                "spring_ke_min",
                "spring_ke_max",
                "spring_ke_mean",
                "spring_ke_gmin",
                "spring_ke_gmax",
                "spring_ke_gmean",
                "spring_kd_min",
                "spring_kd_max",
                "spring_kd_mean",
                "spring_kd_gmin",
                "spring_kd_gmax",
                "spring_kd_gmean",
            ]
        )

    def _apply_scale(self, points: np.ndarray, scale: float, reverse_z: bool) -> np.ndarray:
        scaled = points.astype(np.float32) * scale
        if reverse_z and scaled.size:
            scaled[..., 2] *= -1.0
        return scaled

    def _compute_z_shift(self, points: np.ndarray, z_offset: float) -> float:
        if points.size == 0:
            return z_offset
        min_z = float(points[:, 2].min())
        if min_z <= 0.0:
            return -min_z + z_offset
        return z_offset

    def _apply_scale_z(self, points: np.ndarray, scale: float, z_shift: float, reverse_z: bool) -> np.ndarray:
        scaled = self._apply_scale(points, scale, reverse_z)
        if scaled.size:
            scaled[..., 2] += z_shift
        return scaled

    def _object_mask(self, data: SpringMassPKL, frame: int) -> np.ndarray:
        mask = np.ones(data.object_points.shape[1], dtype=bool)
        if self.config.get("filter_visibility") and data.object_visibilities.size:
            mask &= data.object_visibilities[frame].astype(bool)
        if self.config.get("filter_motion_valid") and data.object_motions_valid.size:
            mask &= data.object_motions_valid[frame].astype(bool)
        return mask

    def _frame_valid_mask(self, data: SpringMassPKL, frame: int, indices: np.ndarray) -> np.ndarray:
        valid = np.ones(indices.shape[0], dtype=bool)
        if self.config.get("filter_visibility") and data.object_visibilities.size:
            valid &= data.object_visibilities[frame][indices].astype(bool)
        if self.config.get("filter_motion_valid") and data.object_motions_valid.size:
            valid &= data.object_motions_valid[frame][indices].astype(bool)
        return valid

    def _parse_color(self, value: Any, default: tuple[float, float, float]) -> np.ndarray:
        if value is None:
            return np.array(default, dtype=np.float32)
        if isinstance(value, str):
            parts = value.replace(",", " ").split()
            if len(parts) == 3:
                return np.array([float(p) for p in parts], dtype=np.float32)
            return np.array(default, dtype=np.float32)
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 3:
            return np.array(value, dtype=np.float32)
        return np.array(default, dtype=np.float32)

    def _compute_visual_offsets(self) -> tuple[np.ndarray, np.ndarray]:
        offset = self.compare_offset
        if offset is None:
            offset = float(self.config.get("compare_offset", 0.0))
        offset = float(offset) if offset is not None else 0.0
        if offset <= 0.0:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        if self.compare_axis == "z":
            axis_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        elif self.compare_axis == "y":
            axis_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            axis_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        if self.compare_symmetric:
            half = 0.5 * offset
            gt_offset = axis_vec * half
            pred_offset = -axis_vec * half
        else:
            gt_offset = axis_vec * offset
            pred_offset = np.zeros(3, dtype=np.float32)
        return gt_offset, pred_offset

    def _apply_contact_params(self, model: newton.Model) -> None:
        base_ke = float(self.config.get("particle_contact_ke", 1.0e3))
        base_kd = float(self.config.get("particle_contact_kd", 1.0e2))
        base_kf = float(self.config.get("particle_contact_kf", 1.0e2))

        collide_elas = float(self.config.get("collide_elas", 1.0))
        collide_fric = float(self.config.get("collide_fric", 0.5))

        model.particle_ke = base_ke * max(collide_elas, 0.0)
        model.particle_kd = base_kd
        model.particle_kf = base_kf
        model.particle_mu = max(collide_fric, 0.0)

        if getattr(model, "shape_count", 0) and model.shape_material_ke is not None:
            shape_count = int(model.shape_material_ke.shape[0])
            obj_elas = float(self.config.get("collide_object_elas", 1.0))
            obj_fric = float(self.config.get("collide_object_fric", 0.5))
            model.shape_material_ke.assign([base_ke * max(obj_elas, 0.0)] * shape_count)
            if model.shape_material_kd is not None:
                model.shape_material_kd.assign([base_kd] * shape_count)
            if model.shape_material_kf is not None:
                model.shape_material_kf.assign([base_kf] * shape_count)
            if model.shape_material_mu is not None:
                model.shape_material_mu.assign([max(obj_fric, 0.0)] * shape_count)

    def _setup_sim(self) -> None:
        scale = float(self.config.get("scale", 1.0))
        z_offset = float(self.config.get("z_offset", 0.0))
        reverse_z = bool(self.config.get("reverse_z", False))
        self.fps = float(self.config.get("FPS", 30.0))
        self.substeps = int(self.config.get("substeps", self.config.get("num_substeps", 1)))
        self.dt = 1.0 / self.fps / self.substeps

        use_controllers = bool(self.config.get("use_controllers", True))
        particle_mass = float(self.config.get("particle_mass", 1.0))
        add_ground = not bool(self.config.get("no_ground", False))
        self.drag_damping = float(self.config.get("drag_damping", 0.0))
        self.enable_ground_penalty = bool(self.config.get("enable_ground_penalty", True))
        self.ground_height = float(self.config.get("ground_height", 0.0))
        self.ground_k = float(self.config.get("ground_k", 1.0e4))
        self.ground_kd = float(self.config.get("ground_kd", 1.0e2))

        frame0 = 0
        predict_mask = self._object_mask(self.predict_data, frame0)
        predict_points = self.predict_data.object_points[frame0][predict_mask]
        scaled_predict = self._apply_scale(predict_points, scale, reverse_z)
        z_shift = self._compute_z_shift(scaled_predict, z_offset)

        neighbor_mode = str(self.config.get("spring_neighbor_mode", "knn")).lower()
        object_radius = self.config.get("object_radius", None)
        object_max_neighbours = self.config.get("object_max_neighbours", None)
        controller_radius = self.config.get("controller_radius", None)
        controller_max_neighbours = self.config.get("controller_max_neighbours", None)
        k_neighbors = self.config.get("k_neighbors", 6)
        controller_k = self.config.get("controller_k", 1)

        spring_ke = float(self.config.get("spring_ke", 3.0e4))
        spring_kd = float(self.config.get("spring_kd", 100.0))
        particle_radius = float(self.config.get("particle_radius", 0.02))

        mapping = map_pkl_to_newton(
            data=self.predict_data,
            frame=frame0,
            scale=scale,
            z_offset=z_offset,
            reverse_z=reverse_z,
            particle_radius=particle_radius,
            mass=particle_mass,
            spring_ke=spring_ke,
            spring_kd=spring_kd,
            k_neighbors=int(k_neighbors),
            add_ground=add_ground,
            use_controllers=use_controllers,
            controller_k=int(controller_k),
            controller_ke=spring_ke,
            controller_kd=spring_kd,
            filter_visibility=bool(self.config.get("filter_visibility", False)),
            filter_motion_valid=bool(self.config.get("filter_motion_valid", False)),
            spring_neighbor_mode=neighbor_mode,
            object_radius=object_radius,
            object_max_neighbours=object_max_neighbours,
            controller_radius=controller_radius,
            controller_max_neighbours=controller_max_neighbours,
            requires_grad=True,
        )

        self.mapping = mapping
        self.model = mapping.model
        self.self_collision = bool(self.config.get("self_collision", True))
        self.self_collision_grad = bool(self.config.get("self_collision_grad", False))
        if not self.self_collision or not self.self_collision_grad:
            if self.self_collision and not self.self_collision_grad:
                LOGGER.warning(
                    "Self-collision gradients disabled; set self_collision_grad: true to enable."
                )
            self.model.particle_grid = None

        if self.model.spring_rest_length is not None:
            rest_np = self.model.spring_rest_length.numpy()
            if rest_np.size:
                min_rest = float(rest_np.min())
                LOGGER.info("Spring rest length min=%.3e", min_rest)
        self._apply_contact_params(self.model)

        self.solver = newton.solvers.SolverSemiImplicit(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.enable_collisions = bool(self.config.get("enable_collisions", False))
        if self.enable_collisions:
            self.collision_pipeline = newton.examples.create_collision_pipeline(
                self.model, collision_pipeline_type="standard"
            )
        else:
            self.collision_pipeline = None

        self.object_count = mapping.object_points.shape[0]
        self.pkl_object_indices = mapping.pkl_object_indices

        # Cache initial state to reset between iterations
        self.initial_q = wp.array(self.state_0.particle_q.numpy(), dtype=wp.vec3, device=self.model.device)
        self.initial_qd = wp.array(self.state_0.particle_qd.numpy(), dtype=wp.vec3, device=self.model.device)

        # Prepare GT points and valid mask for loss
        gt_points = []
        valid_mask = []
        for frame in range(self.train_frame):
            pts = self.gt_data.object_points[frame][self.pkl_object_indices]
            pts = self._apply_scale_z(pts, scale, z_shift, reverse_z)
            gt_points.append(pts)
            valid_mask.append(self._frame_valid_mask(self.gt_data, frame, self.pkl_object_indices))

        gt_points = np.asarray(gt_points, dtype=np.float32)
        valid_mask = np.asarray(valid_mask, dtype=np.int32)
        self.frame_valid_counts = valid_mask.sum(axis=1).astype(np.int32)
        self.frame_loss_denoms = np.maximum(self.frame_valid_counts, 1).astype(np.float32) * 3.0

        self.gt_points_wp = wp.array(gt_points.reshape(-1, 3), dtype=wp.vec3, device=self.model.device)
        self.valid_wp = wp.array(valid_mask.reshape(-1), dtype=wp.int32, device=self.model.device)

        # Controller targets
        ctrl_points = self.gt_data.controller_points
        if ctrl_points.size == 0 and self.predict_data.controller_points.size:
            ctrl_points = self.predict_data.controller_points
        if ctrl_points.size:
            ctrl_points = ctrl_points[: self.train_frame]
            ctrl_points = self._apply_scale_z(ctrl_points, scale, z_shift, reverse_z)
        else:
            ctrl_points = np.zeros((self.train_frame, 0, 3), dtype=np.float32)
        self.ctrl_points = ctrl_points

        controller_indices = mapping.controller_particle_indices.astype(np.int32)
        self.controller_count = min(controller_indices.shape[0], ctrl_points.shape[1])
        if use_controllers and self.controller_count > 0:
            controller_indices = controller_indices[: self.controller_count]
            self.controller_indices_wp = wp.array(
                controller_indices.tolist(), dtype=int, device=self.model.device
            )
            self.controller_targets_wp = wp.array(
                [wp.vec3(0.0, 0.0, 0.0)] * self.controller_count,
                dtype=wp.vec3,
                device=self.model.device,
            )
        else:
            self.controller_indices_wp = None
            self.controller_targets_wp = None

        # Trainable params
        train_params = self.config.get("train_params", ["spring_ke", "spring_kd"])
        self.train_params = {str(p) for p in train_params}
        self.train_lr = float(self.config.get("base_lr", 1.0e-3))
        self.adam_beta1 = float(self.config.get("adam_beta1", 0.9))
        self.adam_beta2 = float(self.config.get("adam_beta2", 0.99))
        self.adam_eps = float(self.config.get("adam_eps", 1.0e-8))
        self.adam_step = 0
        self.grad_clip = float(self.config.get("grad_clip", 0.0))
        self.use_log_params = bool(self.config.get("log_params", True))
        self.log_param_stats = bool(self.config.get("log_param_stats", True))
        self.debug_nan_check = bool(self.config.get("debug_nan_check", True))

        LOGGER.info("use_log_params=%s", self.use_log_params)
        if self.model.spring_stiffness is not None:
            LOGGER.info("spring_stiffness.requires_grad=%s", self.model.spring_stiffness.requires_grad)
        if self.model.spring_damping is not None:
            LOGGER.info("spring_damping.requires_grad=%s", self.model.spring_damping.requires_grad)
        self.debug_nan_every = int(self.config.get("debug_nan_every", 1))
        self.debug_nan_break = bool(self.config.get("debug_nan_break", True))
        self.debug_nan_param_check = bool(self.config.get("debug_nan_param_check", True))
        self.skip_update_on_nan_grad = bool(self.config.get("skip_update_on_nan_grad", True))

        self.spring_ke_min = float(self.config.get("spring_ke_min", 0.0))
        self.spring_ke_max = float(self.config.get("spring_ke_max", 1.0e6))
        self.spring_kd_min = float(self.config.get("spring_kd_min", 0.0))
        self.spring_kd_max = float(self.config.get("spring_kd_max", 1.0e3))
        self.spring_ke_log_min = None
        self.spring_ke_log_max = None
        self.spring_kd_log_min = None
        self.spring_kd_log_max = None
        if self.use_log_params:
            ke_min = max(self.spring_ke_min, 1.0e-8)
            ke_max = max(self.spring_ke_max, ke_min * 1.0e-2)
            kd_min = max(self.spring_kd_min, 1.0e-8)
            kd_max = max(self.spring_kd_max, kd_min * 1.0e-2)
            self.spring_ke_log_min = float(np.log(ke_min))
            self.spring_ke_log_max = float(np.log(ke_max))
            self.spring_kd_log_min = float(np.log(kd_min))
            self.spring_kd_log_max = float(np.log(kd_max))

        self.spring_ke_m = None
        self.spring_ke_v = None
        self.spring_kd_m = None
        self.spring_kd_v = None
        self.spring_ke_log = None
        self.spring_kd_log = None
        if self.use_log_params:
            if "spring_ke" in self.train_params and self.model.spring_stiffness is not None:
                ke_np = self.model.spring_stiffness.numpy()
                ke_np = np.maximum(ke_np, max(self.spring_ke_min, 1.0e-8))
                self.spring_ke_log = wp.array(
                    np.log(ke_np).astype(np.float32),
                    dtype=float,
                    device=self.model.device,
                    requires_grad=True,
                )
                self.spring_ke_m = wp.zeros(
                    self.model.spring_stiffness.shape[0], dtype=float, device=self.model.device
                )
                self.spring_ke_v = wp.zeros(
                    self.model.spring_stiffness.shape[0], dtype=float, device=self.model.device
                )
            if "spring_kd" in self.train_params and self.model.spring_damping is not None:
                kd_np = self.model.spring_damping.numpy()
                kd_np = np.maximum(kd_np, max(self.spring_kd_min, 1.0e-8))
                self.spring_kd_log = wp.array(
                    np.log(kd_np).astype(np.float32),
                    dtype=float,
                    device=self.model.device,
                    requires_grad=True,
                )
                self.spring_kd_m = wp.zeros(
                    self.model.spring_damping.shape[0], dtype=float, device=self.model.device
                )
                self.spring_kd_v = wp.zeros(
                    self.model.spring_damping.shape[0], dtype=float, device=self.model.device
                )
        else:
            if "spring_ke" in self.train_params and self.model.spring_stiffness is not None:
                self.spring_ke_m = wp.zeros(
                    self.model.spring_stiffness.shape[0], dtype=float, device=self.model.device
                )
                self.spring_ke_v = wp.zeros(
                    self.model.spring_stiffness.shape[0], dtype=float, device=self.model.device
                )
            if "spring_kd" in self.train_params and self.model.spring_damping is not None:
                self.spring_kd_m = wp.zeros(
                    self.model.spring_damping.shape[0], dtype=float, device=self.model.device
                )
                self.spring_kd_v = wp.zeros(
                    self.model.spring_damping.shape[0], dtype=float, device=self.model.device
                )

    def _stats(self, arr: wp.array | None) -> tuple[float | None, float | None, float | None]:
        if arr is None:
            return None, None, None
        data = arr.numpy()
        if data.size == 0:
            return None, None, None
        return float(data.min()), float(data.max()), float(data.mean())

    def _grad_stats(self, arr: wp.array | None) -> tuple[float | None, float | None, float | None]:
        if arr is None or arr.grad is None:
            return None, None, None
        data = arr.grad.numpy()
        if data.size == 0:
            return None, None, None
        return float(data.min()), float(data.max()), float(data.mean())

    def _finite_stats(self, arr: np.ndarray) -> dict[str, float | int | None]:
        if arr.size == 0:
            return {
                "max_abs": None,
                "max_norm": None,
                "min": None,
                "max": None,
                "bad": 0,
            }
        flat = arr.reshape(-1)
        finite_mask = np.isfinite(flat)
        bad = int(flat.size - int(finite_mask.sum()))
        if finite_mask.any():
            finite_vals = flat[finite_mask]
            max_abs = float(np.max(np.abs(finite_vals)))
            min_val = float(np.min(finite_vals))
            max_val = float(np.max(finite_vals))
        else:
            max_abs = None
            min_val = None
            max_val = None

        max_norm = None
        if arr.ndim == 2 and arr.shape[1] == 3:
            finite_rows = np.all(np.isfinite(arr), axis=1)
            if np.any(finite_rows):
                norms = np.linalg.norm(arr[finite_rows], axis=1)
                max_norm = float(np.max(norms)) if norms.size else None

        return {
            "max_abs": max_abs,
            "max_norm": max_norm,
            "min": min_val,
            "max": max_val,
            "bad": bad,
        }

    def _reset_states(self) -> None:
        wp.copy(self.state_0.particle_q, self.initial_q)
        wp.copy(self.state_0.particle_qd, self.initial_qd)
        wp.copy(self.state_1.particle_q, self.initial_q)
        wp.copy(self.state_1.particle_qd, self.initial_qd)

    def _step_frame(
        self,
        frame: int,
        loss: wp.array | None,
        monitor: bool,
        viewer,
        gt_offset: np.ndarray,
        pred_offset: np.ndarray,
    ) -> None:
        frame_index = frame - 1
        next_index = frame
        drag_factor = max(0.0, 1.0 - self.drag_damping * self.dt)

        if self.use_log_params:
            if self.spring_ke_log is not None and self.model.spring_stiffness is not None:
                wp.launch(
                    exp_param_kernel,
                    dim=self.model.spring_stiffness.shape[0],
                    inputs=[
                        self.spring_ke_log,
                        self.spring_ke_min,
                        self.spring_ke_max,
                    ],
                    outputs=[self.model.spring_stiffness],
                    device=self.model.device,
                )
            if self.spring_kd_log is not None and self.model.spring_damping is not None:
                wp.launch(
                    exp_param_kernel,
                    dim=self.model.spring_damping.shape[0],
                    inputs=[
                        self.spring_kd_log,
                        self.spring_kd_min,
                        self.spring_kd_max,
                    ],
                    outputs=[self.model.spring_damping],
                    device=self.model.device,
                )

        for substep in range(self.substeps):
            self.state_0.clear_forces()
            if self.enable_ground_penalty and self.object_count > 0:
                wp.launch(
                    apply_ground_penalty_kernel,
                    dim=self.object_count,
                    inputs=[
                        self.state_0.particle_q,
                        self.state_0.particle_qd,
                        self.state_0.particle_f,
                        self.object_count,
                        self.ground_height,
                        self.ground_k,
                        self.ground_kd,
                    ],
                    device=self.model.device,
                )
            if self.controller_indices_wp is not None and self.controller_targets_wp is not None:
                alpha = float(substep + 1) / float(self.substeps)
                ctrl_a = self.ctrl_points[frame_index]
                ctrl_b = self.ctrl_points[next_index]
                targets = (1.0 - alpha) * ctrl_a[: self.controller_count] + alpha * ctrl_b[
                    : self.controller_count
                ]
                self.controller_targets_wp.assign([wp.vec3(*pt.tolist()) for pt in targets])
                wp.launch(
                    set_controller_points_kernel,
                    dim=self.controller_count,
                    inputs=[
                        self.state_0.particle_q,
                        self.state_0.particle_qd,
                        self.controller_indices_wp,
                        self.controller_targets_wp,
                        self.dt,
                    ],
                    device=self.model.device,
                    record_tape=False,
                )

            contacts = None
            if self.collision_pipeline is not None:
                contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, contacts, self.dt)
            if drag_factor < 1.0 and self.state_1.particle_count:
                wp.launch(
                    apply_drag_kernel,
                    dim=self.state_1.particle_count,
                    inputs=[self.state_1.particle_qd, drag_factor],
                    device=self.model.device,
                )
            self.state_0, self.state_1 = self.state_1, self.state_0

        if loss is not None:
            frame_offset = frame * self.object_count
            denom = float(self.frame_loss_denoms[frame])
            wp.launch(
                track_loss_kernel,
                dim=self.object_count,
                inputs=[
                    self.state_0.particle_q,
                    self.gt_points_wp,
                    self.valid_wp,
                    frame_offset,
                    denom,
                ],
                outputs=[loss],
                device=self.model.device,
            )

        if monitor and viewer is not None:
            if not viewer.is_running():
                self.monitor_viewer = None
                return
            sim_time = float(frame) / self.fps
            viewer.begin_frame(sim_time)

            pred = self.state_0.particle_q.numpy()[: self.object_count]
            pred_points = pred + pred_offset[None, :]
            pred_wp = wp.array(pred_points.tolist(), dtype=wp.vec3, device=self.model.device)
            pred_radii = wp.full(
                pred_points.shape[0],
                float(self.config.get("particle_radius", 0.02)),
                dtype=float,
                device=self.model.device,
            )
            pred_colors = wp.full(
                pred_points.shape[0],
                wp.vec3(*self.predict_color.tolist()),
                dtype=wp.vec3,
                device=self.model.device,
            )
            viewer.log_points("/predict/points", pred_wp, pred_radii, pred_colors, hidden=False)

            if self.controller_count > 0:
                positions = self.state_0.particle_q.numpy()
                ctrl_pred = positions[: self.model.particle_count][
                    self.mapping.controller_particle_indices[: self.controller_count]
                ] + pred_offset[None, :]
                ctrl_wp = wp.array(ctrl_pred.tolist(), dtype=wp.vec3, device=self.model.device)
                ctrl_radii = wp.full(
                    ctrl_pred.shape[0],
                    float(self.config.get("particle_radius", 0.02)) * 1.25,
                    dtype=float,
                    device=self.model.device,
                )
                ctrl_colors = wp.full(
                    ctrl_pred.shape[0],
                    wp.vec3(*self.predict_controller_color.tolist()),
                    dtype=wp.vec3,
                    device=self.model.device,
                )
                viewer.log_points("/predict/controllers", ctrl_wp, ctrl_radii, ctrl_colors, hidden=False)

            gt_points = self.gt_points_wp.numpy().reshape(self.train_frame, self.object_count, 3)[frame]
            gt_points_vis = gt_points + gt_offset[None, :]
            gt_wp = wp.array(gt_points_vis.tolist(), dtype=wp.vec3, device=self.model.device)
            gt_radii = wp.full(
                gt_points_vis.shape[0],
                float(self.config.get("particle_radius", 0.02)),
                dtype=float,
                device=self.model.device,
            )
            gt_colors = wp.full(
                gt_points_vis.shape[0],
                wp.vec3(*self.gt_color.tolist()),
                dtype=wp.vec3,
                device=self.model.device,
            )
            viewer.log_points("/gt/points", gt_wp, gt_radii, gt_colors, hidden=False)

            if self.ctrl_points.shape[1] > 0:
                ctrl_vis = self.ctrl_points[frame] + gt_offset[None, :]
                ctrl_wp = wp.array(ctrl_vis.tolist(), dtype=wp.vec3, device=self.model.device)
                ctrl_radii = wp.full(
                    ctrl_vis.shape[0],
                    float(self.config.get("particle_radius", 0.02)) * 1.25,
                    dtype=float,
                    device=self.model.device,
                )
                ctrl_colors = wp.full(
                    ctrl_vis.shape[0],
                    wp.vec3(*self.gt_controller_color.tolist()),
                    dtype=wp.vec3,
                    device=self.model.device,
                )
                viewer.log_points("/gt/controllers", ctrl_wp, ctrl_radii, ctrl_colors, hidden=False)

            viewer.end_frame()

    def _rollout(self, monitor: bool = False) -> None:
        self._reset_states()

        viewer = None
        monitor_limit = self.train_frame
        gt_offset = np.zeros(3, dtype=np.float32)
        pred_offset = np.zeros(3, dtype=np.float32)
        if monitor:
            if self.monitor_viewer is None:
                self.monitor_viewer = newton.viewer.ViewerGL()
                self.monitor_viewer.set_model(self.model)
                self.monitor_viewer.show_particles = False
            viewer = self.monitor_viewer
            gt_offset, pred_offset = self._compute_visual_offsets()
            if self.monitor_frames > 0:
                monitor_limit = min(self.monitor_frames, self.train_frame)

        for frame in range(1, self.train_frame):
            if viewer is None or frame < monitor_limit:
                self._step_frame(
                    frame=frame,
                    loss=None,
                    monitor=monitor,
                    viewer=viewer,
                    gt_offset=gt_offset,
                    pred_offset=pred_offset,
                )

    def _apply_gradients(self) -> None:
        if "spring_ke" in self.train_params:
            if self.use_log_params:
                # Manual chain rule: d(loss)/d(log_param) = d(loss)/d(param) * d(param)/d(log_param)
                # For param = exp(log_param), d(param)/d(log_param) = exp(log_param) = param
                if self.spring_ke_log is None or self.model.spring_stiffness is None:
                    return
                if self.spring_ke_m is None or self.spring_ke_v is None:
                    return
                if self.model.spring_stiffness.grad is None:
                    LOGGER.warning("spring_stiffness.grad is None - gradient not computed!")
                    return

                # Copy gradient from spring_stiffness to spring_ke_log with chain rule
                if self.spring_ke_log.grad is None:
                    self.spring_ke_log.grad = wp.zeros_like(self.spring_ke_log)

                # d(loss)/d(log_ke) = d(loss)/d(ke) * ke
                ke_grad_np = self.model.spring_stiffness.grad.numpy()
                ke_np = self.model.spring_stiffness.numpy()
                log_grad = ke_grad_np * ke_np
                self.spring_ke_log.grad.assign(log_grad)

                param = self.spring_ke_log
                min_val = self.spring_ke_log_min
                max_val = self.spring_ke_log_max
            else:
                param = self.model.spring_stiffness
                if param is None or param.grad is None:
                    return
                if self.spring_ke_m is None or self.spring_ke_v is None:
                    return
                min_val = self.spring_ke_min
                max_val = self.spring_ke_max

            self.adam_step += 1
            bias_correction1 = 1.0 - self.adam_beta1 ** self.adam_step
            bias_correction2 = 1.0 - self.adam_beta2 ** self.adam_step
            wp.launch(
                adam_update_kernel,
                dim=param.shape[0],
                inputs=[
                    param,
                    param.grad,
                    self.spring_ke_m,
                    self.spring_ke_v,
                    self.train_lr,
                    self.adam_beta1,
                    self.adam_beta2,
                    bias_correction1,
                    bias_correction2,
                    self.adam_eps,
                    self.grad_clip,
                    min_val,
                    max_val,
                ],
                device=self.model.device,
            )
        if "spring_kd" in self.train_params:
            if self.use_log_params:
                # Manual chain rule for damping parameter
                if self.spring_kd_log is None or self.model.spring_damping is None:
                    return
                if self.spring_kd_m is None or self.spring_kd_v is None:
                    return
                if self.model.spring_damping.grad is None:
                    return

                # Copy gradient from spring_damping to spring_kd_log with chain rule
                if self.spring_kd_log.grad is None:
                    self.spring_kd_log.grad = wp.zeros_like(self.spring_kd_log)

                # d(loss)/d(log_kd) = d(loss)/d(kd) * kd
                kd_grad_np = self.model.spring_damping.grad.numpy()
                kd_np = self.model.spring_damping.numpy()
                log_grad = kd_grad_np * kd_np
                self.spring_kd_log.grad.assign(log_grad)

                param = self.spring_kd_log
                min_val = self.spring_kd_log_min
                max_val = self.spring_kd_log_max
            else:
                param = self.model.spring_damping
                if param is None or param.grad is None:
                    return
                if self.spring_kd_m is None or self.spring_kd_v is None:
                    return
                min_val = self.spring_kd_min
                max_val = self.spring_kd_max

            if "spring_ke" not in self.train_params:
                self.adam_step += 1
            bias_correction1 = 1.0 - self.adam_beta1 ** self.adam_step
            bias_correction2 = 1.0 - self.adam_beta2 ** self.adam_step
            wp.launch(
                adam_update_kernel,
                dim=param.shape[0],
                inputs=[
                    param,
                    param.grad,
                    self.spring_kd_m,
                    self.spring_kd_v,
                    self.train_lr,
                    self.adam_beta1,
                    self.adam_beta2,
                    bias_correction1,
                    bias_correction2,
                    self.adam_eps,
                    self.grad_clip,
                    min_val,
                    max_val,
                ],
                device=self.model.device,
            )

    def train(self, max_iter: int | None = None) -> dict[str, Any]:
        iterations = int(max_iter or self.config.get("iterations", 20))
        best_loss = None
        best_params = None

        for i in range(iterations):
            start = time.time()
            self._reset_states()
            total_loss_value = 0.0
            nan_frame = None
            q_pre_stats = {"bad": 0}
            qd_pre_stats = {"bad": 0}
            q_stats = {"max_abs": None, "max_norm": None, "min": None, "max": None, "bad": 0}
            qd_stats = {"max_abs": None, "max_norm": None, "min": None, "max": None, "bad": 0}
            ke_gbad_max = 0
            kd_gbad_max = 0

            for frame in range(1, self.train_frame):
                loss = wp.zeros(1, dtype=float, requires_grad=True, device=self.model.device)
                tape = wp.Tape()
                with tape:
                    self._step_frame(
                        frame=frame,
                        loss=loss,
                        monitor=False,
                        viewer=None,
                        gt_offset=np.zeros(3, dtype=np.float32),
                        pred_offset=np.zeros(3, dtype=np.float32),
                    )

                q_pre_stats = {"bad": 0}
                qd_pre_stats = {"bad": 0}
                if self.debug_nan_check and ((i + 1) % self.debug_nan_every == 0):
                    q_pre_stats = self._finite_stats(self.state_0.particle_q.numpy())
                    qd_pre_stats = self._finite_stats(self.state_0.particle_qd.numpy())

                tape.backward(loss)

                ke_gbad_frame = 0
                kd_gbad_frame = 0
                if self.debug_nan_param_check:
                    if self.use_log_params:
                        if self.spring_ke_log is not None and self.spring_ke_log.grad is not None:
                            ke_gbad_frame = self._finite_stats(self.spring_ke_log.grad.numpy())["bad"]
                        if self.spring_kd_log is not None and self.spring_kd_log.grad is not None:
                            kd_gbad_frame = self._finite_stats(self.spring_kd_log.grad.numpy())["bad"]
                    else:
                        if self.model.spring_stiffness is not None and self.model.spring_stiffness.grad is not None:
                            ke_gbad_frame = self._finite_stats(self.model.spring_stiffness.grad.numpy())["bad"]
                        if self.model.spring_damping is not None and self.model.spring_damping.grad is not None:
                            kd_gbad_frame = self._finite_stats(self.model.spring_damping.grad.numpy())["bad"]

                if (ke_gbad_frame or kd_gbad_frame) and self.skip_update_on_nan_grad:
                    LOGGER.warning(
                        "Skip update at iter %d frame %d due to non-finite grad (ke=%d kd=%d)",
                        i + 1,
                        frame,
                        ke_gbad_frame,
                        kd_gbad_frame,
                    )
                else:
                    self._apply_gradients()

                if self.debug_nan_param_check:
                    if self.use_log_params:
                        if self.spring_ke_log is not None and self.spring_ke_log.grad is not None:
                            ke_gbad_max = max(
                                ke_gbad_max,
                                self._finite_stats(self.spring_ke_log.grad.numpy())["bad"],
                            )
                        if self.spring_kd_log is not None and self.spring_kd_log.grad is not None:
                            kd_gbad_max = max(
                                kd_gbad_max,
                                self._finite_stats(self.spring_kd_log.grad.numpy())["bad"],
                            )
                    else:
                        if self.model.spring_stiffness is not None and self.model.spring_stiffness.grad is not None:
                            ke_gbad_max = max(
                                ke_gbad_max,
                                self._finite_stats(self.model.spring_stiffness.grad.numpy())["bad"],
                            )
                        if self.model.spring_damping is not None and self.model.spring_damping.grad is not None:
                            kd_gbad_max = max(
                                kd_gbad_max,
                                self._finite_stats(self.model.spring_damping.grad.numpy())["bad"],
                            )

                if self.debug_nan_check and ((i + 1) % self.debug_nan_every == 0):
                    q_np = self.state_0.particle_q.numpy()
                    qd_np = self.state_0.particle_qd.numpy()
                    q_stats = self._finite_stats(q_np)
                    qd_stats = self._finite_stats(qd_np)
                    if q_stats["bad"] or qd_stats["bad"]:
                        nan_frame = frame
                        LOGGER.error(
                            "Non-finite detected at iter %d frame %d | q_bad=%d qd_bad=%d",
                            i + 1,
                            frame,
                            q_stats["bad"],
                            qd_stats["bad"],
                        )
                        if self.debug_nan_break:
                            break

                tape.zero()

                total_loss_value += float(loss.numpy()[0])

                if nan_frame is not None:
                    break

            if nan_frame is not None and self.debug_nan_break:
                break

            ke_stats = (None, None, None)
            ke_grad_stats = (None, None, None)
            kd_stats = (None, None, None)
            kd_grad_stats = (None, None, None)
            ke_bad = 0
            kd_bad = 0
            ke_gbad = ke_gbad_max
            kd_gbad = kd_gbad_max
            if self.log_param_stats:
                ke_stats = self._stats(self.model.spring_stiffness)
                kd_stats = self._stats(self.model.spring_damping)
                if self.use_log_params:
                    ke_grad_stats = self._grad_stats(self.spring_ke_log)
                    kd_grad_stats = self._grad_stats(self.spring_kd_log)
                else:
                    ke_grad_stats = self._grad_stats(self.model.spring_stiffness)
                    kd_grad_stats = self._grad_stats(self.model.spring_damping)
                if self.debug_nan_param_check:
                    if self.model.spring_stiffness is not None:
                        ke_bad = self._finite_stats(self.model.spring_stiffness.numpy())["bad"]
                        if self.use_log_params:
                            if self.spring_ke_log is not None and self.spring_ke_log.grad is not None:
                                ke_gbad = self._finite_stats(self.spring_ke_log.grad.numpy())["bad"]
                        else:
                            if self.model.spring_stiffness.grad is not None:
                                ke_gbad = self._finite_stats(self.model.spring_stiffness.grad.numpy())["bad"]
                    if self.model.spring_damping is not None:
                        kd_bad = self._finite_stats(self.model.spring_damping.numpy())["bad"]
                        if self.use_log_params:
                            if self.spring_kd_log is not None and self.spring_kd_log.grad is not None:
                                kd_gbad = self._finite_stats(self.spring_kd_log.grad.numpy())["bad"]
                        else:
                            if self.model.spring_damping.grad is not None:
                                kd_gbad = self._finite_stats(self.model.spring_damping.grad.numpy())["bad"]
            loss_value = total_loss_value / max(self.train_frame - 1, 1)
            elapsed = time.time() - start

            if best_loss is None or loss_value < best_loss:
                best_loss = loss_value
                best_params = {
                    "spring_stiffness": self.model.spring_stiffness.numpy().copy(),
                    "spring_damping": self.model.spring_damping.numpy().copy(),
                }
                best_path = os.path.join(self.base_dir, "train", "best_params.pkl")
                with open(best_path, "wb") as f:
                    pickle.dump(best_params, f)
                best_iter_path = os.path.join(self.base_dir, "train", f"best_iter_{i + 1}.pkl")
                with open(best_iter_path, "wb") as f:
                    pickle.dump(best_params, f)

            if (i + 1) % self.log_every == 0:
                LOGGER.info(
                    "Iter %d | loss=%.6f | %.2fs | ke=(%.3e, %.3e, %.3e) | kd=(%.3e, %.3e, %.3e)",
                    i + 1,
                    loss_value,
                    elapsed,
                    ke_stats[0] if ke_stats[0] is not None else float("nan"),
                    ke_stats[1] if ke_stats[1] is not None else float("nan"),
                    ke_stats[2] if ke_stats[2] is not None else float("nan"),
                    kd_stats[0] if kd_stats[0] is not None else float("nan"),
                    kd_stats[1] if kd_stats[1] is not None else float("nan"),
                    kd_stats[2] if kd_stats[2] is not None else float("nan"),
                )
                self._log_writer.writerow(
                    [
                        i + 1,
                        f"{loss_value:.6f}",
                        f"{elapsed:.4f}",
                        q_pre_stats["bad"],
                        qd_pre_stats["bad"],
                        q_stats["max_abs"],
                        q_stats["max_norm"],
                        q_stats["min"],
                        q_stats["max"],
                        qd_stats["max_abs"],
                        qd_stats["max_norm"],
                        qd_stats["min"],
                        qd_stats["max"],
                        q_stats["bad"],
                        qd_stats["bad"],
                        ke_bad,
                        ke_gbad,
                        kd_bad,
                        kd_gbad,
                        ke_stats[0],
                        ke_stats[1],
                        ke_stats[2],
                        ke_grad_stats[0],
                        ke_grad_stats[1],
                        ke_grad_stats[2],
                        kd_stats[0],
                        kd_stats[1],
                        kd_stats[2],
                        kd_grad_stats[0],
                        kd_grad_stats[1],
                        kd_grad_stats[2],
                    ]
                )
                self._log_file.flush()

            if self.monitor and (i % self.monitor_every == 0):
                self._rollout(monitor=True)

        self._log_file.close()
        return best_params or {}

    def simulate_trajectory(
        self,
        params: dict[str, Any] | None = None,
        frame_limit: int | None = None,
    ) -> np.ndarray:
        scale = float(self.config.get("scale", 1.0))
        z_offset = float(self.config.get("z_offset", 0.0))
        reverse_z = bool(self.config.get("reverse_z", False))
        fps = float(self.config.get("FPS", 30.0))
        substeps = int(self.config.get("substeps", self.config.get("num_substeps", 1)))
        dt = 1.0 / fps / substeps

        use_controllers = bool(self.config.get("use_controllers", True))
        particle_mass = float(self.config.get("particle_mass", 1.0))
        add_ground = not bool(self.config.get("no_ground", False))
        drag_damping = float(self.config.get("drag_damping", 0.0))
        enable_ground_penalty = bool(self.config.get("enable_ground_penalty", True))
        ground_height = float(self.config.get("ground_height", 0.0))
        ground_k = float(self.config.get("ground_k", 1.0e4))
        ground_kd = float(self.config.get("ground_kd", 1.0e2))

        frame0 = 0
        predict_mask = self._object_mask(self.predict_data, frame0)
        predict_points = self.predict_data.object_points[frame0][predict_mask]
        scaled_predict = self._apply_scale(predict_points, scale, reverse_z)
        z_shift = self._compute_z_shift(scaled_predict, z_offset)

        neighbor_mode = str(self.config.get("spring_neighbor_mode", "knn")).lower()
        object_radius = self.config.get("object_radius", None)
        object_max_neighbours = self.config.get("object_max_neighbours", None)
        controller_radius = self.config.get("controller_radius", None)
        controller_max_neighbours = self.config.get("controller_max_neighbours", None)
        k_neighbors = self.config.get("k_neighbors", 6)
        controller_k = self.config.get("controller_k", 1)

        spring_ke = float(self.config.get("spring_ke", 3.0e4))
        spring_kd = float(self.config.get("spring_kd", 100.0))
        particle_radius = float(self.config.get("particle_radius", 0.02))

        mapping = map_pkl_to_newton(
            data=self.predict_data,
            frame=frame0,
            scale=scale,
            z_offset=z_offset,
            reverse_z=reverse_z,
            particle_radius=particle_radius,
            mass=particle_mass,
            spring_ke=spring_ke,
            spring_kd=spring_kd,
            k_neighbors=int(k_neighbors),
            add_ground=add_ground,
            use_controllers=use_controllers,
            controller_k=int(controller_k),
            controller_ke=spring_ke,
            controller_kd=spring_kd,
            filter_visibility=bool(self.config.get("filter_visibility", False)),
            filter_motion_valid=bool(self.config.get("filter_motion_valid", False)),
            spring_neighbor_mode=neighbor_mode,
            object_radius=object_radius,
            object_max_neighbours=object_max_neighbours,
            controller_radius=controller_radius,
            controller_max_neighbours=controller_max_neighbours,
            requires_grad=False,
        )

        model = mapping.model
        if not bool(self.config.get("self_collision", True)):
            model.particle_grid = None
        self._apply_contact_params(model)

        if params:
            if "spring_stiffness" in params and model.spring_stiffness is not None:
                model.spring_stiffness.assign(params["spring_stiffness"])
            if "spring_damping" in params and model.spring_damping is not None:
                model.spring_damping.assign(params["spring_damping"])

        solver = newton.solvers.SolverSemiImplicit(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        enable_collisions = bool(self.config.get("enable_collisions", False))
        collision_pipeline = None
        if enable_collisions:
            collision_pipeline = newton.examples.create_collision_pipeline(
                model, collision_pipeline_type="standard"
            )

        object_count = mapping.object_points.shape[0]

        ctrl_points = self.gt_data.controller_points
        if ctrl_points.size == 0 and self.predict_data.controller_points.size:
            ctrl_points = self.predict_data.controller_points
        if ctrl_points.size:
            ctrl_points = ctrl_points[: self.train_frame]
            ctrl_points = self._apply_scale_z(ctrl_points, scale, z_shift, reverse_z)
        else:
            ctrl_points = np.zeros((self.train_frame, 0, 3), dtype=np.float32)

        controller_indices = mapping.controller_particle_indices.astype(np.int32)
        controller_count = min(controller_indices.shape[0], ctrl_points.shape[1])
        if use_controllers and controller_count > 0:
            controller_indices = controller_indices[:controller_count]
            controller_indices_wp = wp.array(
                controller_indices.tolist(), dtype=int, device=model.device
            )
            controller_targets_wp = wp.array(
                [wp.vec3(0.0, 0.0, 0.0)] * controller_count,
                dtype=wp.vec3,
                device=model.device,
            )
        else:
            controller_indices_wp = None
            controller_targets_wp = None

        drag_factor = max(0.0, 1.0 - drag_damping * dt)
        frame_limit = self.train_frame if frame_limit is None else min(frame_limit, self.train_frame)

        positions = [state_0.particle_q.numpy()[:object_count]]

        for frame in range(1, frame_limit):
            frame_index = frame - 1
            next_index = frame
            for substep in range(substeps):
                state_0.clear_forces()
                if enable_ground_penalty and object_count > 0:
                    wp.launch(
                        apply_ground_penalty_kernel,
                        dim=object_count,
                        inputs=[
                            state_0.particle_q,
                            state_0.particle_qd,
                            state_0.particle_f,
                            object_count,
                            ground_height,
                            ground_k,
                            ground_kd,
                        ],
                        device=model.device,
                    )
                if controller_indices_wp is not None and controller_targets_wp is not None:
                    alpha = float(substep + 1) / float(substeps)
                    ctrl_a = ctrl_points[frame_index]
                    ctrl_b = ctrl_points[next_index]
                    targets = (1.0 - alpha) * ctrl_a[:controller_count] + alpha * ctrl_b[:controller_count]
                    controller_targets_wp.assign([wp.vec3(*pt.tolist()) for pt in targets])
                    wp.launch(
                        set_controller_points_kernel,
                        dim=controller_count,
                        inputs=[
                            state_0.particle_q,
                            state_0.particle_qd,
                            controller_indices_wp,
                            controller_targets_wp,
                            dt,
                        ],
                        device=model.device,
                        record_tape=False,
                    )

                contacts = None
                if collision_pipeline is not None:
                    contacts = model.collide(state_0, collision_pipeline=collision_pipeline)
                solver.step(state_0, state_1, control, contacts, dt)
                if drag_factor < 1.0 and state_1.particle_count:
                    wp.launch(
                        apply_drag_kernel,
                        dim=state_1.particle_count,
                        inputs=[state_1.particle_qd, drag_factor],
                        device=model.device,
                    )
                state_0, state_1 = state_1, state_0

            positions.append(state_0.particle_q.numpy()[:object_count])

        return np.stack(positions, axis=0)

    def test(self, params: dict[str, Any] | None = None, save_path: str | None = None) -> None:
        if params is None:
            best_path = os.path.join(self.base_dir, "train", "best_params.pkl")
            if os.path.exists(best_path):
                with open(best_path, "rb") as f:
                    params = pickle.load(f)

        trajectory = self.simulate_trajectory(params)
        if save_path is None:
            save_path = os.path.join(self.base_dir, "train", "inference.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(trajectory, f)
        LOGGER.info("Saved inference trajectory to %s", save_path)


def load_yaml_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping")
    return data
