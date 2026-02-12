from __future__ import annotations

import logging
import os
import pickle
import time
from dataclasses import dataclass
from typing import Any

import cma
import numpy as np
import warp as wp
import yaml

import newton
import newton.examples

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover
    cKDTree = None

from phystwin.mapping.pkl_mapping import SpringMassPKL, SpringMassPKLPair, load_pkl, map_pkl_to_newton


LOGGER = logging.getLogger("phystwin.cma_optimize")


@wp.kernel
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


@dataclass
class ParamSpec:
    name: str
    min_value: float
    max_value: float
    kind: str = "float"  # "float" or "int"


class OptimizerCMA:
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
    ):
        self.data_path = data_path
        self.base_dir = base_dir
        self.config = config
        self.device = device
        self.monitor = monitor
        self.monitor_frames = monitor_frames
        self.monitor_every = max(1, monitor_every)

        wp.init()
        try:
            wp.set_device(device)
        except Exception:
            LOGGER.warning("Failed to set Warp device %s; using default", device)

        os.makedirs(os.path.join(self.base_dir, "optimizeCMA"), exist_ok=True)

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
        self.train_frame = min(train_frame, self.num_frames)
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

        self._init_param_specs()

    def _init_param_specs(self) -> None:
        neighbor_mode = str(self.config.get("spring_neighbor_mode", "knn")).lower()
        def rng(key: str, default_min: float, default_max: float) -> tuple[float, float]:
            min_key = f"{key}_min"
            max_key = f"{key}_max"
            min_val = float(self.config.get(min_key, default_min))
            max_val = float(self.config.get(max_key, default_max))
            return min_val, max_val

        spring_ke_min, spring_ke_max = rng("spring_ke", 1.0e3, 1.0e5)
        spring_kd_min, spring_kd_max = rng("spring_kd", 0.0, 2.0e2)
        radius_min, radius_max = rng("particle_radius", 0.005, 0.05)
        if neighbor_mode == "radius":
            object_radius_min, object_radius_max = rng("object_radius", 0.005, 0.05)
            object_k_min, object_k_max = rng("object_max_neighbours", 10, 50)

            self.param_specs = [
                ParamSpec("spring_ke", spring_ke_min, spring_ke_max),
                ParamSpec("spring_kd", spring_kd_min, spring_kd_max),
                ParamSpec("particle_radius", radius_min, radius_max),
                ParamSpec("object_radius", object_radius_min, object_radius_max),
                ParamSpec("object_max_neighbours", object_k_min, object_k_max, kind="int"),
            ]
        else:
            k_min, k_max = rng("k_neighbors", 4, 50)

            self.param_specs = [
                ParamSpec("spring_ke", spring_ke_min, spring_ke_max),
                ParamSpec("spring_kd", spring_kd_min, spring_kd_max),
                ParamSpec("particle_radius", radius_min, radius_max),
                ParamSpec("k_neighbors", k_min, k_max, kind="int"),
            ]

    def _apply_contact_params(self, model: newton.Model, params: dict[str, Any]) -> None:
        base_ke = float(self.config.get("particle_contact_ke", 1.0e3))
        base_kd = float(self.config.get("particle_contact_kd", 1.0e2))
        base_kf = float(self.config.get("particle_contact_kf", 1.0e2))

        collide_elas = float(params.get("collide_elas", self.config.get("collide_elas", 1.0)))
        collide_fric = float(params.get("collide_fric", self.config.get("collide_fric", 0.5)))

        model.particle_ke = base_ke * max(collide_elas, 0.0)
        model.particle_kd = base_kd
        model.particle_kf = base_kf
        model.particle_mu = max(collide_fric, 0.0)

        if getattr(model, "shape_count", 0) and model.shape_material_ke is not None:
            shape_count = int(model.shape_material_ke.shape[0])
            obj_elas = float(params.get("collide_object_elas", self.config.get("collide_object_elas", 1.0)))
            obj_fric = float(params.get("collide_object_fric", self.config.get("collide_object_fric", 0.5)))
            model.shape_material_ke.assign([base_ke * max(obj_elas, 0.0)] * shape_count)
            if model.shape_material_kd is not None:
                model.shape_material_kd.assign([base_kd] * shape_count)
            if model.shape_material_kf is not None:
                model.shape_material_kf.assign([base_kf] * shape_count)
            if model.shape_material_mu is not None:
                model.shape_material_mu.assign([max(obj_fric, 0.0)] * shape_count)

    def _normalize(self, value: float, spec: ParamSpec) -> float:
        return (value - spec.min_value) / (spec.max_value - spec.min_value)

    def _denormalize(self, value: float, spec: ParamSpec) -> float:
        return value * (spec.max_value - spec.min_value) + spec.min_value

    def _params_from_vector(self, x: list[float]) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for xi, spec in zip(x, self.param_specs, strict=False):
            val = float(self._denormalize(xi, spec))
            if spec.kind == "int":
                val = int(round(val))
            params[spec.name] = val
        return params

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

    def _smooth_l1(self, pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None) -> float:
        diff = np.abs(pred - gt)
        loss = np.where(diff < 1.0, 0.5 * diff**2.0, diff - 0.5)
        loss = loss.sum(axis=1)
        if valid is not None:
            loss = loss[valid]
            denom = max(int(valid.sum()), 1) * 3.0
        else:
            denom = pred.shape[0] * 3.0
        return float(loss.sum() / denom)

    def _chamfer_loss(self, pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None) -> float:
        if valid is not None:
            gt = gt[valid]
        if gt.size == 0 or pred.size == 0:
            return 0.0
        if cKDTree is None:
            # Fallback: chunked brute force
            chunk = 1024
            min_d2 = []
            for i in range(0, gt.shape[0], chunk):
                block = gt[i : i + chunk]
                d2 = np.sum((block[:, None, :] - pred[None, :, :]) ** 2, axis=2)
                min_d2.append(d2.min(axis=1))
            min_d2 = np.concatenate(min_d2, axis=0)
        else:
            tree = cKDTree(pred)
            dist, _ = tree.query(gt, k=1)
            min_d2 = dist**2
        return float(min_d2.sum() / max(gt.shape[0], 1))

    def _compute_visual_offsets(self, mapping) -> tuple[np.ndarray, np.ndarray]:
        if self.compare_offset is None:
            if mapping.object_points.size:
                extents = mapping.object_points.max(axis=0) - mapping.object_points.min(axis=0)
                offset = float(max(extents[0], extents[1]) * 1.5)
            else:
                offset = 1.0
        else:
            offset = float(self.compare_offset)

        if self.compare_axis == "y":
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

    def _simulate(self, params: dict[str, Any], monitor: bool = False) -> tuple[float, dict[str, float]]:
        scale = float(self.config.get("scale", 1.0))
        z_offset = float(self.config.get("z_offset", 0.0))
        reverse_z = bool(self.config.get("reverse_z", False))
        fps = float(self.config.get("FPS", 30.0))
        substeps = int(self.config.get("substeps", self.config.get("num_substeps", 1)))
        dt = 1.0 / fps / substeps

        use_controllers = bool(self.config.get("use_controllers", True))
        particle_mass = float(self.config.get("particle_mass", 1.0))
        add_ground = not bool(self.config.get("no_ground", False))

        frame0 = 0
        predict_mask = self._object_mask(self.predict_data, frame0)
        predict_points = self.predict_data.object_points[frame0][predict_mask]
        scaled_predict = self._apply_scale(predict_points, scale, reverse_z)
        z_shift = self._compute_z_shift(scaled_predict, z_offset)

        neighbor_mode = str(self.config.get("spring_neighbor_mode", "knn")).lower()
        object_radius = params.get("object_radius", self.config.get("object_radius", None))
        object_max_neighbours = params.get(
            "object_max_neighbours", self.config.get("object_max_neighbours", None)
        )
        k_neighbors = params.get("k_neighbors", self.config.get("k_neighbors", 6))
        controller_k = self.config.get("controller_k", 1)

        mapping = map_pkl_to_newton(
            data=self.predict_data,
            frame=frame0,
            scale=scale,
            z_offset=z_offset,
            reverse_z=reverse_z,
            particle_radius=float(params["particle_radius"]),
            mass=particle_mass,
            spring_ke=float(params["spring_ke"]),
            spring_kd=float(params["spring_kd"]),
            k_neighbors=int(k_neighbors),
            add_ground=add_ground,
            use_controllers=use_controllers,
            controller_k=int(controller_k),
            controller_ke=float(params["spring_ke"]),
            controller_kd=float(params["spring_kd"]),
            filter_visibility=bool(self.config.get("filter_visibility", False)),
            filter_motion_valid=bool(self.config.get("filter_motion_valid", False)),
            spring_neighbor_mode=neighbor_mode,
            object_radius=object_radius,
            object_max_neighbours=object_max_neighbours,
        )

        model = mapping.model
        self._apply_contact_params(model, params)
        solver = newton.solvers.SolverSemiImplicit(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.examples.create_collision_pipeline(
            model, collision_pipeline_type="standard"
        )

        object_count = mapping.object_points.shape[0]
        pkl_object_indices = mapping.pkl_object_indices

        gt_points = []
        gt_vis = []
        gt_motion = []
        for frame in range(self.train_frame):
            pts = self.gt_data.object_points[frame][pkl_object_indices]
            pts = self._apply_scale_z(pts, scale, z_shift, reverse_z)
            gt_points.append(pts)
            if self.gt_data.object_visibilities.size:
                gt_vis.append(self.gt_data.object_visibilities[frame][pkl_object_indices].astype(bool))
            else:
                gt_vis.append(np.ones(pts.shape[0], dtype=bool))
            if self.gt_data.object_motions_valid.size:
                gt_motion.append(self.gt_data.object_motions_valid[frame][pkl_object_indices].astype(bool))
            else:
                gt_motion.append(np.ones(pts.shape[0], dtype=bool))

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

        chamfer_weight = float(self.config.get("chamfer_weight", 1.0))
        track_weight = float(self.config.get("track_weight", 1.0))
        acc_weight = float(self.config.get("acc_weight", 0.0))

        total_loss = 0.0
        chamfer_total = 0.0
        track_total = 0.0
        acc_total = 0.0

        prev_acc = None
        acc_enabled = False

        viewer = None
        monitor_limit = self.train_frame
        gt_offset = np.zeros(3, dtype=np.float32)
        pred_offset = np.zeros(3, dtype=np.float32)
        if monitor:
            if self.monitor_viewer is None:
                self.monitor_viewer = newton.viewer.ViewerGL()
                self.monitor_viewer.set_model(model)
                self.monitor_viewer.show_particles = False
            viewer = self.monitor_viewer
            gt_offset, pred_offset = self._compute_visual_offsets(mapping)
            if self.monitor_frames > 0:
                monitor_limit = min(self.monitor_frames, self.train_frame)

        for frame in range(1, self.train_frame):
            frame_index = frame - 1
            next_index = frame
            v_start = state_0.particle_qd.numpy()[:object_count]
            for substep in range(substeps):
                state_0.clear_forces()
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
                    )

                contacts = model.collide(state_0, collision_pipeline=collision_pipeline)
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0

            pred = state_0.particle_q.numpy()[:object_count]
            gt = gt_points[frame]
            vis_mask = gt_vis[frame]
            motion_mask = gt_motion[frame]

            chamfer = self._chamfer_loss(pred, gt, vis_mask)
            track = self._smooth_l1(pred, gt, motion_mask)

            chamfer_total += chamfer_weight * chamfer
            track_total += track_weight * track

            v_end = state_0.particle_qd.numpy()[:object_count]
            cur_acc = v_end - v_start
            if acc_enabled and acc_weight > 0.0 and prev_acc is not None:
                acc_loss = self._smooth_l1(cur_acc, prev_acc, None)
                acc_total += acc_weight * acc_loss
            prev_acc = cur_acc
            acc_enabled = True

            if viewer is not None and frame < monitor_limit:
                if not viewer.is_running():
                    self.monitor_viewer = None
                    viewer = None
                else:
                    sim_time = float(frame) / fps
                    viewer.begin_frame(sim_time)

                pred_points = pred + pred_offset[None, :]
                pred_wp = wp.array(pred_points.tolist(), dtype=wp.vec3, device=model.device)
                pred_radii = wp.full(pred_points.shape[0], float(params["particle_radius"]), dtype=float, device=model.device)
                pred_colors = wp.full(pred_points.shape[0], wp.vec3(*self.predict_color.tolist()), dtype=wp.vec3, device=model.device)
                viewer.log_points("/predict/points", pred_wp, pred_radii, pred_colors, hidden=False)

                if controller_count > 0:
                    positions = state_0.particle_q.numpy()
                    ctrl_pred = positions[controller_indices[:controller_count]] + pred_offset[None, :]
                    ctrl_wp = wp.array(ctrl_pred.tolist(), dtype=wp.vec3, device=model.device)
                    ctrl_radii = wp.full(
                        ctrl_pred.shape[0],
                        float(params["particle_radius"]) * 1.25,
                        dtype=float,
                        device=model.device,
                    )
                    ctrl_colors = wp.full(
                        ctrl_pred.shape[0],
                        wp.vec3(*self.predict_controller_color.tolist()),
                        dtype=wp.vec3,
                        device=model.device,
                    )
                    viewer.log_points("/predict/controllers", ctrl_wp, ctrl_radii, ctrl_colors, hidden=False)

                gt_points_vis = gt + gt_offset[None, :]
                gt_wp = wp.array(gt_points_vis.tolist(), dtype=wp.vec3, device=model.device)
                gt_radii = wp.full(gt_points_vis.shape[0], float(params["particle_radius"]), dtype=float, device=model.device)
                gt_colors = wp.full(gt_points_vis.shape[0], wp.vec3(*self.gt_color.tolist()), dtype=wp.vec3, device=model.device)
                viewer.log_points("/gt/points", gt_wp, gt_radii, gt_colors, hidden=False)

                if ctrl_points.shape[1] > 0:
                    ctrl_vis = ctrl_points[frame] + gt_offset[None, :]
                    ctrl_wp = wp.array(ctrl_vis.tolist(), dtype=wp.vec3, device=model.device)
                    ctrl_radii = wp.full(ctrl_vis.shape[0], float(params["particle_radius"]) * 1.25, dtype=float, device=model.device)
                    ctrl_colors = wp.full(ctrl_vis.shape[0], wp.vec3(*self.gt_controller_color.tolist()), dtype=wp.vec3, device=model.device)
                    viewer.log_points("/gt/controllers", ctrl_wp, ctrl_radii, ctrl_colors, hidden=False)

                    viewer.end_frame()

        denom = max(self.train_frame - 1, 1)
        total_loss = (chamfer_total + track_total + acc_total) / denom
        components = {
            "chamfer": chamfer_total / denom,
            "track": track_total / denom,
            "acc": acc_total / denom,
        }
        return total_loss, components

    def optimize(self, max_iter: int = 100) -> None:
        x_init = []
        for spec in self.param_specs:
            cfg_val = self.config.get(spec.name, 0.5 * (spec.min_value + spec.max_value))
            val = float(cfg_val)
            if val < spec.min_value or val > spec.max_value:
                LOGGER.warning(
                    "Initial %s=%.6g out of bounds [%.6g, %.6g]; clamping.",
                    spec.name,
                    val,
                    spec.min_value,
                    spec.max_value,
                )
                val = min(max(val, spec.min_value), spec.max_value)
            x_init.append(self._normalize(val, spec))

        LOGGER.info("Initial params: %s", self._params_from_vector(x_init))
        if self.monitor:
            self._simulate(self._params_from_vector(x_init), monitor=True)
        std = 1.0 / 6.0
        es = cma.CMAEvolutionStrategy(x_init, std, {"bounds": [0.0, 1.0], "seed": 42})
        es.optimize(self.error_func, iterations=max_iter)

        res = es.result
        optimal_x = np.array(res[0]).astype(np.float32).tolist()
        optimal_error = res[1]
        optimal_params = self._params_from_vector(optimal_x)

        LOGGER.info("Optimal params: %s", optimal_params)
        LOGGER.info("Optimal error: %s", optimal_error)
        if self.monitor:
            self._simulate(optimal_params, monitor=True)

        results_path = os.path.join(self.base_dir, "optimal_params.pkl")
        with open(results_path, "wb") as f:
            pickle.dump(optimal_params, f)

    def error_func(self, parameters: list[float]) -> float:
        params = self._params_from_vector(parameters)
        self.eval_count += 1
        start = time.time()
        monitor = self.monitor and (self.eval_count % self.monitor_every == 0)
        total, components = self._simulate(params, monitor=monitor)
        elapsed = time.time() - start
        if self.eval_count % self.log_every == 0:
            LOGGER.info(
                "Eval %d | loss=%.6f (chamfer=%.6f, track=%.6f, acc=%.6f) | %.2fs | %s",
                self.eval_count,
                total,
                components["chamfer"],
                components["track"],
                components["acc"],
                elapsed,
                params,
            )
        return total


def load_yaml_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping")
    return data
