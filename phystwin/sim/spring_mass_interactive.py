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


def _configure_joint_hold(model: newton.Model, control: newton.Control, target_ke: float, target_kd: float) -> None:
    """Configure a simple global joint PD hold toward the model's reference pose."""
    if model.joint_count == 0 or model.joint_dof_count == 0:
        return
    if model.joint_target_ke is not None:
        model.joint_target_ke.fill_(float(target_ke))
    if model.joint_target_kd is not None:
        model.joint_target_kd.fill_(float(target_kd))
    if control is not None and control.joint_target_pos is not None and model.joint_q is not None:
        control.joint_target_pos.assign(model.joint_q)
    if control is not None and control.joint_target_vel is not None:
        control.joint_target_vel.zero_()


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
        self.sim_substeps = int(_pick_param_alias(pkl_params, args, ("substeps", "num_substeps"), "substeps"))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        scale = float(_pick_param(pkl_params, args, "scale"))
        z_offset = float(_pick_param(pkl_params, args, "z_offset"))
        reverse_z = bool(_pick_param(pkl_params, args, "reverse_z"))
        particle_radius = float(_pick_param(pkl_params, args, "particle_radius"))
        particle_mass = float(_pick_param_alias(pkl_params, args, ("mass", "particle_mass"), "mass"))
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

        predict_mask = _object_mask(
            spring_data,
            frame=0,
            filter_visibility=filter_visibility,
            filter_motion_valid=filter_motion_valid,
        )
        predict_points = spring_data.object_points[0][predict_mask]
        scaled_predict = _apply_scale(predict_points, scale, reverse_z)
        z_shift = _compute_z_shift(scaled_predict, z_offset)

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
        self.spring_contacts = None
        self.spring_collisions_enabled = bool(args.enable_spring_collisions) or _requires_newton_contacts(args.spring_solver)
        if self.spring_collisions_enabled:
            self.spring_collision_pipeline = newton.examples.create_collision_pipeline(
                self.spring_model, args=args, collision_pipeline_type="unified"
            )
            self.spring_contacts = self.spring_model.collide(
                self.spring_state_0,
                collision_pipeline=self.spring_collision_pipeline,
                soft_contact_margin=self.spring_soft_contact_margin,
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
            self.robot_collision_pipeline = newton.examples.create_collision_pipeline(
                self.robot_model, args=args, collision_pipeline_type="unified"
            )
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
        self.object_radius_value = (
            float(self.spring_model.particle_radius.numpy()[0])
            if self.spring_model.particle_radius is not None
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

    def _init_robot_spring_collider_sync(self) -> None:
        """Create body-index mapping for syncing robot poses into spring colliders."""
        robot_by_key = {k: i for i, k in enumerate(self.robot_model.body_key)}
        spring_by_key = {k: i for i, k in enumerate(self.spring_model.body_key)}

        robot_indices: list[int] = []
        spring_indices: list[int] = []
        for key, r_idx in robot_by_key.items():
            s_idx = spring_by_key.get(key)
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
                    "No overlapping body keys found. Falling back to index-based robot->spring collider sync."
                )
            else:
                LOGGER.warning("No overlapping body keys found between robot and spring collider models.")

    def _sync_robot_to_spring_colliders(self) -> None:
        """Copy robot body transforms/velocities into spring-model kinematic colliders."""
        if self._robot_to_spring_robot_body_idx.size == 0:
            return

        r_idx = self._robot_to_spring_robot_body_idx
        s_idx = self._robot_to_spring_spring_body_idx

        robot_q = self.robot_state_0.body_q.numpy()
        robot_qd = self.robot_state_0.body_qd.numpy()
        spring_q = self.spring_state_0.body_q.numpy()
        spring_qd = self.spring_state_0.body_qd.numpy()

        spring_q[s_idx] = robot_q[r_idx]
        spring_qd[s_idx] = robot_qd[r_idx]

        self.spring_state_0.body_q.assign(spring_q)
        self.spring_state_0.body_qd.assign(spring_qd)

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
        for _ in range(self.sim_substeps):
            # Robot step.
            self.robot_state_0.clear_forces()
            self.viewer.apply_forces(self.robot_state_0)
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
                    soft_contact_margin=self.spring_soft_contact_margin,
                )
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
            self.viewer.log_points("/spring/controllers", ctrl_wp, ctrl_radii, ctrl_colors, hidden=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = newton.examples.create_parser()
    parser.add_argument("--pkl", type=str, required=True, help="Path to spring-mass PKL file.")
    parser.add_argument("--urdf", type=str, required=True, help="Path to robot URDF file.")
    parser.add_argument("--substeps", type=int, default=20, help="Simulation substeps.")
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
        "--mesh-show-points",
        action="store_true",
        help="Keep rendering object points when mesh visualization is enabled.",
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
        "spring_soft_contact_mu": 0.9,
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
