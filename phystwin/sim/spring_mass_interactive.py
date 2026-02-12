"""Interactive spring-mass simulation with controller group drag."""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import warp as wp
import yaml

import newton
import newton.examples

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from phystwin.mapping.pkl_mapping import SpringMassPKL, SpringMassPKLPair, load_pkl, map_pkl_to_newton
from phystwin.sim.mesh_utils import triangulate_points


LOGGER = logging.getLogger("phystwin.spring_mass_interactive")


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
def apply_pick_force_kernel(
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    pick_index: wp.array(dtype=int),
    pick_target: wp.array(dtype=wp.vec3),
    stiffness: float,
    damping: float,
):
    idx = pick_index[0]
    if idx < 0:
        return
    x = q[idx]
    v = qd[idx]
    target = pick_target[0]
    f[idx] += stiffness * (target - x) - damping * v


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _load_params(path: str | None) -> dict | None:
    if not path:
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _config_value(config: dict, keys: tuple[str, ...], default):
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


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


class Example:
    def __init__(self, viewer, args: argparse.Namespace):
        self.viewer = viewer

        config = _load_config(args.config)
        params = _load_params(args.params)
        if params:
            config.update(params)

        data = load_pkl(args.pkl)
        if isinstance(data, SpringMassPKLPair):
            self.predict_data = data.predict
        else:
            self.predict_data = data

        fps = _config_value(config, ("FPS", "fps"), args.fps)
        substeps = _config_value(config, ("substeps", "num_substeps"), args.substeps)
        self.fps = float(fps)
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = int(substeps)
        self.sim_dt = self.frame_dt / self.sim_substeps

        scale = _config_value(config, ("scale",), args.scale)
        z_offset = _config_value(config, ("z_offset",), args.z_offset)
        particle_radius = _config_value(config, ("particle_radius", "object_radius"), args.particle_radius)
        particle_mass = _config_value(config, ("particle_mass", "mass"), args.mass)
        spring_ke = _config_value(config, ("spring_ke", "init_spring_Y"), args.spring_ke)
        spring_kd = _config_value(config, ("spring_kd", "dashpot_damping"), args.spring_kd)
        k_neighbors = _config_value(config, ("k_neighbors", "object_max_neighbours"), args.k_neighbors)
        neighbor_mode = _config_value(config, ("spring_neighbor_mode",), "knn")
        object_radius = _config_value(config, ("object_radius",), None)
        object_max_neighbours = _config_value(config, ("object_max_neighbours",), None)
        reverse_z = bool(_config_value(config, ("reverse_z",), False))
        use_controllers = bool(_config_value(config, ("use_controllers",), True))
        controller_k = _config_value(config, ("controller_k",), args.controller_k)
        controller_ke = _config_value(config, ("controller_ke",), args.controller_ke)
        controller_kd = _config_value(config, ("controller_kd",), args.controller_kd)
        controller_mass = _config_value(config, ("controller_mass",), args.controller_mass)
        if controller_mass is None:
            controller_mass = particle_mass
        filter_visibility = bool(_config_value(config, ("filter_visibility",), False))
        filter_motion_valid = bool(_config_value(config, ("filter_motion_valid",), False))
        visualize_mesh = bool(_config_value(config, ("visualize_mesh", "mesh_visualize"), args.visualize_mesh))
        mesh_method = str(_config_value(config, ("mesh_method",), args.mesh_method))
        mesh_max_edge = _config_value(config, ("mesh_max_edge",), args.mesh_max_edge)
        if mesh_max_edge is not None:
            mesh_max_edge = float(mesh_max_edge)
        mesh_edge_factor = float(_config_value(config, ("mesh_edge_factor",), args.mesh_edge_factor))
        default_show_points = not visualize_mesh
        mesh_show_points = bool(
            _config_value(config, ("mesh_show_points", "visualize_points"), default_show_points)
        )

        self.sim_time = 0.0
        self.frame_index = 0

        predict_mask = _object_mask(
            self.predict_data,
            frame=0,
            filter_visibility=filter_visibility,
            filter_motion_valid=filter_motion_valid,
        )
        predict_points = self.predict_data.object_points[0][predict_mask]
        scaled_predict = _apply_scale(predict_points, scale, reverse_z)
        z_shift = _compute_z_shift(scaled_predict, z_offset)
        self.scale = float(scale)
        self.z_shift = float(z_shift)
        self.reverse_z = reverse_z

        self.mapping = map_pkl_to_newton(
            data=self.predict_data,
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
            spring_neighbor_mode=str(neighbor_mode),
            object_radius=object_radius,
            object_max_neighbours=object_max_neighbours,
        )
        self.model = self.mapping.model
        if self.model.particle_radius is not None:
            self.object_radius_value = float(self.model.particle_radius.numpy()[0])
        else:
            self.object_radius_value = float(particle_radius)

        self.mesh_enabled = visualize_mesh
        self.mesh_show_points = mesh_show_points
        self.mesh_point_count = int(self.mapping.object_points.shape[0])
        self.mesh_indices_wp = None
        if self.mesh_enabled:
            try:
                tri = triangulate_points(
                    self.mapping.object_points,
                    method=mesh_method,
                    max_edge=mesh_max_edge,
                    edge_factor=mesh_edge_factor,
                )
            except Exception as exc:
                LOGGER.warning("Mesh triangulation failed (%s). Falling back to points.", exc)
                self.mesh_enabled = False
            else:
                if tri.size == 0:
                    LOGGER.warning("Mesh triangulation produced no faces. Falling back to points.")
                    self.mesh_enabled = False
                else:
                    tri_flat = tri.reshape(-1).astype(np.int32)
                    self.mesh_indices_wp = wp.array(tri_flat.tolist(), dtype=wp.int32, device=self.model.device)

        if params:
            if "spring_stiffness" in params and self.model.spring_stiffness is not None:
                arr = np.asarray(params["spring_stiffness"], dtype=np.float32)
                if arr.shape == self.model.spring_stiffness.shape:
                    self.model.spring_stiffness.assign(arr)
            if "spring_damping" in params and self.model.spring_damping is not None:
                arr = np.asarray(params["spring_damping"], dtype=np.float32)
                if arr.shape == self.model.spring_damping.shape:
                    self.model.spring_damping.assign(arr)

        if controller_mass is not None and controller_mass > 0.0:
            ctrl_indices = self.mapping.controller_particle_indices.astype(np.int32)
            if ctrl_indices.size and self.model.particle_mass is not None and self.model.particle_inv_mass is not None:
                mass_np = self.model.particle_mass.numpy()
                inv_np = self.model.particle_inv_mass.numpy()
                mass_np[ctrl_indices] = float(controller_mass)
                inv_np[ctrl_indices] = 1.0 / float(controller_mass)
                self.model.particle_mass.assign(mass_np)
                self.model.particle_inv_mass.assign(inv_np)

        if args.disable_particle_contacts:
            self.model.particle_grid = None

        self.solver = newton.solvers.SolverSemiImplicit(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.controller_drag_enabled = bool(args.enable_controller_drag)
        self.controller_drag_stiffness = args.controller_drag_stiffness
        self.controller_drag_damping = args.controller_drag_damping

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self._frame_camera_on_particles()
        self._init_particle_picking(args)

        self.object_indices = self.mapping.object_particle_indices.astype(np.int32)
        self.controller_indices = self.mapping.controller_particle_indices.astype(np.int32)
        self.controller_indices_wp = None
        self.controller_targets_wp = None
        self.controller_count = 0
        if use_controllers and self.controller_indices.size:
            self.controller_indices = self.controller_indices.copy()
            self.controller_count = self.controller_indices.size
            self.controller_indices_wp = wp.array(
                self.controller_indices.tolist(),
                dtype=int,
                device=self.model.device,
            )
            self.controller_targets_wp = wp.array(
                [wp.vec3(0.0, 0.0, 0.0)] * self.controller_indices.size,
                dtype=wp.vec3,
                device=self.model.device,
            )

        self.controller_dragging = False
        self.controller_drag_depth = None
        self.controller_drag_anchor = None
        self.controller_drag_base = None
        self.controller_drag_targets = None
        self.controller_idle_positions = None
        if self.controller_drag_enabled and self.controller_count > 0:
            positions = self.state_0.particle_q.numpy()[self.controller_indices]
            if positions.size:
                self.controller_idle_positions = positions.copy()

        self.drag_target_color = wp.vec3(1.0, 0.3, 0.9)
        self.drag_line_color = wp.vec3(1.0, 0.6, 0.2)
        self._empty_vec3 = wp.array([], dtype=wp.vec3, device=self.model.device)
        self._empty_float = wp.array([], dtype=float, device=self.model.device)

    def _frame_camera_on_particles(self) -> None:
        if not hasattr(self.viewer, "set_camera"):
            return
        positions = self.model.particle_q.numpy()
        if positions.size == 0:
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
        try:
            self.viewer.set_camera(wp.vec3(*pos.tolist()), pitch, yaw)
        except Exception:
            return

    def _init_particle_picking(self, args: argparse.Namespace) -> None:
        self.pick_enabled = args.enable_particle_picking
        self.pick_stiffness = args.pick_stiffness
        self.pick_damping = args.pick_damping
        self.pick_radius = args.pick_radius
        self.pick_depth = None
        self.pick_index = wp.array([-1], dtype=int, device=self.model.device)
        self.pick_target = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=self.model.device)

        if not self.pick_enabled and not self.controller_drag_enabled:
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

    def _pick_particle_from_ray(self, ray_start: np.ndarray, ray_dir: np.ndarray) -> int:
        positions = self.state_0.particle_q.numpy()
        if positions.size == 0:
            return -1
        d = ray_dir / max(np.linalg.norm(ray_dir), 1.0e-8)
        p = ray_start
        v = positions - p[None, :]
        t = v @ d
        t = np.maximum(t, 0.0)
        proj = p[None, :] + t[:, None] * d[None, :]
        dist2 = np.sum((positions - proj) ** 2, axis=1)
        idx = int(np.argmin(dist2))
        if dist2[idx] > self.pick_radius * self.pick_radius:
            return -1
        self.pick_depth = float(t[idx])
        return idx

    def _start_controller_drag(self, ray_start_np: np.ndarray, ray_dir_np: np.ndarray) -> None:
        if self.controller_count == 0:
            return
        positions = self.state_0.particle_q.numpy()[self.controller_indices]
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
            self.controller_targets_wp.assign(
                [wp.vec3(*pt.tolist()) for pt in self.controller_drag_targets]
            )

    def _on_mouse_press(self, x, y, button, modifiers):
        if not self.pick_enabled and not self.controller_drag_enabled:
            return
        try:
            import pyglet  # noqa: PLC0415
        except Exception:
            return
        if self.controller_drag_enabled:
            if button == pyglet.window.mouse.RIGHT and (modifiers & pyglet.window.key.MOD_SHIFT):
                fb_x, fb_y = self._to_framebuffer_coords(x, y)
                ray_start, ray_dir = self.viewer.camera.get_world_ray(fb_x, fb_y)
                ray_start_np = np.array([ray_start.x, ray_start.y, ray_start.z], dtype=np.float32)
                ray_dir_np = np.array([ray_dir.x, ray_dir.y, ray_dir.z], dtype=np.float32)
                self._start_controller_drag(ray_start_np, ray_dir_np)
                return
        if button != pyglet.window.mouse.RIGHT:
            return
        fb_x, fb_y = self._to_framebuffer_coords(x, y)
        ray_start, ray_dir = self.viewer.camera.get_world_ray(fb_x, fb_y)
        ray_start_np = np.array([ray_start.x, ray_start.y, ray_start.z], dtype=np.float32)
        ray_dir_np = np.array([ray_dir.x, ray_dir.y, ray_dir.z], dtype=np.float32)
        idx = self._pick_particle_from_ray(ray_start_np, ray_dir_np)
        self.pick_index.assign([idx])

        if idx >= 0 and self.pick_depth is not None:
            target = ray_start_np + ray_dir_np * self.pick_depth
            self.pick_target.assign([wp.vec3(*target.tolist())])

    def _on_mouse_release(self, x, y, button, modifiers):
        if self.controller_dragging:
            self.controller_dragging = False
            if self.controller_drag_targets is not None:
                self.controller_idle_positions = self.controller_drag_targets.copy()
        if self.pick_enabled:
            self.pick_index.assign([-1])
            self.pick_depth = None

    def _on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.controller_drag_enabled and self.controller_dragging:
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
            return
        if not self.pick_enabled or self.pick_depth is None:
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
        target = ray_start_np + ray_dir_np * self.pick_depth
        self.pick_target.assign([wp.vec3(*target.tolist())])

    def _update_controller_targets(self) -> None:
        if self.controller_indices_wp is None or self.controller_targets_wp is None:
            return
        if self.controller_count == 0:
            return
        if self.controller_drag_targets is None:
            return
        targets = self.controller_drag_targets
        self.controller_targets_wp.assign([wp.vec3(*pt.tolist()) for pt in targets])
        wp.launch(
            apply_controller_force_kernel,
            dim=self.controller_count,
            inputs=[
                self.state_0.particle_q,
                self.state_0.particle_qd,
                self.state_0.particle_f,
                self.controller_indices_wp,
                self.controller_targets_wp,
                float(self.controller_drag_stiffness),
                float(self.controller_drag_damping),
            ],
            device=self.model.device,
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self._update_controller_targets()
            self.viewer.apply_forces(self.state_0)
            if self.pick_enabled:
                wp.launch(
                    apply_pick_force_kernel,
                    dim=1,
                    inputs=[
                        self.state_0.particle_q,
                        self.state_0.particle_qd,
                        self.state_0.particle_f,
                        self.pick_index,
                        self.pick_target,
                        self.pick_stiffness,
                        self.pick_damping,
                    ],
                    device=self.model.device,
                )
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def _log_drag_marker(self) -> None:
        if not self.controller_drag_enabled or not self.controller_dragging:
            self.viewer.log_lines("/drag/line", self._empty_vec3, self._empty_vec3, self._empty_vec3)
            self.viewer.log_points("/drag/target", self._empty_vec3, self._empty_float, self._empty_vec3)
            return
        if self.controller_drag_targets is None or self.controller_drag_anchor is None:
            return
        center = self.controller_drag_targets.mean(axis=0)
        target = center
        starts = wp.array([wp.vec3(*self.controller_drag_anchor.tolist())], dtype=wp.vec3, device=self.model.device)
        ends = wp.array([wp.vec3(*target.tolist())], dtype=wp.vec3, device=self.model.device)
        colors = wp.array([self.drag_line_color], dtype=wp.vec3, device=self.model.device)
        self.viewer.log_lines("/drag/line", starts, ends, colors, hidden=False)

        point = wp.array([wp.vec3(*target.tolist())], dtype=wp.vec3, device=self.model.device)
        radii = wp.full(1, float(self.model.particle_radius.numpy()[0]) * 1.5, dtype=float, device=self.model.device)
        colors = wp.array([self.drag_target_color], dtype=wp.vec3, device=self.model.device)
        self.viewer.log_points("/drag/target", point, radii, colors, hidden=False)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self._log_predict_points()
        self._log_drag_marker()
        self.viewer.end_frame()

    def _log_predict_points(self) -> None:
        positions = self.state_0.particle_q.numpy()
        if positions.size == 0:
            return
        obj_points = positions[self.object_indices]
        points_wp = wp.array(obj_points.tolist(), dtype=wp.vec3, device=self.model.device)
        if self.mesh_indices_wp is not None and obj_points.shape[0] == self.mesh_point_count:
            self.viewer.log_mesh(
                "/predict/mesh",
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
            device=self.model.device,
        )
        colors_wp = wp.full(obj_points.shape[0], wp.vec3(0.8, 0.7, 0.2), dtype=wp.vec3, device=self.model.device)
        self.viewer.log_points("/predict/points", points_wp, radii_wp, colors_wp, hidden=not show_points)

        if self.controller_indices.size:
            ctrl_points = positions[self.controller_indices]
            ctrl_wp = wp.array(ctrl_points.tolist(), dtype=wp.vec3, device=self.model.device)
            ctrl_radii = wp.full(
                ctrl_points.shape[0],
                float(self.model.particle_radius.numpy()[0]) * 1.25,
                dtype=float,
                device=self.model.device,
            )
            ctrl_colors = wp.full(
                ctrl_points.shape[0],
                wp.vec3(0.2, 1.0, 0.2),
                dtype=wp.vec3,
                device=self.model.device,
            )
            self.viewer.log_points("/predict/controllers", ctrl_wp, ctrl_radii, ctrl_colors, hidden=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--pkl",
        type=str,
        default="newton/examples/final_data.pkl",
        help="Path to spring-mass PKL file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="phystwin/config/cloth.yaml",
        help="Path to YAML config for predict parameters.",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="",
        help="Path to params PKL to override config or apply trained spring arrays.",
    )
    parser.add_argument("--fps", type=float, default=60.0, help="Simulation FPS.")
    parser.add_argument("--substeps", type=int, default=10, help="Simulation substeps.")
    parser.add_argument("--scale", type=float, default=0.2, help="Uniform scale for positions.")
    parser.add_argument("--z-offset", type=float, default=0.5, help="Lift all particles above the ground.")
    parser.add_argument("--particle-radius", type=float, default=0.02, help="Particle collision radius.")
    parser.add_argument("--mass", type=float, default=1.0, help="Uniform particle mass.")
    parser.add_argument("--spring-ke", type=float, default=1.0e3, help="Spring stiffness.")
    parser.add_argument("--spring-kd", type=float, default=1.0e1, help="Spring damping.")
    parser.add_argument("--k-neighbors", type=int, default=6, help="kNN springs per particle.")
    parser.add_argument("--no-ground", action="store_true", help="Disable ground plane.")
    parser.add_argument(
        "--disable-particle-contacts",
        action="store_true",
        help="Disable particle-particle contact (avoid gradient instability).",
    )
    parser.add_argument("--controller-k", type=int, default=1, help="Connections per controller point.")
    parser.add_argument("--controller-ke", type=float, default=1.0e4, help="Controller spring stiffness.")
    parser.add_argument("--controller-kd", type=float, default=1.0e1, help="Controller spring damping.")
    parser.add_argument("--controller-mass", type=float, default=None, help="Controller particle mass.")
    parser.add_argument(
        "--enable-particle-picking",
        action="store_true",
        help="Enable right-click picking for particles (ViewerGL only).",
    )
    parser.add_argument(
        "--enable-controller-drag",
        action="store_true",
        help="Enable Shift+Right drag to move all controller points as a group.",
    )
    parser.add_argument(
        "--controller-drag-stiffness",
        type=float,
        default=2.0e4,
        help="Controller drag spring stiffness.",
    )
    parser.add_argument(
        "--controller-drag-damping",
        type=float,
        default=1.0e2,
        help="Controller drag damping.",
    )
    parser.add_argument("--pick-stiffness", type=float, default=2.0e4, help="Picking spring stiffness.")
    parser.add_argument("--pick-damping", type=float, default=1.0e2, help="Picking spring damping.")
    parser.add_argument("--pick-radius", type=float, default=0.02, help="Picking radius in world units.")
    parser.add_argument(
        "--visualize-mesh",
        action="store_true",
        help="Render object points as a fixed triangle mesh.",
    )
    parser.add_argument(
        "--mesh-method",
        type=str,
        default="pca_delaunay",
        choices=("pca_delaunay",),
        help="Triangulation method for mesh visualization.",
    )
    parser.add_argument(
        "--mesh-max-edge",
        type=float,
        default=None,
        help="Maximum edge length for mesh triangles (auto if None).",
    )
    parser.add_argument(
        "--mesh-edge-factor",
        type=float,
        default=2.5,
        help="Multiplier for median nearest-neighbor distance when auto-sizing mesh edges.",
    )
    parser.add_argument(
        "--mesh-show-points",
        action="store_true",
        help="Keep rendering object points when mesh visualization is enabled.",
    )

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
