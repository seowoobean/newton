# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load spring-mass data from a PKL file and simulate it in Newton."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

import numpy as np
import warp as wp

import newton
import newton.examples

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from phystwin.mapping.pkl_mapping import (
    MappingPklToNewton,
    SpringMassPKL,
    SpringMassPKLPair,
    build_model_from_pkl,
    load_pkl,
    map_pkl_to_newton,
)
from phystwin.sim.mesh_utils import triangulate_points


LOGGER = logging.getLogger("phystwin.spring_mass_from_pkl")


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


def _config_value(config: dict, keys: tuple[str, ...], default):
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _parse_color(value, default: tuple[float, float, float]) -> np.ndarray:
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


def _object_mask(
    data: SpringMassPKL,
    frame: int,
    filter_visibility: bool,
    filter_motion_valid: bool,
) -> np.ndarray:
    mask = np.ones(data.object_points.shape[1], dtype=bool)
    if filter_visibility and data.object_visibilities.size:
        mask &= data.object_visibilities[frame].astype(bool)
    if filter_motion_valid and data.object_motions_valid.size:
        mask &= data.object_motions_valid[frame].astype(bool)
    return mask


def _compute_z_shift(points: np.ndarray, z_offset: float) -> float:
    if points.size == 0:
        return z_offset
    min_z = float(points[:, 2].min())
    if min_z <= 0.0:
        return -min_z + z_offset
    return z_offset


def _apply_scale(points: np.ndarray, scale: float, reverse_z: bool) -> np.ndarray:
    scaled = points.astype(np.float32) * scale
    if reverse_z and scaled.size:
        scaled[..., 2] *= -1.0
    return scaled


def _apply_scale_z(points: np.ndarray, scale: float, z_shift: float, reverse_z: bool) -> np.ndarray:
    scaled = _apply_scale(points, scale, reverse_z)
    if scaled.size:
        scaled[..., 2] += z_shift
    return scaled


class Example:
    def __init__(self, viewer, args: argparse.Namespace):
        self.viewer = viewer

        config = _load_config(args.config)
        data = load_pkl(args.pkl)
        if isinstance(data, SpringMassPKLPair):
            self.predict_data = data.predict
            self.gt_data = data.gt
        else:
            self.predict_data = data
            self.gt_data = data

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
        controller_radius = _config_value(config, ("controller_radius",), None)
        controller_max_neighbours = _config_value(config, ("controller_max_neighbours",), None)
        reverse_z = bool(_config_value(config, ("reverse_z",), False))
        default_use_controllers = args.use_controllers or self.predict_data.controller_points.size > 0
        use_controllers = bool(_config_value(config, ("use_controllers",), default_use_controllers))
        controller_k = _config_value(config, ("controller_k", "controller_max_neighbours"), args.controller_k)
        controller_ke = _config_value(config, ("controller_ke",), args.controller_ke)
        controller_kd = _config_value(config, ("controller_kd",), args.controller_kd)
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
        mesh_show_gt = bool(_config_value(config, ("mesh_show_gt",), args.mesh_show_gt))

        self.sim_time = 0.0
        predict_frames = int(self.predict_data.object_points.shape[0])
        gt_frames = int(self.gt_data.object_points.shape[0])
        self.num_frames = min(predict_frames, gt_frames) if gt_frames else predict_frames
        self.frame_index = args.frame % self.num_frames if self.num_frames else 0

        predict_mask = _object_mask(
            self.predict_data,
            frame=args.frame,
            filter_visibility=filter_visibility,
            filter_motion_valid=filter_motion_valid,
        )
        predict_points = self.predict_data.object_points[args.frame][predict_mask]
        scaled_predict = _apply_scale(predict_points, scale, reverse_z)
        z_shift = _compute_z_shift(scaled_predict, z_offset)
        self.z_shift = z_shift
        self.scale = float(scale)
        self.reverse_z = reverse_z
        self.compare_axis = _config_value(config, ("compare_axis",), args.compare_axis)
        self.compare_offset = _config_value(config, ("compare_offset",), args.compare_offset)
        self.compare_symmetric = bool(_config_value(config, ("compare_symmetric",), args.compare_symmetric))

        gt_color = _parse_color(
            _config_value(config, ("gt_color",), args.gt_color),
            default=(0.2, 0.6, 1.0),
        )
        gt_controller_color = _parse_color(
            _config_value(config, ("gt_controller_color",), args.gt_controller_color),
            default=(1.0, 0.4, 0.4),
        )
        predict_color = _parse_color(
            _config_value(config, ("predict_color",), args.predict_color),
            default=(0.8, 0.7, 0.2),
        )
        predict_controller_color = _parse_color(
            _config_value(config, ("predict_controller_color",), args.predict_controller_color),
            default=(0.2, 1.0, 0.2),
        )
        self.gt_color = wp.vec3(*gt_color.tolist())
        self.gt_controller_color = wp.vec3(*gt_controller_color.tolist())
        self.predict_color = wp.vec3(*predict_color.tolist())
        self.predict_controller_color = wp.vec3(*predict_controller_color.tolist())

        self.mapping = map_pkl_to_newton(
            data=self.predict_data,
            frame=args.frame,
            scale=scale,
            z_offset=z_offset,
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
            controller_radius=controller_radius,
            controller_max_neighbours=controller_max_neighbours,
        )
        self.model = self.mapping.model
        self.object_indices = self.mapping.object_particle_indices.astype(np.int32)
        if self.model.particle_radius is not None:
            self.object_radius_value = float(self.model.particle_radius.numpy()[0])
        else:
            self.object_radius_value = float(particle_radius)

        self.mesh_enabled = visualize_mesh
        self.mesh_show_points = mesh_show_points
        self.mesh_show_gt = mesh_show_gt
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

        if args.disable_particle_contacts:
            self.model.particle_grid = None

        self.solver = newton.solvers.SolverSemiImplicit(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self._init_gt_cache(
            particle_radius=float(particle_radius),
            filter_visibility=filter_visibility,
            filter_motion_valid=filter_motion_valid,
        )
        self._frame_camera_on_particles()
        self._init_particle_picking(args)

        self.controller_indices = self.mapping.controller_particle_indices.astype(np.int32)
        self.controller_indices_wp = None
        self.controller_targets_wp = None
        if use_controllers and self.controller_indices.size:
            controller_count = min(self.controller_indices.size, self.gt_controller_points.shape[1])
            self.controller_indices = self.controller_indices[:controller_count]
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

    def _frame_camera_on_particles(self) -> None:
        if not hasattr(self.viewer, "set_camera"):
            return
        if self.model.particle_q is None:
            return
        positions = self.model.particle_q.numpy()
        if positions.size == 0:
            return
        all_positions = [positions]
        if hasattr(self, "gt_points_by_frame") and self.gt_points_by_frame:
            all_positions.append(self.gt_points_by_frame[self.frame_index] + self.gt_visual_offset)
        merged = np.concatenate(all_positions, axis=0)
        center = merged.mean(axis=0)
        extents = merged.max(axis=0) - merged.min(axis=0)
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

        if not self.pick_enabled:
            return
        if not hasattr(self.viewer, "renderer") or not hasattr(self.viewer, "camera"):
            return

        renderer = self.viewer.renderer
        if not hasattr(renderer, "register_mouse_press"):
            return

        renderer.register_mouse_press(self._on_mouse_press)
        renderer.register_mouse_release(self._on_mouse_release)
        renderer.register_mouse_drag(self._on_mouse_drag)

    def _init_gt_cache(
        self,
        particle_radius: float,
        filter_visibility: bool,
        filter_motion_valid: bool,
    ) -> None:
        self.gt_points_by_frame: list[np.ndarray] = []
        for frame in range(self.num_frames):
            mask = _object_mask(
                self.gt_data,
                frame=frame,
                filter_visibility=filter_visibility,
                filter_motion_valid=filter_motion_valid,
            )
            points = self.gt_data.object_points[frame][mask]
            points = _apply_scale_z(points, self.scale, self.z_shift, self.reverse_z)
            self.gt_points_by_frame.append(points)

        ctrl_points = self.gt_data.controller_points
        if ctrl_points.size == 0 and self.predict_data.controller_points.size:
            ctrl_points = self.predict_data.controller_points
        if ctrl_points.size:
            ctrl_points = ctrl_points[: self.num_frames]
            self.gt_controller_points = _apply_scale_z(ctrl_points, self.scale, self.z_shift, self.reverse_z)
        else:
            self.gt_controller_points = np.zeros((self.num_frames, 0, 3), dtype=np.float32)

        if self.compare_offset is None:
            if self.mapping.object_points.size:
                extents = self.mapping.object_points.max(axis=0) - self.mapping.object_points.min(axis=0)
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
            self.gt_visual_offset = axis_vec * half
            self.pred_visual_offset = -axis_vec * half
        else:
            self.gt_visual_offset = axis_vec * offset
            self.pred_visual_offset = np.zeros(3, dtype=np.float32)

        self.gt_radius = particle_radius

    def _update_controller_targets(self, frame_index: int, next_index: int, alpha: float) -> None:
        if self.controller_indices_wp is None or self.controller_targets_wp is None:
            return
        if self.gt_controller_points.shape[1] == 0:
            return

        ctrl_points_a = self.gt_controller_points[frame_index]
        ctrl_points_b = self.gt_controller_points[next_index]
        ctrl_count = min(ctrl_points_a.shape[0], self.controller_indices.size)
        if ctrl_count == 0:
            return

        targets = (1.0 - alpha) * ctrl_points_a[:ctrl_count] + alpha * ctrl_points_b[:ctrl_count]
        self.controller_targets_wp.assign([wp.vec3(*pt.tolist()) for pt in targets])
        wp.launch(
            set_controller_points_kernel,
            dim=ctrl_count,
            inputs=[
                self.state_0.particle_q,
                self.state_0.particle_qd,
                self.controller_indices_wp,
                self.controller_targets_wp,
                self.sim_dt,
            ],
            device=self.model.device,
        )

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
        d = ray_dir / np.linalg.norm(ray_dir)
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

    def _on_mouse_press(self, x, y, button, modifiers):
        if not self.pick_enabled:
            return
        try:
            import pyglet  # noqa: PLC0415
        except Exception:
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
        self.pick_index.assign([-1])
        self.pick_depth = None

    def _on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
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

    def simulate(self):
        frame_index = self.frame_index % self.num_frames
        next_index = (frame_index + 1) % self.num_frames
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            alpha = float(substep + 1) / float(self.sim_substeps)
            self._update_controller_targets(frame_index, next_index, alpha)
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
        self.frame_index = (self.frame_index + 1) % self.num_frames

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self._log_predict_points()
        self._log_gt_points()
        self.viewer.end_frame()

    def _log_predict_points(self) -> None:
        positions = self.state_0.particle_q.numpy()
        if positions.size == 0:
            return
        if self.object_indices.size:
            obj_positions = positions[self.object_indices]
        else:
            obj_positions = positions
        pred_points = obj_positions + self.pred_visual_offset[None, :]
        points_wp = wp.array(pred_points.tolist(), dtype=wp.vec3, device=self.model.device)
        if self.mesh_indices_wp is not None and pred_points.shape[0] == self.mesh_point_count:
            self.viewer.log_mesh(
                "/predict/mesh",
                points_wp,
                self.mesh_indices_wp,
                hidden=not self.mesh_enabled,
                backface_culling=False,
            )
        show_points = (not self.mesh_enabled) or self.mesh_show_points
        radii_wp = wp.full(
            pred_points.shape[0],
            self.object_radius_value,
            dtype=float,
            device=self.model.device,
        )
        colors_wp = wp.full(pred_points.shape[0], self.predict_color, dtype=wp.vec3, device=self.model.device)
        self.viewer.log_points("/predict/points", points_wp, radii_wp, colors_wp, hidden=not show_points)

        if self.controller_indices.size:
            ctrl_points = positions[self.controller_indices] + self.pred_visual_offset[None, :]
            ctrl_wp = wp.array(ctrl_points.tolist(), dtype=wp.vec3, device=self.model.device)
            ctrl_radii = wp.full(
                ctrl_points.shape[0],
                float(self.model.particle_radius.numpy()[0]) * 1.25,
                dtype=float,
                device=self.model.device,
            )
            ctrl_colors = wp.full(
                ctrl_points.shape[0],
                self.predict_controller_color,
                dtype=wp.vec3,
                device=self.model.device,
            )
            self.viewer.log_points("/predict/controllers", ctrl_wp, ctrl_radii, ctrl_colors, hidden=False)

    def _log_gt_points(self) -> None:
        if not self.gt_points_by_frame:
            return
        gt_points = self.gt_points_by_frame[self.frame_index] + self.gt_visual_offset
        if gt_points.size == 0:
            return
        points_wp = wp.array(gt_points.tolist(), dtype=wp.vec3, device=self.model.device)
        if (
            self.mesh_show_gt
            and self.mesh_indices_wp is not None
            and gt_points.shape[0] == self.mesh_point_count
        ):
            self.viewer.log_mesh(
                "/gt/mesh",
                points_wp,
                self.mesh_indices_wp,
                hidden=not self.mesh_enabled,
                backface_culling=False,
            )
        if self.mesh_enabled and self.mesh_show_gt:
            show_points = self.mesh_show_points
        else:
            show_points = True
        radii_wp = wp.full(gt_points.shape[0], self.gt_radius, dtype=float, device=self.model.device)
        colors_wp = wp.full(gt_points.shape[0], self.gt_color, dtype=wp.vec3, device=self.model.device)
        self.viewer.log_points("/gt/points", points_wp, radii_wp, colors_wp, hidden=not show_points)

        if self.gt_controller_points.shape[1] == 0:
            return
        ctrl_points = self.gt_controller_points[self.frame_index] + self.gt_visual_offset
        if ctrl_points.size == 0:
            return
        ctrl_wp = wp.array(ctrl_points.tolist(), dtype=wp.vec3, device=self.model.device)
        ctrl_radii = wp.full(ctrl_points.shape[0], self.gt_radius * 1.25, dtype=float, device=self.model.device)
        ctrl_colors = wp.full(ctrl_points.shape[0], self.gt_controller_color, dtype=wp.vec3, device=self.model.device)
        self.viewer.log_points("/gt/controllers", ctrl_wp, ctrl_radii, ctrl_colors, hidden=False)


def main() -> None:
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
    parser.add_argument("--fps", type=float, default=60.0, help="Simulation FPS.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to use for initial positions.")
    parser.add_argument("--scale", type=float, default=0.2, help="Uniform scale for positions.")
    parser.add_argument("--z-offset", type=float, default=0.5, help="Lift all particles above the ground.")
    parser.add_argument("--particle-radius", type=float, default=0.02, help="Particle collision radius.")
    parser.add_argument("--mass", type=float, default=1.0, help="Uniform particle mass.")
    parser.add_argument("--spring-ke", type=float, default=1.0e3, help="Spring stiffness.")
    parser.add_argument("--spring-kd", type=float, default=1.0e1, help="Spring damping.")
    parser.add_argument("--k-neighbors", type=int, default=6, help="kNN springs per particle.")
    parser.add_argument("--substeps", type=int, default=10, help="Simulation substeps.")
    parser.add_argument("--no-ground", action="store_true", help="Disable ground plane.")
    parser.add_argument(
        "--disable-particle-contacts",
        action="store_true",
        help="Disable particle-particle contact (avoid gradient instability).",
    )
    parser.add_argument(
        "--use-controllers",
        action="store_true",
        help="Add controller points as fixed particles and connect with springs.",
    )
    parser.add_argument("--controller-k", type=int, default=1, help="Connections per controller point.")
    parser.add_argument("--controller-ke", type=float, default=1.0e4, help="Controller spring stiffness.")
    parser.add_argument("--controller-kd", type=float, default=1.0e1, help="Controller spring damping.")
    parser.add_argument(
        "--enable-particle-picking",
        action="store_true",
        help="Enable right-click picking for particles (ViewerGL only).",
    )
    parser.add_argument("--pick-stiffness", type=float, default=2.0e4, help="Picking spring stiffness.")
    parser.add_argument("--pick-damping", type=float, default=1.0e2, help="Picking spring damping.")
    parser.add_argument("--pick-radius", type=float, default=0.02, help="Picking radius in world units.")
    parser.add_argument(
        "--compare-offset",
        type=float,
        default=None,
        help="Offset distance to separate GT and predict (auto if None).",
    )
    parser.add_argument(
        "--compare-axis",
        type=str,
        default="x",
        choices=("x", "y"),
        help="Axis along which GT is offset for comparison.",
    )
    parser.add_argument(
        "--compare-symmetric",
        action="store_true",
        help="Split the GT/predict offset to opposite sides.",
    )
    parser.add_argument(
        "--gt-color",
        type=str,
        default="0.2,0.6,1.0",
        help="GT object color as r,g,b in [0,1].",
    )
    parser.add_argument(
        "--gt-controller-color",
        type=str,
        default="1.0,0.4,0.4",
        help="GT controller color as r,g,b in [0,1].",
    )
    parser.add_argument(
        "--predict-color",
        type=str,
        default="0.8,0.7,0.2",
        help="Predict color as r,g,b in [0,1].",
    )
    parser.add_argument(
        "--predict-controller-color",
        type=str,
        default="0.2,1.0,0.2",
        help="Predict controller color as r,g,b in [0,1].",
    )
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
    parser.add_argument(
        "--mesh-show-gt",
        action="store_true",
        help="Render GT mesh if point counts match the predicted mesh.",
    )

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
