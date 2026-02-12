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
import pickle
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.examples


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


@dataclass
class SpringMassPKL:
    controller_mask: np.ndarray
    controller_points: np.ndarray
    object_points: np.ndarray
    object_colors: np.ndarray
    object_visibilities: np.ndarray
    object_motions_valid: np.ndarray
    surface_points: np.ndarray
    interior_points: np.ndarray


def load_pkl(path: str) -> SpringMassPKL:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return SpringMassPKL(
        controller_mask=obj["controller_mask"],
        controller_points=obj["controller_points"],
        object_points=obj["object_points"],
        object_colors=obj["object_colors"],
        object_visibilities=obj["object_visibilities"],
        object_motions_valid=obj["object_motions_valid"],
        surface_points=obj["surface_points"],
        interior_points=obj["interior_points"],
    )


def build_knn_springs(points: np.ndarray, k: int, max_dist: float | None) -> np.ndarray:
    """Build undirected springs using k-nearest neighbors."""
    n = points.shape[0]
    k = max(1, min(k, n - 1))
    max_dist2 = None if max_dist is None else max_dist * max_dist
    springs: set[tuple[int, int]] = set()
    for i in range(n):
        d2 = np.sum((points - points[i]) ** 2, axis=1)
        d2[i] = np.inf
        nn = np.argpartition(d2, k)[:k]
        for j in nn:
            if max_dist2 is not None and d2[j] > max_dist2:
                continue
            a, b = (i, j) if i < j else (j, i)
            springs.add((a, b))
    springs_arr = np.array(sorted(springs), dtype=np.int32)
    return springs_arr


def build_model_from_pkl(
    data: SpringMassPKL,
    frame: int,
    scale: float,
    z_offset: float,
    particle_radius: float,
    mass: float,
    spring_ke: float,
    spring_kd: float,
    k_neighbors: int,
    max_spring_length: float | None,
    add_ground: bool,
    use_controllers: bool,
    controller_k: int,
    controller_ke: float,
    controller_kd: float,
) -> newton.Model:
    builder = newton.ModelBuilder()
    if add_ground:
        builder.add_ground_plane()

    points = data.object_points[frame].astype(np.float32) * scale
    min_z = float(points[:, 2].min()) if points.size else 0.0
    if min_z <= 0.0:
        points[:, 2] += (-min_z + z_offset)
    else:
        points[:, 2] += z_offset

    velocities = np.zeros_like(points, dtype=np.float32)

    builder.add_particles(
        pos=points.tolist(),
        vel=velocities.tolist(),
        mass=[mass] * points.shape[0],
        radius=[particle_radius] * points.shape[0],
    )

    springs = build_knn_springs(points, k_neighbors, max_spring_length)
    for i, j in springs:
        builder.add_spring(i, j, spring_ke, spring_kd, control=0.0)

    if use_controllers:
        ctrl = data.controller_points[frame].astype(np.float32) * scale
        ctrl[:, 2] += z_offset
        ctrl_start = builder.particle_count
        builder.add_particles(
            pos=ctrl.tolist(),
            vel=[[0.0, 0.0, 0.0]] * ctrl.shape[0],
            mass=[0.0] * ctrl.shape[0],
            radius=[particle_radius] * ctrl.shape[0],
        )

        # Connect controller points to nearest object points
        ctrl_k = max(1, min(controller_k, points.shape[0]))
        for c in range(ctrl.shape[0]):
            d2 = np.sum((points - ctrl[c]) ** 2, axis=1)
            nn = np.argpartition(d2, ctrl_k)[:ctrl_k]
            for j in nn:
                builder.add_spring(
                    ctrl_start + c,
                    j,
                    controller_ke,
                    controller_kd,
                    control=0.0,
                )

    return builder.finalize()


class Example:
    def __init__(self, viewer, args: argparse.Namespace):
        self.viewer = viewer

        data = load_pkl(args.pkl)
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.model = build_model_from_pkl(
            data=data,
            frame=args.frame,
            scale=args.scale,
            z_offset=args.z_offset,
            particle_radius=args.particle_radius,
            mass=args.mass,
            spring_ke=args.spring_ke,
            spring_kd=args.spring_kd,
            k_neighbors=args.k_neighbors,
            max_spring_length=args.max_spring_length,
            add_ground=not args.no_ground,
            use_controllers=args.use_controllers,
            controller_k=args.controller_k,
            controller_ke=args.controller_ke,
            controller_kd=args.controller_kd,
        )

        if args.disable_particle_contacts:
            self.model.particle_grid = None

        self.solver = newton.solvers.SolverSemiImplicit(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)
        self._frame_camera_on_particles()
        self._init_particle_picking(args)

    def _frame_camera_on_particles(self) -> None:
        if not hasattr(self.viewer, "set_camera"):
            return
        if self.model.particle_q is None:
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
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
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

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


def main() -> None:
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--pkl",
        type=str,
        default="newton/examples/final_data.pkl",
        help="Path to spring-mass PKL file.",
    )
    parser.add_argument("--frame", type=int, default=0, help="Frame index to use for initial positions.")
    parser.add_argument("--scale", type=float, default=0.2, help="Uniform scale for positions.")
    parser.add_argument("--z-offset", type=float, default=0.5, help="Lift all particles above the ground.")
    parser.add_argument("--particle-radius", type=float, default=0.02, help="Particle collision radius.")
    parser.add_argument("--mass", type=float, default=1.0, help="Uniform particle mass.")
    parser.add_argument("--spring-ke", type=float, default=1.0e3, help="Spring stiffness.")
    parser.add_argument("--spring-kd", type=float, default=1.0e1, help="Spring damping.")
    parser.add_argument("--k-neighbors", type=int, default=6, help="kNN springs per particle.")
    parser.add_argument("--max-spring-length", type=float, default=None, help="Max spring length filter.")
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

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
