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

"""Inspect model parameter arrays for a simple cloth setup."""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

import newton


def _summarize_array(name: str, array: wp.array | None, full: bool) -> None:
    if array is None:
        print(f"{name}: None")
        return
    data = array.numpy()
    if full:
        print(f"{name}:\n{data}")
        return
    if data.size == 0:
        print(f"{name}: empty")
        return
    if data.dtype.fields is not None:
        print(f"{name}: shape={data.shape}, dtype={data.dtype} (structured)")
        return
    flat = data.reshape(-1)
    try:
        min_v = flat.min()
        max_v = flat.max()
        mean_v = flat.mean()
    except Exception:
        print(f"{name}: shape={data.shape}, dtype={data.dtype}")
        return
    print(
        f"{name}: shape={data.shape}, dtype={data.dtype}, "
        f"min={min_v:.6g}, max={max_v:.6g}, mean={mean_v:.6g}"
    )


def _list_all_wp_arrays(model: newton.Model, full: bool) -> None:
    print("\n== All Model Warp Arrays ==")
    items = []
    for key, value in model.__dict__.items():
        if isinstance(value, wp.array):
            items.append((key, value))
    if not items:
        print("(no wp.array fields found)")
        return
    for key, value in sorted(items, key=lambda kv: kv[0]):
        _summarize_array(key, value, full)


def build_cloth_model(args: argparse.Namespace) -> newton.Model:
    """Build a simple cloth model and return the finalized model."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 4.0),
        rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=args.width,
        dim_y=args.height,
        cell_x=args.cell_x,
        cell_y=args.cell_y,
        mass=args.mass,
        fix_left=True,
        tri_ke=args.tri_ke,
        tri_ka=args.tri_ka,
        tri_kd=args.tri_kd,
        tri_drag=args.tri_drag,
        tri_lift=args.tri_lift,
        edge_ke=args.edge_ke,
        edge_kd=args.edge_kd,
        add_springs=args.add_springs,
        spring_ke=args.spring_ke,
        spring_kd=args.spring_kd,
        particle_radius=args.particle_radius,
    )

    return builder.finalize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect model parameter arrays.")
    parser.add_argument("--width", type=int, default=16, help="Cloth resolution in x.")
    parser.add_argument("--height", type=int, default=8, help="Cloth resolution in y.")
    parser.add_argument("--cell-x", type=float, default=0.1, help="Cell size in x.")
    parser.add_argument("--cell-y", type=float, default=0.1, help="Cell size in y.")
    parser.add_argument("--mass", type=float, default=0.1, help="Mass per particle.")
    parser.add_argument("--tri-ke", type=float, default=1.0e3, help="Triangle elastic stiffness.")
    parser.add_argument("--tri-ka", type=float, default=1.0e3, help="Triangle area stiffness.")
    parser.add_argument("--tri-kd", type=float, default=1.0e-1, help="Triangle damping.")
    parser.add_argument("--tri-drag", type=float, default=0.0, help="Triangle drag.")
    parser.add_argument("--tri-lift", type=float, default=0.0, help="Triangle lift.")
    parser.add_argument("--edge-ke", type=float, default=10.0, help="Edge (bending) stiffness.")
    parser.add_argument("--edge-kd", type=float, default=0.0, help="Edge (bending) damping.")
    parser.add_argument("--add-springs", action="store_true", help="Add structural springs.")
    parser.add_argument("--spring-ke", type=float, default=1.0e3, help="Spring stiffness.")
    parser.add_argument("--spring-kd", type=float, default=1.0e1, help="Spring damping.")
    parser.add_argument("--particle-radius", type=float, default=0.05, help="Particle radius.")
    parser.add_argument("--full", action="store_true", help="Print full arrays.")
    parser.add_argument(
        "--all-arrays",
        action="store_true",
        help="Print summaries for all wp.array fields on the model.",
    )
    args = parser.parse_args()

    model = build_cloth_model(args)

    print("== Particle Parameters ==")
    _summarize_array("particle_mass", model.particle_mass, args.full)
    _summarize_array("particle_radius", model.particle_radius, args.full)

    print("\n== Spring Parameters ==")
    _summarize_array("spring_stiffness", model.spring_stiffness, args.full)
    _summarize_array("spring_damping", model.spring_damping, args.full)
    _summarize_array("spring_rest_length", model.spring_rest_length, args.full)

    print("\n== Triangle (Cloth) Parameters ==")
    _summarize_array("tri_materials", model.tri_materials, args.full)
    _summarize_array("tri_areas", model.tri_areas, args.full)

    print("\n== Edge (Bending) Parameters ==")
    _summarize_array("edge_bending_properties", model.edge_bending_properties, args.full)
    _summarize_array("edge_rest_angle", model.edge_rest_angle, args.full)

    if args.all_arrays:
        _list_all_wp_arrays(model, args.full)


if __name__ == "__main__":
    main()
