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

"""PKL loading and mapping utilities for spring-mass data."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from enum import IntEnum

import logging

import numpy as np
import warp as wp

import newton

LOGGER = logging.getLogger("phystwin.pkl_mapping")
try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover
    cKDTree = None


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


@dataclass
class SpringMassPKLPair:
    gt: SpringMassPKL
    predict: SpringMassPKL


class MappingPklToNewton:
    """Mapping of PKL frame data to a Newton model."""

    class PointKind(IntEnum):
        """Particle role identifiers for PKL-derived particles."""

        OBJECT = 0
        CONTROLLER = 1

    def __init__(
        self,
        model: newton.Model,
        object_points: np.ndarray,
        controller_points: np.ndarray,
        object_particle_indices: np.ndarray,
        controller_particle_indices: np.ndarray,
        pkl_object_indices: np.ndarray,
        pkl_controller_indices: np.ndarray,
        object_colors: np.ndarray,
        object_visibilities: np.ndarray,
        object_motions_valid: np.ndarray,
        controller_mask: np.ndarray,
        surface_points: np.ndarray,
        interior_points: np.ndarray,
    ) -> None:
        self.model = model
        self.object_points = object_points
        self.controller_points = controller_points
        self.object_particle_indices = object_particle_indices
        self.controller_particle_indices = controller_particle_indices
        self.pkl_object_indices = pkl_object_indices
        self.pkl_controller_indices = pkl_controller_indices
        self.object_colors = object_colors
        self.object_visibilities = object_visibilities
        self.object_motions_valid = object_motions_valid
        self.controller_mask = controller_mask
        self.surface_points = surface_points
        self.interior_points = interior_points


def _to_spring_mass_pkl(obj: dict) -> SpringMassPKL:
    """Convert a dict payload into a SpringMassPKL."""
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


def load_pkl(path: str) -> SpringMassPKL | SpringMassPKLPair:
    """Load a PKL file into structured dataclasses."""
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        has_gt = "gt" in obj
        has_predict = "predict" in obj or "pred" in obj
        if has_gt and has_predict:
            predict_key = "predict" if "predict" in obj else "pred"
            return SpringMassPKLPair(
                gt=_to_spring_mass_pkl(obj["gt"]),
                predict=_to_spring_mass_pkl(obj[predict_key]),
            )

    return _to_spring_mass_pkl(obj)


def _register_pkl_particle_attributes(builder: newton.ModelBuilder) -> None:
    """Register per-particle attributes used to track PKL mapping."""
    builder.add_custom_attribute(
        newton.ModelBuilder.CustomAttribute(
            name="point_kind",
            namespace="pkl",
            dtype=wp.int32,
            frequency=newton.Model.AttributeFrequency.PARTICLE,
            default=-1,
        )
    )
    builder.add_custom_attribute(
        newton.ModelBuilder.CustomAttribute(
            name="point_index",
            namespace="pkl",
            dtype=wp.int32,
            frequency=newton.Model.AttributeFrequency.PARTICLE,
            default=-1,
        )
    )
    builder.add_custom_attribute(
        newton.ModelBuilder.CustomAttribute(
            name="object_color",
            namespace="pkl",
            dtype=wp.vec3,
            frequency=newton.Model.AttributeFrequency.PARTICLE,
            default=wp.vec3(1.0, 1.0, 1.0),
        )
    )
    builder.add_custom_attribute(
        newton.ModelBuilder.CustomAttribute(
            name="object_visible",
            namespace="pkl",
            dtype=wp.int32,
            frequency=newton.Model.AttributeFrequency.PARTICLE,
            default=1,
        )
    )
    builder.add_custom_attribute(
        newton.ModelBuilder.CustomAttribute(
            name="object_motion_valid",
            namespace="pkl",
            dtype=wp.int32,
            frequency=newton.Model.AttributeFrequency.PARTICLE,
            default=1,
        )
    )


def build_knn_springs(points: np.ndarray, k: int) -> np.ndarray:
    """Build undirected springs using k-nearest neighbors."""
    n = points.shape[0]
    k = max(1, min(k, n - 1))
    springs: set[tuple[int, int]] = set()
    for i in range(n):
        d2 = np.sum((points - points[i]) ** 2, axis=1)
        d2[i] = np.inf
        nn = np.argpartition(d2, k)[:k]
        for j in nn:
            a, b = (i, j) if i < j else (j, i)
            springs.add((a, b))
    springs_arr = np.array(sorted(springs), dtype=np.int32)
    return springs_arr


def build_radius_springs(
    points: np.ndarray,
    radius: float,
    max_neighbors: int | None,
    min_rest: float = 1e-4,
) -> np.ndarray:
    """Build undirected springs using radius-based neighbors."""
    n = points.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.int32)
    springs: set[tuple[int, int]] = set()

    if cKDTree is not None:
        tree = cKDTree(points)
        for i in range(n):
            idx = tree.query_ball_point(points[i], radius)
            if not idx:
                continue
            idx = [j for j in idx if j != i]
            if max_neighbors is not None and len(idx) > max_neighbors:
                d2 = np.sum((points[idx] - points[i]) ** 2, axis=1)
                order = np.argsort(d2)[:max_neighbors]
                idx = [idx[k] for k in order]
            for j in idx:
                d2 = float(np.sum((points[j] - points[i]) ** 2))
                if d2 <= min_rest * min_rest:
                    continue
                a, b = (i, j) if i < j else (j, i)
                springs.add((a, b))
    else:
        for i in range(n):
            d2 = np.sum((points - points[i]) ** 2, axis=1)
            d2[i] = np.inf
            idx = np.where(d2 <= radius * radius)[0]
            if max_neighbors is not None and idx.size > max_neighbors:
                order = np.argsort(d2[idx])[:max_neighbors]
                idx = idx[order]
            for j in idx:
                if d2[j] <= min_rest * min_rest:
                    continue
                a, b = (i, int(j)) if i < j else (int(j), i)
                springs.add((a, b))

    springs_arr = np.array(sorted(springs), dtype=np.int32)
    return springs_arr


def map_pkl_to_newton(
    data: SpringMassPKL,
    frame: int,
    scale: float,
    z_offset: float,
    particle_radius: float,
    mass: float,
    spring_ke: float,
    spring_kd: float,
    k_neighbors: int,
    add_ground: bool,
    use_controllers: bool,
    controller_k: int,
    controller_ke: float,
    controller_kd: float,
    reverse_z: bool = False,
    requires_grad: bool = False,
    filter_visibility: bool = False,
    filter_motion_valid: bool = False,
    spring_neighbor_mode: str = "knn",
    object_radius: float | None = None,
    object_max_neighbours: int | None = None,
    controller_radius: float | None = None,
    controller_max_neighbours: int | None = None,
    builder: newton.ModelBuilder | None = None,
) -> MappingPklToNewton:
    """Map PKL frame data onto a Newton model and return index mappings.

    Args:
        data: Loaded PKL dataset.
        frame: Frame index for initial state.
        scale: Uniform scale applied to PKL coordinates.
        z_offset: Vertical offset applied after scaling.
        reverse_z: If True, flip the Z axis before scaling/offset.
        particle_radius: Radius assigned to particles.
        mass: Mass assigned to object particles.
        spring_ke: Spring stiffness for object-object springs.
        spring_kd: Spring damping for object-object springs.
        k_neighbors: kNN connections per object particle.
        add_ground: Whether to add a ground plane.
        use_controllers: Whether to add controller particles and springs.
        controller_k: kNN connections per controller particle.
        controller_ke: Spring stiffness for controller-object springs.
        controller_kd: Spring damping for controller-object springs.
        spring_neighbor_mode: Neighbor mode ("knn" or "radius").
        object_radius: Radius for object neighbor search when using radius mode.
        object_max_neighbours: Max neighbors per object in radius mode.
        controller_radius: Radius for controller neighbor search in radius mode.
        controller_max_neighbours: Max neighbors per controller in radius mode.
        filter_visibility: If True, drop object points marked invisible.
        filter_motion_valid: If True, drop object points with invalid motion.
        requires_grad: If True, enable gradient tracking in the Newton model.

    Returns:
        MappingPklToNewton: Mapping information and the built model.
    """
    if frame < 0 or frame >= data.object_points.shape[0]:
        raise ValueError(f"Frame index {frame} is out of range for {data.object_points.shape[0]} frames.")

    object_points = np.asarray(data.object_points[frame], dtype=np.float32)
    object_colors = np.asarray(data.object_colors[frame], dtype=np.float32) if data.object_colors.size else None
    object_visibilities = (
        np.asarray(data.object_visibilities[frame], dtype=bool) if data.object_visibilities.size else None
    )
    object_motions_valid = (
        np.asarray(data.object_motions_valid[frame], dtype=bool) if data.object_motions_valid.size else None
    )

    if object_visibilities is None:
        object_visibilities = np.ones(object_points.shape[0], dtype=bool)
    if object_motions_valid is None:
        object_motions_valid = np.ones(object_points.shape[0], dtype=bool)
    if object_colors is None:
        object_colors = np.ones_like(object_points, dtype=np.float32)

    mask = np.ones(object_points.shape[0], dtype=bool)
    if filter_visibility:
        mask &= object_visibilities
    if filter_motion_valid:
        mask &= object_motions_valid

    pkl_object_indices = np.flatnonzero(mask).astype(np.int32)
    object_points = object_points[mask]
    object_colors = object_colors[mask]
    object_visibilities = object_visibilities[mask]
    object_motions_valid = object_motions_valid[mask]

    if reverse_z and object_points.size:
        object_points = object_points.copy()
        object_points[:, 2] *= -1.0
    object_points = object_points * scale
    min_z = float(object_points[:, 2].min()) if object_points.size else 0.0
    if min_z <= 0.0:
        object_points[:, 2] += (-min_z + z_offset)
    else:
        object_points[:, 2] += z_offset

    velocities = np.zeros_like(object_points, dtype=np.float32)

    if builder is None:
        builder = newton.ModelBuilder()
    if add_ground:
        builder.add_ground_plane()

    _register_pkl_particle_attributes(builder)
    builder.add_particles(
        pos=object_points.tolist(),
        vel=velocities.tolist(),
        mass=[mass] * object_points.shape[0],
        radius=[particle_radius] * object_points.shape[0],
        custom_attributes={
            "pkl:point_kind": [MappingPklToNewton.PointKind.OBJECT] * object_points.shape[0],
            "pkl:point_index": pkl_object_indices.tolist(),
            "pkl:object_color": object_colors.tolist(),
            "pkl:object_visible": object_visibilities.astype(np.int32).tolist(),
            "pkl:object_motion_valid": object_motions_valid.astype(np.int32).tolist(),
        },
    )

    neighbor_mode = spring_neighbor_mode.lower()
    if neighbor_mode == "radius":
        effective_radius = object_radius
        if effective_radius is None:
            neighbor_mode = "knn"
        else:
            springs = build_radius_springs(
                object_points,
                float(effective_radius),
                object_max_neighbours if object_max_neighbours is not None else k_neighbors,
            )
    if neighbor_mode != "radius":
        springs = build_knn_springs(object_points, k_neighbors)
    for i, j in springs:
        builder.add_spring(i, j, spring_ke, spring_kd, control=0.0)

    controller_points = np.zeros((0, 3), dtype=np.float32)
    controller_particle_indices = np.zeros((0,), dtype=np.int32)
    pkl_controller_indices = np.zeros((0,), dtype=np.int32)
    controller_spring_counts: np.ndarray | None = None

    if use_controllers and data.controller_points.size:
        controller_points = np.asarray(data.controller_points[frame], dtype=np.float32)
        if reverse_z and controller_points.size:
            controller_points = controller_points.copy()
            controller_points[:, 2] *= -1.0
        controller_points = controller_points * scale
        controller_points[:, 2] += z_offset
        ctrl_start = builder.particle_count
        controller_particle_indices = np.arange(
            ctrl_start, ctrl_start + controller_points.shape[0], dtype=np.int32
        )
        pkl_controller_indices = np.arange(controller_points.shape[0], dtype=np.int32)
        controller_spring_counts = np.zeros(controller_points.shape[0], dtype=np.int32)

        builder.add_particles(
            pos=controller_points.tolist(),
            vel=[[0.0, 0.0, 0.0]] * controller_points.shape[0],
            mass=[0.0] * controller_points.shape[0],
            radius=[particle_radius] * controller_points.shape[0],
            custom_attributes={
                "pkl:point_kind": [MappingPklToNewton.PointKind.CONTROLLER] * controller_points.shape[0],
                "pkl:point_index": pkl_controller_indices.tolist(),
            },
        )

        # Connect controller points to nearest object points
        if object_points.shape[0] > 0 and controller_points.shape[0] > 0:
            def add_controller_spring(ctrl_idx: int, obj_idx: int) -> None:
                builder.add_spring(
                    ctrl_start + ctrl_idx,
                    int(obj_idx),
                    controller_ke,
                    controller_kd,
                    control=0.0,
                )
                controller_spring_counts[ctrl_idx] += 1

            ctrl_k = max(1, min(controller_k, object_points.shape[0]))
            for c in range(controller_points.shape[0]):
                d2 = np.sum((object_points - controller_points[c]) ** 2, axis=1)
                nn = np.argpartition(d2, ctrl_k)[:ctrl_k]
                for j in nn:
                    add_controller_spring(c, j)


    model = builder.finalize(requires_grad=requires_grad)

    object_particle_indices = np.arange(object_points.shape[0], dtype=np.int32)
    return MappingPklToNewton(
        model=model,
        object_points=object_points,
        controller_points=controller_points,
        object_particle_indices=object_particle_indices,
        controller_particle_indices=controller_particle_indices,
        pkl_object_indices=pkl_object_indices,
        pkl_controller_indices=pkl_controller_indices,
        object_colors=object_colors,
        object_visibilities=object_visibilities,
        object_motions_valid=object_motions_valid,
        controller_mask=np.asarray(data.controller_mask, dtype=np.int32),
        surface_points=np.asarray(data.surface_points, dtype=np.float32),
        interior_points=np.asarray(data.interior_points, dtype=np.float32),
    )


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
    add_ground: bool,
    use_controllers: bool,
    controller_k: int,
    controller_ke: float,
    controller_kd: float,
    reverse_z: bool = False,
    requires_grad: bool = False,
    spring_neighbor_mode: str = "knn",
    object_radius: float | None = None,
    object_max_neighbours: int | None = None,
    controller_radius: float | None = None,
    controller_max_neighbours: int | None = None,
) -> newton.Model:
    """Build a Newton model from PKL data for a single frame."""
    return map_pkl_to_newton(
        data=data,
        frame=frame,
        scale=scale,
        z_offset=z_offset,
        reverse_z=reverse_z,
        requires_grad=requires_grad,
        particle_radius=particle_radius,
        mass=mass,
        spring_ke=spring_ke,
        spring_kd=spring_kd,
        k_neighbors=k_neighbors,
        add_ground=add_ground,
        use_controllers=use_controllers,
        controller_k=controller_k,
        controller_ke=controller_ke,
        controller_kd=controller_kd,
        spring_neighbor_mode=spring_neighbor_mode,
        object_radius=object_radius,
        object_max_neighbours=object_max_neighbours,
        controller_radius=controller_radius,
        controller_max_neighbours=controller_max_neighbours,
    ).model
