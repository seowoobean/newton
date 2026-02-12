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

"""Mesh utilities for visualizing point-based cloth data."""

from __future__ import annotations

import logging

import numpy as np

LOGGER = logging.getLogger("phystwin.mesh_utils")

try:  # pragma: no cover - optional dependency
    from scipy.spatial import Delaunay, cKDTree  # type: ignore
except Exception:  # pragma: no cover
    Delaunay = None
    cKDTree = None


def _project_to_plane(points: np.ndarray) -> np.ndarray:
    """Project 3D points onto a best-fit 2D plane using PCA.

    Args:
        points: (N, 3) point cloud.

    Returns:
        (N, 2) projected coordinates.
    """
    centered = points - points.mean(axis=0, keepdims=True)
    _u, s, vh = np.linalg.svd(centered, full_matrices=False)
    if s.size < 2 or s[1] <= 1e-8:
        raise ValueError("Point cloud is nearly colinear; cannot build a 2D projection.")
    basis = vh[:2].T  # (3, 2)
    return centered @ basis


def _estimate_edge_threshold(points: np.ndarray, edge_factor: float) -> float | None:
    """Estimate a reasonable max edge length from nearest-neighbor distances."""
    if edge_factor <= 0.0:
        return None
    if cKDTree is None or points.shape[0] < 3:
        return None
    tree = cKDTree(points)
    k = min(7, points.shape[0])
    dists, _ = tree.query(points, k=k)
    if dists.ndim == 1:
        return None
    nn = dists[:, 1] if dists.shape[1] > 1 else dists[:, 0]
    median = float(np.median(nn))
    if not np.isfinite(median) or median <= 0.0:
        return None
    return median * edge_factor


def _filter_triangles(points: np.ndarray, triangles: np.ndarray, max_edge: float | None) -> np.ndarray:
    """Filter degenerate triangles and optionally prune long edges."""
    if triangles.size == 0:
        return triangles
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    areas = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    mask = areas > 1e-10
    if max_edge is not None and max_edge > 0.0:
        e0 = np.linalg.norm(p0 - p1, axis=1)
        e1 = np.linalg.norm(p1 - p2, axis=1)
        e2 = np.linalg.norm(p2 - p0, axis=1)
        mask &= (e0 <= max_edge) & (e1 <= max_edge) & (e2 <= max_edge)
    return triangles[mask]


def triangulate_points(
    points: np.ndarray,
    method: str = "pca_delaunay",
    max_edge: float | None = None,
    edge_factor: float = 2.5,
) -> np.ndarray:
    """Build a fixed triangle topology from a 3D point cloud.

    Args:
        points: (N, 3) point cloud used as mesh vertices.
        method: Triangulation method. Currently supports "pca_delaunay".
        max_edge: Optional max edge length to prune long triangles.
        edge_factor: Multiplier applied to the median nearest-neighbor distance
            when max_edge is not provided.

    Returns:
        (T, 3) int32 triangle indices.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected points with shape (N, 3).")
    if points.shape[0] < 3:
        return np.zeros((0, 3), dtype=np.int32)
    if not np.isfinite(points).all():
        raise ValueError("Points contain NaNs or infs; cannot triangulate.")

    if method != "pca_delaunay":
        raise ValueError(f"Unsupported triangulation method: {method}")
    if Delaunay is None:
        raise RuntimeError("Triangulation requires scipy (scipy.spatial.Delaunay).")

    uv = _project_to_plane(points)
    try:
        tri = Delaunay(uv, qhull_options="QJ").simplices.astype(np.int32)
    except Exception as exc:
        raise RuntimeError(f"Delaunay triangulation failed: {exc}") from exc

    if max_edge is None:
        max_edge = _estimate_edge_threshold(points, edge_factor=edge_factor)

    tri = _filter_triangles(points, tri, max_edge=max_edge)
    if tri.size == 0:
        LOGGER.warning("Triangulation produced no faces after filtering.")
    return tri
