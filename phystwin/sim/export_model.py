"""Export a single spring-mass PKL payload from a source PKL file."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from phystwin.mapping.pkl_mapping import SpringMassPKL, SpringMassPKLPair, load_pkl


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


def _apply_param_overrides(config: dict, params: dict | None, keys: tuple[str, ...]) -> None:
    if not params:
        return
    for key in keys:
        if key not in params:
            continue
        value = params[key]
        if value is None:
            continue
        if np.isscalar(value):
            config[key] = value


def _keep_first_frame(points: np.ndarray, nframes: int) -> np.ndarray:
    arr = np.asarray(points)
    if arr.ndim >= 1 and arr.shape[0] == nframes:
        return arr[:1].copy()
    return arr


def _filter_params_for_interactive(params: dict) -> dict:
    return {k: v for k, v in params.items() if k in INTERACTIVE_PARAM_KEYS and v is not None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkl",
        type=str,
        required=True,
        help="Path to spring-mass PKL file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="phystwin/config/cloth.yaml",
        help="Path to YAML config for model parameters.",
    )
    parser.add_argument(
        "--out-pkl",
        type=str,
        required=True,
        help="Output PKL path (single SpringMassPKL payload).",
    )
    parser.add_argument(
        "--apply-transform",
        action="store_true",
        help="Bake scale/reverse_z/z_offset into PKL point arrays.",
    )
    parser.add_argument(
        "--optimal-params",
        type=str,
        default="phystwin/experiments_optimization/double_lift_cloth_1/optimal_params.pkl",
        help="Path to CMA optimal params PKL (optional).",
    )
    parser.add_argument(
        "--best-params",
        type=str,
        default="phystwin/experiments/double_lift_cloth_1/train/best_params.pkl",
        help="Path to best params PKL. All entries are embedded under 'params' in the output PKL.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    optimal_params = _load_params(args.optimal_params)
    best_params = _load_params(args.best_params)
    override_keys = (
        "scale",
        "z_offset",
        "reverse_z",
    )
    _apply_param_overrides(config, optimal_params, override_keys)
    _apply_param_overrides(config, best_params, override_keys)

    data = load_pkl(args.pkl)
    if isinstance(data, SpringMassPKLPair):
        data = data.predict

    scale = float(config.get("scale", 1.0))
    z_offset = float(config.get("z_offset", 0.0))
    reverse_z = bool(config.get("reverse_z", False))

    predict_points = np.asarray(data.object_points[0], dtype=np.float32)
    scaled_predict = _apply_scale(predict_points, scale, reverse_z)
    z_shift = _compute_z_shift(scaled_predict, z_offset)

    export_data = data
    if args.apply_transform:
        def _transform(points: np.ndarray) -> np.ndarray:
            if points.size == 0:
                return points
            pts = _apply_scale(points.reshape(-1, 3), scale, reverse_z)
            if pts.size:
                pts = pts.copy()
                pts[:, 2] += z_shift
            return pts.reshape(points.shape)

        export_data = SpringMassPKL(
            controller_mask=export_data.controller_mask,
            controller_points=_transform(export_data.controller_points),
            object_points=_transform(export_data.object_points),
            object_colors=export_data.object_colors,
            object_visibilities=export_data.object_visibilities,
            object_motions_valid=export_data.object_motions_valid,
            surface_points=_transform(export_data.surface_points),
            interior_points=_transform(export_data.interior_points),
        )

    nframes = int(export_data.object_points.shape[0]) if np.asarray(export_data.object_points).ndim >= 1 else 1
    export_data = SpringMassPKL(
        controller_mask=np.asarray(export_data.controller_mask),
        controller_points=_keep_first_frame(export_data.controller_points, nframes),
        object_points=_keep_first_frame(export_data.object_points, nframes),
        object_colors=_keep_first_frame(export_data.object_colors, nframes),
        object_visibilities=_keep_first_frame(export_data.object_visibilities, nframes),
        object_motions_valid=_keep_first_frame(export_data.object_motions_valid, nframes),
        surface_points=_keep_first_frame(export_data.surface_points, nframes),
        interior_points=_keep_first_frame(export_data.interior_points, nframes),
    )

    merged_params: dict = {}
    merged_params.update(config)
    if optimal_params:
        merged_params.update(optimal_params)
    if best_params:
        merged_params.update(best_params)
    merged_params = _filter_params_for_interactive(merged_params)

    payload = {
        "controller_mask": export_data.controller_mask,
        "controller_points": export_data.controller_points,
        "object_points": export_data.object_points,
        "object_colors": export_data.object_colors,
        "object_visibilities": export_data.object_visibilities,
        "object_motions_valid": export_data.object_motions_valid,
        "surface_points": export_data.surface_points,
        "interior_points": export_data.interior_points,
        "params": merged_params,
    }

    with open(args.out_pkl, "wb") as f:
        pickle.dump(payload, f)


if __name__ == "__main__":
    main()
