"""Pure numerical helpers for diagnostic cross-camera alignment.

The transforms in this module always use column-vector homogeneous semantics:
``p_target = T_target_from_source @ p_source``.  Point arrays are stored as rows,
so the equivalent implementation is ``p @ R.T + t``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply ``T_target_from_source`` to an ``Nx3`` row-point array."""

    xyz = _points(points, "points")
    matrix = _transform(transform)
    return xyz @ matrix[:3, :3].T + matrix[:3, 3]


def fit_rigid_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the proper least-squares ``T_target_from_source`` using SVD."""

    source_xyz = _points(source, "source")
    target_xyz = _points(target, "target")
    if source_xyz.shape != target_xyz.shape or source_xyz.shape[0] < 3:
        raise ValueError(
            "source and target must be corresponding Nx3 arrays with N >= 3"
        )
    source_center = source_xyz.mean(axis=0)
    target_center = target_xyz.mean(axis=0)
    covariance = (source_xyz - source_center).T @ (target_xyz - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def robust_point_to_point_icp(
    anchor: np.ndarray,
    moving: np.ndarray,
    *,
    initial: np.ndarray | None = None,
    maximum_correspondence_distance_m: float = 0.030,
    trim_fraction: float = 0.80,
    max_iterations: int = 40,
    tolerance_m: float = 1e-7,
) -> dict[str, Any]:
    """Diagnostic trimmed point-to-point ICP with an identity-default initial pose.

    This deterministic implementation is primarily the comparison estimator and
    synthetic-test oracle.  The live diagnostic CLI uses robust point-to-plane
    Open3D ICP as its primary estimator and records both results.
    """

    if not 0.1 <= trim_fraction <= 1.0:
        raise ValueError("trim_fraction must be in [0.1, 1]")
    if maximum_correspondence_distance_m <= 0 or max_iterations <= 0:
        raise ValueError("ICP distance and iteration limits must be positive")
    fixed = _points(anchor, "anchor")
    source = _points(moving, "moving")
    transform = np.eye(4) if initial is None else _transform(initial).copy()
    from scipy.spatial import cKDTree

    tree = cKDTree(fixed)
    history: list[float] = []
    correspondence_count = 0
    for _ in range(max_iterations):
        corrected = apply_transform(source, transform)
        distances, indices = tree.query(corrected, workers=1)
        valid = np.isfinite(distances) & (
            distances <= maximum_correspondence_distance_m
        )
        if int(valid.sum()) < 6:
            raise ValueError("ICP has fewer than six inlier correspondences")
        valid_indices = np.flatnonzero(valid)
        keep = max(6, math.floor(valid_indices.size * trim_fraction))
        chosen = valid_indices[
            np.argsort(distances[valid_indices], kind="stable")[:keep]
        ]
        incremental = fit_rigid_transform(corrected[chosen], fixed[indices[chosen]])
        transform = incremental @ transform
        median = float(np.median(distances[chosen]))
        history.append(median)
        correspondence_count = int(chosen.size)
        step = np.linalg.norm(incremental[:3, 3])
        trace = float(np.clip((np.trace(incremental[:3, :3]) - 1.0) / 2.0, -1, 1))
        angle = math.acos(trace)
        if step <= tolerance_m and angle <= 1e-7:
            break
    return {
        "T_anchor_from_moving_residual": transform,
        "iterations": len(history),
        "correspondence_count": correspondence_count,
        "median_history_m": history,
    }


def distance_summary_mm(values_m: np.ndarray) -> dict[str, float | int]:
    """Return the contract's complete residual summary in millimetres."""

    values = np.asarray(values_m, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("residual summary requires at least one finite value")
    return {
        "count": int(values.size),
        "median_mm": float(np.median(values) * 1000.0),
        "p75_mm": float(np.quantile(values, 0.75) * 1000.0),
        "p90_mm": float(np.quantile(values, 0.90) * 1000.0),
        "p95_mm": float(np.quantile(values, 0.95) * 1000.0),
        "p99_mm": float(np.quantile(values, 0.99) * 1000.0),
        "rmse_mm": float(np.sqrt(np.mean(values * values)) * 1000.0),
    }


def frozen_overlap_mask(
    before_distance_m: np.ndarray, *, maximum_distance_m: float = 0.030
) -> np.ndarray:
    """Freeze the pre-correction cross-camera overlap cohort."""

    distance = np.asarray(before_distance_m, dtype=np.float64).reshape(-1)
    if not math.isfinite(maximum_distance_m) or maximum_distance_m <= 0:
        raise ValueError("maximum overlap distance must be finite and positive")
    return np.isfinite(distance) & (distance <= maximum_distance_m)


def normalized_image_coordinates(
    uv: np.ndarray,
    *,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> dict[str, np.ndarray]:
    """Compute normalized horizontal, vertical, and radial image coordinates."""

    pixels = np.asarray(uv, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("uv must be Nx2")
    if min(width, height) <= 0 or min(fx, fy) <= 0:
        raise ValueError("image sizes and focal lengths must be positive")
    u = pixels[:, 0]
    v = pixels[:, 1]
    return {
        "x_n": (u - cx) / width,
        "y_n": (v - cy) / height,
        "u_over_w": u / width,
        "v_over_h": v / height,
        "radius": np.sqrt(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2),
    }


def binned_residuals(
    coordinate: np.ndarray,
    residual_m: np.ndarray,
    *,
    signed_point_to_plane_m: np.ndarray | None = None,
    bins: int = 5,
    edges: np.ndarray | None = None,
) -> list[dict[str, float | int | None]]:
    """Return deterministic equal-width bins spanning the observed coordinate."""

    x = np.asarray(coordinate, dtype=np.float64).reshape(-1)
    residual = np.asarray(residual_m, dtype=np.float64).reshape(-1)
    signed = (
        None
        if signed_point_to_plane_m is None
        else np.asarray(signed_point_to_plane_m, dtype=np.float64).reshape(-1)
    )
    if x.shape != residual.shape or (signed is not None and signed.shape != x.shape):
        raise ValueError("coordinate and residual arrays must have equal shape")
    if bins < 2:
        raise ValueError("at least two bins are required")
    finite = np.isfinite(x) & np.isfinite(residual)
    if signed is not None:
        finite &= np.isfinite(signed)
    x = x[finite]
    residual = residual[finite]
    signed = None if signed is None else signed[finite]
    if x.size < bins:
        raise ValueError("too few values for requested bins")
    if edges is None:
        low, high = float(x.min()), float(x.max())
        if math.isclose(low, high):
            bin_edges = np.linspace(low - 0.5, high + 0.5, bins + 1)
        else:
            bin_edges = np.linspace(low, high, bins + 1)
    else:
        bin_edges = np.asarray(edges, dtype=np.float64)
        if bin_edges.shape != (bins + 1,) or not np.all(np.diff(bin_edges) > 0):
            raise ValueError("edges must contain bins + 1 strictly increasing values")
    assignment = np.digitize(x, bin_edges[1:-1])
    output = []
    for index in range(bins):
        mask = assignment == index
        values = residual[mask]
        output.append(
            {
                "bin": index,
                "low": float(bin_edges[index]),
                "high": float(bin_edges[index + 1]),
                "count": int(mask.sum()),
                "median_mm": None
                if not values.size
                else float(np.median(values) * 1000),
                "p95_mm": None
                if not values.size
                else float(np.quantile(values, 0.95) * 1000),
                "signed_point_to_plane_median_mm": (
                    None
                    if signed is None or not values.size
                    else float(np.median(signed[mask]) * 1000)
                ),
            }
        )
    return output


def correlation_summary(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Return Pearson, Spearman, and robust Theil-Sen slope statistics."""

    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("correlation inputs must have equal shape")
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if a.size < 3:
        raise ValueError("correlation requires three samples")
    if np.ptp(a) <= 0 or np.ptp(b) <= 0:
        return {
            "count": int(a.size),
            "defined": False,
            "reason": "constant_x" if np.ptp(a) <= 0 else "constant_y",
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
            "theil_sen_slope": None,
            "theil_sen_intercept": None,
            "theil_sen_slope_ci_low": None,
            "theil_sen_slope_ci_high": None,
        }
    from scipy import stats

    pearson = stats.pearsonr(a, b)
    spearman = stats.spearmanr(a, b)
    slope, intercept, low_slope, high_slope = stats.theilslopes(b, a, 0.95)
    return {
        "count": int(a.size),
        "defined": True,
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "theil_sen_slope": float(slope),
        "theil_sen_intercept": float(intercept),
        "theil_sen_slope_ci_low": float(low_slope),
        "theil_sen_slope_ci_high": float(high_slope),
    }


def point_to_plane_observability(
    transformed_source: np.ndarray,
    anchor_normals: np.ndarray,
    *,
    characteristic_length_m: float | None = None,
    relative_rank_threshold: float = 1e-6,
) -> dict[str, Any]:
    """Diagnose the rank/conditioning of a point-to-plane SE(3) fit."""

    points = _points(transformed_source, "transformed_source")
    normals = _points(anchor_normals, "anchor_normals")
    if points.shape != normals.shape or points.shape[0] < 6:
        raise ValueError("observability requires corresponding point/normal arrays")
    lengths = np.linalg.norm(points - np.median(points, axis=0), axis=1)
    scale = (
        float(np.quantile(lengths, 0.90))
        if characteristic_length_m is None
        else float(characteristic_length_m)
    )
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("characteristic length must be finite and positive")
    jacobian = np.concatenate((normals, np.cross(points, normals) / scale), axis=1)
    information = jacobian.T @ jacobian
    eigenvalues = np.linalg.eigvalsh(information)[::-1]
    relative = eigenvalues / max(float(eigenvalues[0]), np.finfo(float).tiny)
    rank = int(np.sum(relative >= relative_rank_threshold))
    condition = float(eigenvalues[0] / eigenvalues[-1]) if eigenvalues[-1] > 0 else None
    return {
        "information_matrix": information.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "relative_eigenvalues": relative.tolist(),
        "effective_rank": rank,
        "rank_threshold": relative_rank_threshold,
        "condition_number": condition,
        "characteristic_length_m": scale,
        "passed": rank == 6,
    }


def _points(value: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"{name} must be a finite Nx3 array")
    return points


def _transform(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("transform must be finite 4x4")
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-9, rtol=0):
        raise ValueError("transform must be homogeneous")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError("transform rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0):
        raise ValueError("transform rotation must be proper")
    return matrix
