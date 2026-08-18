"""Robust camera-frame support-plane estimators for offline episode setup."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from .types import SupportPlane


def _as_xyz(points: np.ndarray | torch.Tensor) -> np.ndarray:
    value = points.detach().cpu().numpy() if isinstance(points, torch.Tensor) else np.asarray(points)
    if value.ndim != 2 or value.shape[1] < 3:
        raise ValueError("points must have shape N x 3 (or XYZ-prefixed channels)")
    xyz = np.asarray(value[:, :3], dtype=np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)]


def estimate_support_plane(
    points: np.ndarray | torch.Tensor,
    *,
    distance_threshold_m: float,
    ransac_iterations: int = 256,
    seed: int = 0,
    source_frame_indices: Iterable[int] = (),
    config_hash: str | None = None,
) -> SupportPlane:
    """Fit a dominant support plane with RANSAC then SVD inlier refinement."""

    xyz = _as_xyz(points)
    if len(xyz) < 3:
        raise ValueError("at least three finite points are required for a support plane")
    rng = np.random.default_rng(seed)
    best: tuple[int, np.ndarray, float, np.ndarray] | None = None
    for _ in range(ransac_iterations):
        sample = xyz[rng.choice(len(xyz), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            continue
        normal = normal / norm
        offset = -float(normal @ sample[0])
        inliers = np.abs(xyz @ normal + offset) <= distance_threshold_m
        candidate = (int(inliers.sum()), normal, offset, inliers)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None or best[0] < 3:
        raise ValueError("support-plane RANSAC found no non-degenerate consensus")
    inliers = xyz[best[3]]
    centroid = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    # A deterministic sign makes caches and normal-drift reports comparable.
    dominant = int(np.argmax(np.abs(normal)))
    if normal[dominant] < 0.0:
        normal = -normal
    offset = -float(normal @ centroid)
    residual = np.abs(xyz @ normal + offset)
    final_inliers = residual <= distance_threshold_m
    return SupportPlane(
        normal=tuple(float(item) for item in normal),  # type: ignore[arg-type]
        offset=offset,
        distance_threshold_m=float(distance_threshold_m),
        source_frame_indices=tuple(int(item) for item in source_frame_indices),
        inlier_ratio=float(np.mean(final_inliers)),
        residual_median=float(np.median(residual[final_inliers])) if np.any(final_inliers) else float("inf"),
        residual_p95=float(np.percentile(residual[final_inliers], 95)) if np.any(final_inliers) else float("inf"),
        config_hash=config_hash,
    )


def estimate_episode_support_plane(
    frame_points: Iterable[tuple[int, np.ndarray | torch.Tensor]],
    *,
    distance_threshold_m: float,
    ransac_iterations: int = 256,
    config_hash: str | None = None,
) -> SupportPlane:
    """Consensus of independently fitted representative frames for one episode.

    No per-frame model is returned: the resulting plane is explicitly the one
    to reuse for every frame in the episode.
    """

    fitted: list[SupportPlane] = []
    for order, (frame_index, points) in enumerate(frame_points):
        try:
            fitted.append(
                estimate_support_plane(
                    points,
                    distance_threshold_m=distance_threshold_m,
                    ransac_iterations=ransac_iterations,
                    seed=order,
                    source_frame_indices=(frame_index,),
                    config_hash=config_hash,
                )
            )
        except ValueError:
            continue
    if not fitted:
        raise ValueError("no representative frame yielded a support-plane estimate")
    normals = np.asarray([plane.normal_array() for plane in fitted])
    reference = normals[0]
    normals[np.sum(normals * reference[None, :], axis=1) < 0.0] *= -1.0
    normal = np.median(normals, axis=0)
    normal /= np.linalg.norm(normal)
    offsets = np.asarray([plane.offset for plane in fitted], dtype=np.float64)
    return SupportPlane(
        normal=tuple(float(item) for item in normal),  # type: ignore[arg-type]
        offset=float(np.median(offsets)),
        distance_threshold_m=float(distance_threshold_m),
        source_frame_indices=tuple(plane.source_frame_indices[0] for plane in fitted),
        inlier_ratio=float(np.median([plane.inlier_ratio for plane in fitted])),
        residual_median=float(np.median([plane.residual_median for plane in fitted])),
        residual_p95=float(np.median([plane.residual_p95 for plane in fitted])),
        config_hash=config_hash,
    )
