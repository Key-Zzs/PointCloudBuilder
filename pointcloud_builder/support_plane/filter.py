"""Support-plane distance operations."""

from __future__ import annotations

import torch

from pointcloud_builder.types import Tensor

from .types import SupportPlane


def signed_distance(points: Tensor, plane: SupportPlane) -> Tensor:
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape N x 3 (or XYZ-prefixed channels)")
    normal = torch.as_tensor(plane.normal_array(), dtype=points.dtype, device=points.device)
    return points[:, :3] @ normal + float(plane.offset)


def filter_support_plane(
    points: Tensor,
    plane: SupportPlane,
    *,
    distance_threshold_m: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Remove a *symmetric* distance band around the support plane."""

    threshold = float(distance_threshold_m or plane.distance_threshold_m)
    if threshold <= 0.0:
        raise ValueError("support-plane distance threshold must be positive")
    keep = torch.abs(signed_distance(points, plane)) > threshold
    return points[keep], keep
