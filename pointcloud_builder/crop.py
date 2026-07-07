"""Point-cloud crop operations."""

from __future__ import annotations

import torch

from pointcloud_builder.config import CropConfig
from pointcloud_builder.types import Tensor


def crop_point_cloud(point_cloud: Tensor, config: CropConfig) -> tuple[Tensor, Tensor]:
    """Crop an ``N x 3`` or ``N x 6`` point cloud by XYZ workspace bounds.

    RGB columns, when present, are preserved because the mask is applied to the
    full point-cloud tensor and computed only from the first three XYZ columns.
    """

    if point_cloud.ndim != 2 or point_cloud.shape[-1] not in {3, 6}:
        raise ValueError("point_cloud must have shape N x 3 or N x 6")
    if not config.enabled:
        mask = torch.ones(point_cloud.shape[0], dtype=torch.bool, device=point_cloud.device)
        return point_cloud, mask

    xyz = point_cloud[:, :3]
    mask = (
        (xyz[:, 0] >= config.x[0])
        & (xyz[:, 0] <= config.x[1])
        & (xyz[:, 1] >= config.y[0])
        & (xyz[:, 1] <= config.y[1])
        & (xyz[:, 2] >= config.z[0])
        & (xyz[:, 2] <= config.z[1])
    )
    return point_cloud[mask], mask


def crop_points(
    points: Tensor,
    config: CropConfig,
    colors: Tensor | None = None,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Backward-compatible crop helper for XYZ plus optional RGB tensors."""

    if colors is None:
        cropped, mask = crop_point_cloud(points, config)
        return cropped, None, mask
    point_cloud = _point_cloud_from_parts(points, colors)
    cropped, mask = crop_point_cloud(point_cloud, config)
    return cropped[:, :3], cropped[:, 3:], mask


def _point_cloud_from_parts(points: Tensor, colors: Tensor) -> Tensor:
    """Pack XYZ and RGB tensors before applying the unified crop path."""

    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape N x 3")
    if colors.ndim != 2 or colors.shape[-1] != 3:
        raise ValueError("colors must have shape N x 3")
    if int(colors.shape[0]) != int(points.shape[0]):
        raise ValueError("colors and points must have the same first dimension")
    return torch.cat([points, colors], dim=-1)
