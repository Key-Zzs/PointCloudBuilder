"""Point-cloud crop operations."""

from __future__ import annotations

import torch

from pointcloud_builder.config import CropConfig
from pointcloud_builder.types import Tensor


def crop_points(
    points: Tensor,
    config: CropConfig,
    colors: Tensor | None = None,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Crop points and optional colors with axis-aligned camera-frame bounds."""

    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape N x 3")
    if colors is not None and int(colors.shape[0]) != int(points.shape[0]):
        raise ValueError("colors and points must have the same first dimension")
    if not config.enabled:
        mask = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
        return points, colors, mask

    mask = (
        (points[:, 0] >= config.x[0])
        & (points[:, 0] <= config.x[1])
        & (points[:, 1] >= config.y[0])
        & (points[:, 1] <= config.y[1])
        & (points[:, 2] >= config.z[0])
        & (points[:, 2] <= config.z[1])
    )
    cropped_colors = colors[mask] if colors is not None else None
    return points[mask], cropped_colors, mask
