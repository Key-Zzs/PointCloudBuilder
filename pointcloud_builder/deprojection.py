"""Depth image deprojection to camera-frame point clouds."""

from __future__ import annotations

import torch

from pointcloud_builder.camera_model import CameraModel
from pointcloud_builder.types import Tensor


def deproject_depth(depth: Tensor, camera: CameraModel) -> tuple[Tensor, Tensor]:
    """Back-project a depth image into camera coordinates.

    Returns valid XYZ points and the flattened validity mask used to select
    aligned color pixels.
    """

    depth_hw = _normalize_depth_shape(depth, camera)
    z = depth_hw.to(dtype=torch.float32) * camera.depth_scale
    grid_x, grid_y = camera.pixel_grid(z.device)
    x = (grid_x - camera.cx) * z / camera.fx
    y = (grid_y - camera.cy) * z / camera.fy
    points = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    valid_mask = torch.isfinite(points).all(dim=-1) & (points[:, 2] > 0.0)
    return points[valid_mask], valid_mask


def _normalize_depth_shape(depth: Tensor, camera: CameraModel) -> Tensor:
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError("Depth image must have shape H x W or H x W x 1")
    if int(depth.shape[0]) != camera.height or int(depth.shape[1]) != camera.width:
        raise ValueError(
            f"Depth shape {tuple(depth.shape)} does not match "
            f"camera height/width {(camera.height, camera.width)}"
        )
    return depth
