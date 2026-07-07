"""Depth image deprojection to camera-frame point clouds."""

from __future__ import annotations

import torch

from pointcloud_builder.camera_model import CameraIntrinsics
from pointcloud_builder.types import Tensor


def deproject_depth(
    depth: Tensor,
    intrinsics: CameraIntrinsics,
    depth_scale: float,
    *,
    flatten: bool = True,
) -> tuple[Tensor, Tensor]:
    """Back-project a depth image into camera coordinates.

    Depth values are multiplied by ``depth_scale`` before projection. The
    returned mask marks finite positive-depth pixels and is flattened when
    ``flatten`` is true.
    """

    depth_hw = _normalize_depth_shape(depth, intrinsics)
    z = depth_hw.to(dtype=torch.float32) * depth_scale
    grid_x, grid_y = _pixel_grid(intrinsics, z.device)
    x = (grid_x - intrinsics.cx) * z / intrinsics.fx
    y = (grid_y - intrinsics.cy) * z / intrinsics.fy
    points_hw = torch.stack([x, y, z], dim=-1)
    valid_hw = torch.isfinite(points_hw).all(dim=-1) & (points_hw[..., 2] > 0.0)
    if not flatten:
        return points_hw, valid_hw
    points = points_hw.reshape(-1, 3)
    valid_mask = valid_hw.reshape(-1)
    return points[valid_mask], valid_mask


def _pixel_grid(intrinsics: CameraIntrinsics, device: torch.device) -> tuple[Tensor, Tensor]:
    ys = torch.arange(intrinsics.height, dtype=torch.float32, device=device)
    xs = torch.arange(intrinsics.width, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return grid_x, grid_y


def _normalize_depth_shape(depth: Tensor, intrinsics: CameraIntrinsics) -> Tensor:
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError("Depth image must have shape H x W or H x W x 1")
    if int(depth.shape[0]) != intrinsics.height or int(depth.shape[1]) != intrinsics.width:
        raise ValueError(
            f"Depth shape {tuple(depth.shape)} does not match "
            f"camera height/width {(intrinsics.height, intrinsics.width)}"
        )
    return depth
