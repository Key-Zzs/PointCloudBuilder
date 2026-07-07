"""Pinhole camera model helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pointcloud_builder.config import CameraConfig
from pointcloud_builder.types import Tensor


@dataclass(frozen=True)
class CameraModel:
    """Camera intrinsics used for depth deprojection."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float
    aligned_depth_to_color: bool

    @classmethod
    def from_config(cls, config: CameraConfig) -> "CameraModel":
        """Create a camera model from typed config."""

        intrinsics = config.intrinsics
        return cls(
            width=config.width,
            height=config.height,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.cx,
            cy=intrinsics.cy,
            depth_scale=config.depth_scale,
            aligned_depth_to_color=config.aligned_depth_to_color,
        )

    def pixel_grid(self, device: torch.device) -> tuple[Tensor, Tensor]:
        """Return image-space x and y coordinate grids."""

        ys = torch.arange(self.height, dtype=torch.float32, device=device)
        xs = torch.arange(self.width, dtype=torch.float32, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return grid_x, grid_y
