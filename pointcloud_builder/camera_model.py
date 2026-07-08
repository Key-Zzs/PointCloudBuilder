"""Pinhole camera model helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pointcloud_builder.types import Tensor


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics for one image stream."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class CameraExtrinsics:
    """Rigid transform from one camera stream frame to another."""

    rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    translation: tuple[float, float, float]


@dataclass(frozen=True)
class CameraModel:
    """Camera model containing depth and color stream intrinsics."""

    name: str
    depth_scale: float
    aligned_depth_to_color: bool
    color_intrinsics: CameraIntrinsics
    depth_intrinsics: CameraIntrinsics
    depth_to_color_extrinsics: CameraExtrinsics | None = None

    @classmethod
    def from_config(cls, config: Any) -> "CameraModel":
        """Create a camera model from typed config."""

        return cls(
            name=config.name,
            depth_scale=config.depth_scale,
            aligned_depth_to_color=config.aligned_depth_to_color,
            color_intrinsics=config.color_intrinsics,
            depth_intrinsics=config.depth_intrinsics,
            depth_to_color_extrinsics=config.depth_to_color_extrinsics,
        )

    @property
    def active_intrinsics(self) -> CameraIntrinsics:
        """Return intrinsics matching the configured depth alignment mode."""

        if self.aligned_depth_to_color:
            return self.color_intrinsics
        return self.depth_intrinsics

    @property
    def width(self) -> int:
        """Return active image width."""

        return self.active_intrinsics.width

    @property
    def height(self) -> int:
        """Return active image height."""

        return self.active_intrinsics.height

    def pixel_grid(self, device: torch.device) -> tuple[Tensor, Tensor]:
        """Return image-space x and y coordinate grids."""

        intrinsics = self.active_intrinsics
        ys = torch.arange(intrinsics.height, dtype=torch.float32, device=device)
        xs = torch.arange(intrinsics.width, dtype=torch.float32, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return grid_x, grid_y
