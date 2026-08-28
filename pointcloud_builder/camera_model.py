"""Pinhole camera model helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from pointcloud_builder.types import Tensor


@dataclass(frozen=True)
class CameraIntrinsics:
    """Projection model for one image stream.

    Legacy YAML files that contain only the six pinhole values are interpreted
    as ideal/rectified pinhole images.  CameraRig adapters override those
    defaults explicitly for raw factory streams.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str = "none"
    distortion_coeffs: tuple[float, ...] = ()
    pixel_geometry: str = "rectified"
    frame: str = ""

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera image width and height must be positive")
        values = (self.fx, self.fy, self.cx, self.cy, *self.distortion_coeffs)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("camera projection parameters must be finite")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        if self.pixel_geometry not in {"raw", "rectified"}:
            raise ValueError("pixel_geometry must be 'raw' or 'rectified'")
        if self.distortion_model not in {
            "none",
            "brown-conrady",
            "modified-brown-conrady",
            "inverse-brown-conrady",
            "ftheta",
            "kannala-brandt4",
        }:
            raise ValueError(f"unsupported distortion model: {self.distortion_model!r}")
        coefficients = tuple(float(value) for value in self.distortion_coeffs)
        object.__setattr__(self, "distortion_coeffs", coefficients)
        if self.distortion_model == "none" and any(
            abs(value) > 1e-12 for value in coefficients
        ):
            raise ValueError("distortion_model='none' cannot have non-zero coefficients")
        if self.pixel_geometry == "rectified" and (
            self.distortion_model != "none"
            or any(abs(value) > 1e-12 for value in coefficients)
        ):
            raise ValueError(
                "rectified pixel geometry requires distortion_model='none' and zero coefficients"
            )

    @property
    def is_identity_projection(self) -> bool:
        """Whether distortion is numerically an identity mapping."""

        return self.distortion_model == "none" or not any(
            abs(value) > 1e-12 for value in self.distortion_coeffs
        )


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
    def from_config(cls, config: Any) -> CameraModel:
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
