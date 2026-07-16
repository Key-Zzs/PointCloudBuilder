"""Small, dependency-light types shared by FFS backends and the builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import torch

from pointcloud_builder.camera_model import CameraExtrinsics, CameraIntrinsics
from pointcloud_builder.types import StereoIRFrame

Tensor = torch.Tensor


class FFSDisparityBackend(Protocol):
    """Common inference contract for all four FFS routes."""

    name: str
    provenance: Mapping[str, Any]
    last_timing_ms: Mapping[str, float]

    def infer_disparity(self, left_ir: Tensor, right_ir: Tensor) -> Tensor:
        """Return ``[H,W]`` float32 disparity in left-image pixels."""


@dataclass(frozen=True)
class FFSDepthResult:
    """Shared FFS output consumed by the single PointCloudBuilder pipeline."""

    disparity_px: Tensor
    depth_m: Tensor
    valid_mask: Tensor
    intrinsics: CameraIntrinsics
    depth_to_color_extrinsics: CameraExtrinsics | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ResolvedDepth:
    """Depth plus the camera contract that must be used for deprojection."""

    depth: Tensor
    effective_depth_scale: float
    intrinsics: CameraIntrinsics
    depth_to_color_extrinsics: CameraExtrinsics | None
    frame_name: str
    metadata: dict[str, Any] | None = None
    disparity_px: Tensor | None = None
    valid_mask: Tensor | None = None


def frame_field(frame: Any, key: str) -> Any:
    """Read a field from a mapping or one of the supported frame dataclasses."""

    if isinstance(frame, Mapping):
        if key not in frame:
            raise KeyError(f"Frame is missing required field: {key}")
        return frame[key]
    try:
        return getattr(frame, key)
    except AttributeError as exc:
        raise KeyError(f"Frame is missing required field: {key}") from exc


def optional_frame_field(frame: Any, key: str) -> Any | None:
    """Read an optional field without changing the old mapping semantics."""

    if isinstance(frame, Mapping):
        return frame.get(key)
    return getattr(frame, key, None)
