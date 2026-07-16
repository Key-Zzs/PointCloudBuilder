"""Shared type definitions for PointCloudBuilder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TypeAlias

import torch

Tensor: TypeAlias = torch.Tensor
FrameMapping: TypeAlias = Mapping[str, Any]
Meta: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class RGBDFrame:
    """Container for one RGB-D frame."""

    depth: Any
    rgb: Any | None = None
    timestamp: float | None = None
    global_frame_index: int | None = None


@dataclass(frozen=True)
class StereoIRFrame:
    """One synchronized left/right IR frame for optional FFS depth."""

    left_ir: Any
    right_ir: Any
    rgb: Any | None = None
    timestamp: float | None = None
    global_frame_index: int | None = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass(frozen=True)
class PointCloudStages:
    """Intermediate tensors for offline inspection and visualization."""

    raw: Tensor
    cropped: Tensor
    sampled: Tensor
