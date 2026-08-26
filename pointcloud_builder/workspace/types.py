"""Frame-explicit point-cloud data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch


@dataclass(frozen=True)
class FramedPointCloud:
    """An XYZ or XYZRGB tensor with an explicit coordinate frame."""

    points: torch.Tensor
    frame: str
    metadata: dict[str, Any] = field(default_factory=dict)
    finite_policy: Literal["reject"] = "reject"

    def __post_init__(self) -> None:
        if not isinstance(self.points, torch.Tensor):
            raise TypeError("points must be a torch.Tensor")
        if self.points.ndim != 2 or self.points.shape[1] not in {3, 6}:
            raise ValueError("points must have shape Nx3 or Nx6")
        if not self.frame.strip():
            raise ValueError("frame must be a non-empty string")
        if self.finite_policy != "reject":
            raise ValueError("only finite_policy='reject' is supported")
        if not bool(torch.isfinite(self.points).all()):
            raise ValueError("points must contain only finite values")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class WorkspacePointCloud(FramedPointCloud):
    """A FramedPointCloud after an explicit transform into a workspace frame."""
