"""Utility helpers shared by runtime modules."""

from __future__ import annotations

from typing import Any

import torch

from pointcloud_builder.types import RGBDFrame, Tensor


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve a requested device, using CUDA when available for auto mode."""

    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def as_tensor(data: Any, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Convert array-like input to a tensor on the requested device."""

    return torch.as_tensor(data, dtype=dtype, device=device)


def get_frame_value(frame: RGBDFrame | dict[str, Any], key: str) -> Any:
    """Extract a frame field from a dataclass or mapping."""

    if isinstance(frame, RGBDFrame):
        return getattr(frame, key)
    if key in frame:
        return frame[key]
    raise KeyError(f"Frame is missing required field: {key}")


def get_optional_frame_value(frame: RGBDFrame | dict[str, Any], key: str) -> Any | None:
    """Extract an optional frame field from a dataclass or mapping."""

    if isinstance(frame, RGBDFrame):
        return getattr(frame, key)
    return frame.get(key)


def normalize_color(color: Tensor) -> Tensor:
    """Convert image colors to float RGB in the [0, 1] range."""

    color_float = color.to(dtype=torch.float32)
    if color_float.numel() > 0 and torch.nan_to_num(color_float).max() > 1.0:
        color_float = color_float / 255.0
    if color_float.ndim != 3 or color_float.shape[-1] < 3:
        raise ValueError("Color image must have shape H x W x 3 or H x W x C")
    return color_float[..., :3].clamp(0.0, 1.0)


def pack_point_cloud(points: Tensor, colors: Tensor | None) -> Tensor:
    """Pack XYZ and optional RGB tensors into one point-cloud tensor."""

    if colors is None:
        return points
    return torch.cat([points, colors], dim=-1)
