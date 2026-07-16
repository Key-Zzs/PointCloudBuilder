"""Shared disparity-to-metric-depth conversion and validity accounting."""

from __future__ import annotations

from typing import Any

import torch

Tensor = torch.Tensor


def disparity_to_depth(
    disparity_px: Tensor,
    *,
    fx_px: float,
    baseline_m: float,
    epsilon: float = 1e-3,
    remove_invisible: bool = True,
    min_depth_m: float = 0.0,
    max_depth_m: float | None = None,
) -> tuple[Tensor, Tensor, dict[str, int]]:
    """Convert disparity pixels to left-IR metric depth.

    Invalid output depth is always exactly zero.  The reason counters are
    intentionally not mutually exclusive: one pixel can violate more than
    one configured validity rule, and the metadata makes that explicit.
    """

    if disparity_px.ndim != 2:
        raise ValueError(f"Disparity must have shape HxW, got {tuple(disparity_px.shape)}")
    if not torch.isfinite(torch.tensor(fx_px)) or fx_px <= 0.0:
        raise ValueError(f"fx_px must be positive and finite, got {fx_px}")
    if not torch.isfinite(torch.tensor(baseline_m)) or baseline_m <= 0.0:
        raise ValueError(f"baseline_m must be positive and finite, got {baseline_m}")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if min_depth_m < 0.0:
        raise ValueError("min_depth_m must be non-negative")
    if max_depth_m is not None and max_depth_m <= 0.0:
        raise ValueError("max_depth_m must be positive when provided")

    disparity = disparity_px.to(dtype=torch.float32)
    h, w = disparity.shape
    u = torch.arange(w, device=disparity.device, dtype=torch.float32).view(1, w)
    finite = torch.isfinite(disparity)
    non_positive = finite & (disparity <= epsilon)
    positive = finite & (disparity > epsilon)
    invisible = positive & ((u - disparity) < 0.0)
    depth = (float(fx_px) * float(baseline_m)) / disparity
    depth_finite = torch.isfinite(depth)
    below_min = depth_finite & (depth < float(min_depth_m))
    above_max = (
        depth_finite & (depth > float(max_depth_m)) if max_depth_m is not None else torch.zeros_like(depth, dtype=torch.bool)
    )
    valid = finite & (disparity > epsilon) & depth_finite
    if remove_invisible:
        valid = valid & ~invisible
    valid = valid & ~below_min & ~above_max
    depth_m = torch.where(valid, depth, torch.zeros_like(depth)).to(dtype=torch.float32)
    counts: dict[str, int] = {
        "total": int(disparity.numel()),
        "valid": int(valid.sum().item()),
        "invalid": int((~valid).sum().item()),
        "non_finite": int((~finite).sum().item()),
        "non_positive_or_epsilon": int(non_positive.sum().item()),
        "invisible_right_coordinate": int(invisible.sum().item()) if remove_invisible else 0,
        "below_min_depth": int(below_min.sum().item()),
        "above_max_depth": int(above_max.sum().item()),
    }
    return depth_m, valid, counts


def finite_positive_ratio(value: Tensor) -> float:
    """Return a scalar ratio for reports without changing the tensor."""

    if value.numel() == 0:
        return 0.0
    mask = torch.isfinite(value) & (value > 0)
    return float(mask.sum().item() / value.numel())
