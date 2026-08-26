"""Synthetic before/after fusion geometry metrics."""

from __future__ import annotations

import torch


def synthetic_geometry_metrics(
    before: torch.Tensor, after: torch.Tensor, *, voxel_size_m: float
) -> dict[str, float | int]:
    before_xyz = before[:, :3]
    after_xyz = after[:, :3]
    before_plane = before_xyz[before_xyz[:, 2].abs() < 0.01]
    after_plane = after_xyz[after_xyz[:, 2].abs() < 0.01]
    before_box_top = _box_top(before_xyz)
    after_box_top = _box_top(after_xyz)
    before_thickness = _quantile(before_plane[:, 2], 0.95) - _quantile(before_plane[:, 2], 0.05)
    after_thickness = _quantile(after_plane[:, 2], 0.95) - _quantile(after_plane[:, 2], 0.05)
    nearest = _nearest_distances(_bounded(after_xyz), _bounded(before_xyz))
    completeness_distances = _nearest_distances(_bounded(before_xyz), _bounded(after_xyz))
    return {
        "before_point_count": int(before.shape[0]),
        "after_point_count": int(after.shape[0]),
        "duplicate_surface_thickness_before_m": before_thickness,
        "duplicate_surface_thickness_after_m": after_thickness,
        "nearest_neighbor_residual_median_m": float(nearest.median().item()),
        "nearest_neighbor_residual_p95_m": _quantile(nearest, 0.95),
        "point_to_plane_before_median_m": float(before_plane[:, 2].abs().median().item()),
        "point_to_plane_after_median_m": float(after_plane[:, 2].abs().median().item()),
        "plane_signed_bias_shift_m": abs(
            float(after_plane[:, 2].median().item())
            - float(before_plane[:, 2].median().item())
        ),
        "box_top_before_median_z_m": float(before_box_top[:, 2].median().item()),
        "box_top_after_median_z_m": float(after_box_top[:, 2].median().item()),
        "box_top_systematic_shift_m": abs(
            float(after_box_top[:, 2].median().item())
            - float(before_box_top[:, 2].median().item())
        ),
        "voxel_occupancy_reduction": 1.0 - float(after.shape[0]) / max(1, int(before.shape[0])),
        "completeness_at_1_5_voxels": float(
            (completeness_distances <= 1.5 * voxel_size_m).float().mean().item()
        ),
    }


def _box_top(points: torch.Tensor) -> torch.Tensor:
    selected = points[
        (points[:, 0].abs() < 0.16)
        & (points[:, 1].abs() < 0.10)
        & (points[:, 2] > 0.20)
    ]
    if selected.shape[0] == 0:
        raise ValueError("synthetic metrics require visible box-top points")
    return selected


def _bounded(points: torch.Tensor, limit: int = 2000) -> torch.Tensor:
    if points.shape[0] <= limit:
        return points
    indices = torch.linspace(0, points.shape[0] - 1, limit, device=points.device).long()
    return points[indices]


def _nearest_distances(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if query.shape[0] == 0 or reference.shape[0] == 0:
        return torch.full((max(1, query.shape[0]),), float("inf"), device=query.device)
    return torch.cdist(query, reference).min(dim=1).values


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return float("inf")
    return float(torch.quantile(values, q).item())
