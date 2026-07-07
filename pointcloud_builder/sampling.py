"""Fixed-size point-cloud sampling."""

from __future__ import annotations

import torch

from pointcloud_builder.config import SamplingConfig
from pointcloud_builder.types import Meta, Tensor


def sample_points(
    points: Tensor,
    config: SamplingConfig,
    colors: Tensor | None = None,
) -> tuple[Tensor, Tensor | None, Meta]:
    """Sample or pad a point cloud to a fixed number of points."""

    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape N x 3")
    if colors is not None and int(colors.shape[0]) != int(points.shape[0]):
        raise ValueError("colors and points must have the same first dimension")

    num_points = config.num_points
    if points.shape[0] == 0:
        sampled_points = torch.zeros((num_points, 3), dtype=points.dtype, device=points.device)
        sampled_colors = _empty_colors(num_points, colors, points)
        return sampled_points, sampled_colors, {
            "input_count": 0,
            "candidate_count": 0,
            "sampled_count": num_points,
            "empty_input": True,
            "mode": config.mode,
        }

    candidate_idx = _candidate_indices(points, config)
    candidates = points[candidate_idx]
    candidate_colors = colors[candidate_idx] if colors is not None else None
    fixed_idx = _fixed_indices(candidates, config)
    sampled_points = candidates[fixed_idx]
    sampled_colors = candidate_colors[fixed_idx] if candidate_colors is not None else None
    return sampled_points, sampled_colors, {
        "input_count": int(points.shape[0]),
        "candidate_count": int(candidates.shape[0]),
        "sampled_count": int(sampled_points.shape[0]),
        "empty_input": False,
        "mode": config.mode,
    }


def farthest_point_indices(points: Tensor, num_points: int) -> Tensor:
    """Return farthest-point-sampling indices for a point tensor."""

    n = int(points.shape[0])
    if n == 0:
        return torch.empty((0,), dtype=torch.long, device=points.device)
    k = min(n, num_points)
    selected = torch.empty((k,), dtype=torch.long, device=points.device)
    distances = torch.full((n,), float("inf"), dtype=points.dtype, device=points.device)
    selected[0] = 0
    for i in range(1, k):
        last = points[selected[i - 1]].unsqueeze(0)
        distances = torch.minimum(distances, torch.sum((points - last) ** 2, dim=-1))
        selected[i] = torch.argmax(distances)
    return selected


def _candidate_indices(points: Tensor, config: SamplingConfig) -> Tensor:
    mode = config.mode
    n = int(points.shape[0])
    device = points.device
    if mode == "stride":
        return torch.arange(0, n, config.stride, dtype=torch.long, device=device)
    if mode in {"voxel", "voxel_random", "voxel_fps"}:
        return _voxel_representative_indices(points, config.voxel_size)
    return torch.arange(n, dtype=torch.long, device=device)


def _fixed_indices(points: Tensor, config: SamplingConfig) -> Tensor:
    mode = config.mode
    target = config.num_points
    if mode in {"fps", "voxel_fps"}:
        base = farthest_point_indices(points, min(target, int(points.shape[0])))
    elif mode in {"random", "voxel_random"}:
        n = int(points.shape[0])
        base = torch.randperm(n, device=points.device)[: min(target, n)]
    else:
        n = int(points.shape[0])
        base = torch.arange(min(target, n), dtype=torch.long, device=points.device)
    return _pad_indices(base, int(points.shape[0]), target)


def _pad_indices(indices: Tensor, source_count: int, target_count: int) -> Tensor:
    if int(indices.shape[0]) == target_count:
        return indices
    if source_count <= 0:
        return torch.zeros((target_count,), dtype=torch.long, device=indices.device)
    missing = target_count - int(indices.shape[0])
    if missing <= 0:
        return indices[:target_count]
    extra = torch.randint(source_count, (missing,), dtype=torch.long, device=indices.device)
    return torch.cat([indices, extra], dim=0)


def _voxel_representative_indices(points: Tensor, voxel_size: float) -> Tensor:
    if points.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=points.device)
    keys = torch.floor(points / voxel_size).to(dtype=torch.int64)
    _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    representatives: list[Tensor] = []
    for voxel_id in torch.unique(inverse):
        voxel_indices = torch.nonzero(inverse == voxel_id, as_tuple=False).flatten()
        representatives.append(voxel_indices[0])
    if not representatives:
        return torch.empty((0,), dtype=torch.long, device=points.device)
    return torch.stack(representatives).to(dtype=torch.long, device=points.device)


def _empty_colors(num_points: int, colors: Tensor | None, points: Tensor) -> Tensor | None:
    if colors is None:
        return None
    channels = int(colors.shape[-1]) if colors.ndim == 2 else 3
    return torch.zeros((num_points, channels), dtype=colors.dtype, device=points.device)
