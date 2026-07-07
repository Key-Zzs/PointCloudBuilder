"""Fixed-size point-cloud sampling.

All samplers accept ``N x 3`` XYZ or ``N x 6`` XYZRGB tensors and always return
``num_points x C`` when sampling is enabled. Voxel-based modes compute voxel
indices from XYZ only and keep the first input point in each occupied voxel as
the voxel representative, preserving RGB columns when present.
"""

from __future__ import annotations

import torch

from pointcloud_builder.config import SamplingConfig
from pointcloud_builder.types import Meta, Tensor


def sample_point_cloud(point_cloud: Tensor, config: SamplingConfig) -> tuple[Tensor, Meta]:
    """Sample a point cloud to a fixed number of points.

    The implementation is pure PyTorch and can run on CPU or CUDA tensors. FPS
    uses a straightforward PyTorch farthest-point loop and can be replaced later
    by a custom CUDA kernel without changing the public interface.
    """

    _validate_point_cloud(point_cloud)
    if not config.enabled:
        return point_cloud, _sampling_meta(
            input_count=int(point_cloud.shape[0]),
            candidate_count=int(point_cloud.shape[0]),
            sampled_count=int(point_cloud.shape[0]),
            config=config,
            input_empty=int(point_cloud.shape[0]) == 0,
            padded=False,
        )

    target = config.num_points
    channels = int(point_cloud.shape[1])
    input_count = int(point_cloud.shape[0])
    if input_count == 0:
        sampled = torch.zeros((target, channels), dtype=point_cloud.dtype, device=point_cloud.device)
        return sampled, _sampling_meta(
            input_count=0,
            candidate_count=0,
            sampled_count=target,
            config=config,
            input_empty=True,
            padded=True,
        )

    candidates = _candidate_points(point_cloud, config)
    selected = _select_fixed_count(candidates, config)
    padded = int(candidates.shape[0]) < target
    return selected, _sampling_meta(
        input_count=input_count,
        candidate_count=int(candidates.shape[0]),
        sampled_count=int(selected.shape[0]),
        config=config,
        input_empty=False,
        padded=padded,
    )


def sample_points(
    points: Tensor,
    config: SamplingConfig,
    colors: Tensor | None = None,
) -> tuple[Tensor, Tensor | None, Meta]:
    """Backward-compatible wrapper for separate XYZ and optional RGB tensors."""

    if colors is None:
        sampled, meta = sample_point_cloud(points, config)
        return sampled, None, meta
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape N x 3")
    if colors.ndim != 2 or colors.shape[-1] != 3:
        raise ValueError("colors must have shape N x 3")
    if int(points.shape[0]) != int(colors.shape[0]):
        raise ValueError("colors and points must have the same first dimension")
    sampled, meta = sample_point_cloud(torch.cat([points, colors], dim=-1), config)
    return sampled[:, :3], sampled[:, 3:], meta


def farthest_point_indices(points: Tensor, num_points: int) -> Tensor:
    """Return farthest-point-sampling indices computed from XYZ points."""

    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape N x 3")
    n = int(points.shape[0])
    if n == 0 or num_points <= 0:
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


def _candidate_points(point_cloud: Tensor, config: SamplingConfig) -> Tensor:
    if config.mode in {"voxel", "voxel_random", "voxel_fps"}:
        indices = _voxel_representative_indices(point_cloud[:, :3], config.voxel_size)
        return point_cloud[indices]
    return point_cloud


def _select_fixed_count(point_cloud: Tensor, config: SamplingConfig) -> Tensor:
    n = int(point_cloud.shape[0])
    target = config.num_points
    if n == 0:
        return torch.zeros((target, point_cloud.shape[1]), dtype=point_cloud.dtype, device=point_cloud.device)

    if config.mode in {"fps", "voxel_fps"}:
        base_indices = farthest_point_indices(point_cloud[:, :3], min(target, n))
    elif config.mode in {"random", "voxel_random"}:
        generator = _make_generator(config, point_cloud.device)
        base_indices = torch.randperm(n, device=point_cloud.device, generator=generator)[: min(target, n)]
    elif config.mode == "stride":
        base_indices = torch.arange(0, n, config.stride, dtype=torch.long, device=point_cloud.device)
        base_indices = base_indices[: min(target, int(base_indices.shape[0]))]
    else:
        base_indices = torch.arange(min(target, n), dtype=torch.long, device=point_cloud.device)

    sampled = point_cloud[base_indices]
    return _pad_or_trim(sampled, point_cloud, config)


def _pad_or_trim(sampled: Tensor, source: Tensor, config: SamplingConfig) -> Tensor:
    target = config.num_points
    current = int(sampled.shape[0])
    if current == target:
        return sampled
    if current > target:
        return sampled[:target]

    missing = target - current
    if config.pad_mode == "zero" or current == 0:
        padding = torch.zeros((missing, source.shape[1]), dtype=source.dtype, device=source.device)
    else:
        repeat_indices = torch.arange(missing, dtype=torch.long, device=source.device) % current
        padding = sampled[repeat_indices]
    return torch.cat([sampled, padding], dim=0)


def _voxel_representative_indices(points_xyz: Tensor, voxel_size: float) -> Tensor:
    """Return first input index per occupied voxel using XYZ coordinates."""

    if points_xyz.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=points_xyz.device)
    keys = torch.floor(points_xyz / voxel_size).to(dtype=torch.int64)
    _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    source_indices = torch.arange(points_xyz.shape[0], dtype=torch.long, device=points_xyz.device)
    representative = torch.full(
        (int(inverse.max().item()) + 1,),
        int(points_xyz.shape[0]),
        dtype=torch.long,
        device=points_xyz.device,
    )
    representative.scatter_reduce_(0, inverse, source_indices, reduce="amin", include_self=True)
    representative = representative[representative < points_xyz.shape[0]]
    return torch.sort(representative).values


def _make_generator(config: SamplingConfig, device: torch.device) -> torch.Generator | None:
    if not config.deterministic or config.seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    return generator


def _sampling_meta(
    *,
    input_count: int,
    candidate_count: int,
    sampled_count: int,
    config: SamplingConfig,
    input_empty: bool,
    padded: bool,
) -> Meta:
    return {
        "input_count": input_count,
        "candidate_count": candidate_count,
        "sampled_count": sampled_count,
        "num_sampled_points": sampled_count,
        "empty_input": input_empty,
        "input_empty": input_empty,
        "padded": padded,
        "mode": config.mode,
        "sampling_enabled": config.enabled,
        "sampling_mode": config.mode,
        "target_num_points": config.num_points,
        "pad_mode": config.pad_mode,
        "voxel_size": config.voxel_size,
    }


def _validate_point_cloud(point_cloud: Tensor) -> None:
    if point_cloud.ndim != 2 or point_cloud.shape[-1] not in {3, 6}:
        raise ValueError("point_cloud must have shape N x 3 or N x 6")
