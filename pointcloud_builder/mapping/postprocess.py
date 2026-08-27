"""Reuse workspace crop and global sampling for TSDF-extracted point clouds."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from pointcloud_builder.mapping.config import TsdfPostprocessConfig
from pointcloud_builder.sampling import sample_point_cloud
from pointcloud_builder.workspace.crop import crop_workspace_cloud
from pointcloud_builder.workspace.types import WorkspacePointCloud


@dataclass(frozen=True)
class MapPostprocessResult:
    raw: WorkspacePointCloud
    cropped: WorkspacePointCloud
    sampled: WorkspacePointCloud
    crop_ms: float
    sampling_ms: float
    sampling_metadata: dict[str, object]


def postprocess_extracted_cloud(
    points: np.ndarray | torch.Tensor,
    *,
    workspace_frame: str,
    config: TsdfPostprocessConfig,
    device: str | torch.device = "cpu",
) -> MapPostprocessResult:
    """Apply the established workspace crop then sampling implementations."""

    tensor_device = _torch_device(device)
    started = time.perf_counter()
    tensor = (
        points.to(device=tensor_device)
        if isinstance(points, torch.Tensor)
        else torch.as_tensor(np.ascontiguousarray(points), device=tensor_device)
    )
    raw = WorkspacePointCloud(points=tensor, frame=workspace_frame)

    _synchronize(tensor)
    cropped = crop_workspace_cloud(raw, config.crop)
    _synchronize(cropped.points)
    crop_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    sampled_points, metadata = sample_point_cloud(cropped.points, config.sampling)
    _synchronize(sampled_points)
    sampling_ms = (time.perf_counter() - started) * 1000.0
    sampled = WorkspacePointCloud(
        points=sampled_points,
        frame=workspace_frame,
        metadata={**cropped.metadata, "sampling": metadata},
    )
    return MapPostprocessResult(
        raw=raw,
        cropped=cropped,
        sampled=sampled,
        crop_ms=crop_ms,
        sampling_ms=sampling_ms,
        sampling_metadata=dict(metadata),
    )


def cloud_numpy(cloud: WorkspacePointCloud) -> np.ndarray:
    return np.ascontiguousarray(cloud.points.detach().cpu().numpy())


def _torch_device(value: str | torch.device) -> torch.device:
    if isinstance(value, torch.device):
        return value
    normalized = value.strip().lower()
    if normalized.startswith("cuda"):
        suffix = normalized.partition(":")[2]
        return torch.device(f"cuda:{suffix or '0'}")
    return torch.device("cpu")


def _synchronize(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.current_stream(tensor.device).synchronize()
