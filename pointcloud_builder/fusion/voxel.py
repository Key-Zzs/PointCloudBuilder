"""Order-invariant deterministic voxel centroid aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from pointcloud_builder.fusion.config import VoxelFusionConfig
from pointcloud_builder.fusion.provenance import build_fusion_provenance
from pointcloud_builder.fusion.types import FusionResult
from pointcloud_builder.workspace.types import WorkspacePointCloud


def voxel_fuse_workspace_clouds(
    clouds: Sequence[Any], config: VoxelFusionConfig
) -> FusionResult:
    if not clouds:
        raise ValueError("voxel fusion requires at least one camera cloud")
    ordered = sorted(clouds, key=lambda item: item.camera_name)
    names = [str(item.camera_name) for item in ordered]
    if len(names) != len(set(names)):
        raise ValueError("voxel fusion camera names must be unique")
    point_clouds = [item.cloud for item in ordered]
    frame = point_clouds[0].frame
    channels = int(point_clouds[0].points.shape[1])
    device = point_clouds[0].points.device
    dtype = point_clouds[0].points.dtype
    for cloud in point_clouds:
        if cloud.frame != frame:
            raise ValueError("all fusion inputs must share one workspace frame")
        if int(cloud.points.shape[1]) != channels:
            raise ValueError("all fusion inputs must share Nx3 or Nx6 shape")
        if cloud.points.device != device or cloud.points.dtype != dtype:
            raise ValueError("all fusion inputs must share dtype and device")
    origin = torch.tensor(config.origin, dtype=dtype, device=device)
    keys_by_camera = [
        torch.floor((cloud.points[:, :3] - origin) / config.voxel_size_m).to(torch.int64)
        for cloud in point_clouds
    ]
    points = torch.cat([cloud.points for cloud in point_clouds], dim=0)
    keys = torch.cat(keys_by_camera, dim=0)
    if points.shape[0] == 0:
        empty_keys = torch.empty((0, 3), dtype=torch.int64, device=device)
        empty_counts = torch.empty((0,), dtype=torch.int64, device=device)
        provenance = build_fusion_provenance(names, keys_by_camera, empty_keys, empty_counts)
        return FusionResult(
            cloud=WorkspacePointCloud(
                points=points.clone(),
                frame=frame,
                metadata={"fusion": provenance.to_summary()},
            ),
            provenance=provenance,
        )
    order = _canonical_order(keys, points) if config.deterministic else _key_order(keys)
    sorted_keys = keys[order]
    sorted_points = points[order]
    unique_keys, counts = torch.unique_consecutive(sorted_keys, dim=0, return_counts=True)
    ends = torch.cumsum(counts, dim=0)
    starts = ends - counts
    prefix = torch.cat(
        (
            torch.zeros((1, channels), dtype=dtype, device=device),
            torch.cumsum(sorted_points, dim=0),
        ),
        dim=0,
    )
    sums = prefix[ends] - prefix[starts]
    fused = sums / counts.to(dtype=dtype).unsqueeze(1)
    provenance = build_fusion_provenance(names, keys_by_camera, unique_keys, counts)
    return FusionResult(
        cloud=WorkspacePointCloud(
            points=fused,
            frame=frame,
            metadata={
                "fusion": {
                    "voxel_size_m": config.voxel_size_m,
                    "origin": list(config.origin),
                    "deterministic": config.deterministic,
                    **provenance.to_summary(),
                }
            },
        ),
        provenance=provenance,
    )


def _canonical_order(keys: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    order = torch.arange(keys.shape[0], device=keys.device)
    for column in reversed(range(points.shape[1])):
        order = order[torch.argsort(points[order, column], stable=True)]
    for column in reversed(range(3)):
        order = order[torch.argsort(keys[order, column], stable=True)]
    return order

def _key_order(keys: torch.Tensor) -> torch.Tensor:
    order = torch.arange(keys.shape[0], device=keys.device)
    for column in reversed(range(3)):
        order = order[torch.argsort(keys[order, column], stable=True)]
    return order
