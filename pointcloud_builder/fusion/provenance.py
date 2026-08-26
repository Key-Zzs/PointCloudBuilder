"""Provenance bookkeeping separate from fused point tensors."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from pointcloud_builder.fusion.types import FusionProvenance


def build_fusion_provenance(
    names: Sequence[str],
    keys_by_camera: Sequence[torch.Tensor],
    unique_keys: torch.Tensor,
    point_counts: torch.Tensor,
) -> FusionProvenance:
    per_input = {name: int(keys.shape[0]) for name, keys in zip(names, keys_by_camera, strict=True)}
    unique_by_camera = [torch.unique(keys, dim=0, sorted=True) for keys in keys_by_camera]
    unique_contribution = {
        name: int(keys.shape[0])
        for name, keys in zip(names, unique_by_camera, strict=True)
    }
    if unique_keys.shape[0]:
        merged_keys, source_counts = torch.unique(
            torch.cat(unique_by_camera, dim=0),
            dim=0,
            sorted=True,
            return_counts=True,
        )
        if not torch.equal(merged_keys, unique_keys):
            raise RuntimeError("fusion provenance key order differs from fused voxel order")
        source_counts = source_counts.to(dtype=torch.int64)
    else:
        source_counts = torch.empty(
            (0,), dtype=torch.int64, device=unique_keys.device
        )
    return FusionProvenance(
        input_point_count=sum(per_input.values()),
        output_voxel_count=int(unique_keys.shape[0]),
        per_camera_input_count=per_input,
        per_camera_unique_voxel_contribution=unique_contribution,
        multi_camera_voxel_count=int((source_counts > 1).sum().item()),
        voxel_keys=unique_keys,
        per_voxel_source_camera_count=source_counts,
        per_voxel_point_count=point_counts.to(dtype=torch.int64),
    )
