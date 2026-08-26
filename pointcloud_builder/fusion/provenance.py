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
    key_sources: dict[tuple[int, int, int], set[str]] = {}
    unique_contribution: dict[str, int] = {}
    for name, keys in zip(names, keys_by_camera, strict=True):
        local = {tuple(int(value) for value in row) for row in keys.detach().cpu().tolist()}
        unique_contribution[name] = len(local)
        for key in local:
            key_sources.setdefault(key, set()).add(name)
    ordered_keys = [tuple(int(value) for value in row) for row in unique_keys.detach().cpu().tolist()]
    source_counts = torch.tensor(
        [len(key_sources[key]) for key in ordered_keys],
        dtype=torch.int64,
        device=unique_keys.device,
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
