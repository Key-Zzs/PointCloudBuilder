"""Voxel fusion outputs and provenance sidecars."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pointcloud_builder.workspace.types import WorkspacePointCloud


@dataclass(frozen=True)
class FusionProvenance:
    input_point_count: int
    output_voxel_count: int
    per_camera_input_count: dict[str, int]
    per_camera_unique_voxel_contribution: dict[str, int]
    multi_camera_voxel_count: int
    voxel_keys: torch.Tensor
    per_voxel_source_camera_count: torch.Tensor
    per_voxel_point_count: torch.Tensor

    def to_summary(self) -> dict[str, object]:
        return {
            "input_point_count": self.input_point_count,
            "output_voxel_count": self.output_voxel_count,
            "per_camera_input_count": dict(self.per_camera_input_count),
            "per_camera_unique_voxel_contribution": dict(
                self.per_camera_unique_voxel_contribution
            ),
            "multi_camera_voxel_count": self.multi_camera_voxel_count,
            "voxel_keys": self.voxel_keys.tolist(),
            "per_voxel_source_camera_count": self.per_voxel_source_camera_count.tolist(),
            "per_voxel_point_count": self.per_voxel_point_count.tolist(),
        }


@dataclass(frozen=True)
class FusionResult:
    cloud: WorkspacePointCloud
    provenance: FusionProvenance
