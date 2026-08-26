"""Deterministic current-snapshot workspace fusion."""

from pointcloud_builder.fusion.config import VoxelFusionConfig
from pointcloud_builder.fusion.metrics import synthetic_geometry_metrics
from pointcloud_builder.fusion.sampling import sample_fused_cloud
from pointcloud_builder.fusion.types import FusionProvenance, FusionResult
from pointcloud_builder.fusion.voxel import voxel_fuse_workspace_clouds

__all__ = [
    "FusionProvenance",
    "FusionResult",
    "VoxelFusionConfig",
    "sample_fused_cloud",
    "synthetic_geometry_metrics",
    "voxel_fuse_workspace_clouds",
]
