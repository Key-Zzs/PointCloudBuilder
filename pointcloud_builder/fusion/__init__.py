"""Deterministic current-snapshot workspace fusion."""

from pointcloud_builder.fusion.config import VoxelFusionConfig
from pointcloud_builder.fusion.acceptance import evaluate_interference, evaluate_repeatability
from pointcloud_builder.fusion.metrics import synthetic_geometry_metrics
from pointcloud_builder.fusion.real_metrics import (
    CubeMetrics,
    board_surface_metrics,
    contribution_metrics,
    cube_box_voxel_count,
    detect_cube,
    fusion_geometry_metrics,
    symmetric_overlap_metrics,
    voxel_centroids,
)
from pointcloud_builder.fusion.sampling import sample_fused_cloud
from pointcloud_builder.fusion.types import FusionProvenance, FusionResult
from pointcloud_builder.fusion.voxel import voxel_fuse_workspace_clouds

__all__ = [
    "FusionProvenance",
    "FusionResult",
    "CubeMetrics",
    "VoxelFusionConfig",
    "sample_fused_cloud",
    "board_surface_metrics",
    "contribution_metrics",
    "cube_box_voxel_count",
    "detect_cube",
    "evaluate_interference",
    "evaluate_repeatability",
    "fusion_geometry_metrics",
    "symmetric_overlap_metrics",
    "synthetic_geometry_metrics",
    "voxel_fuse_workspace_clouds",
    "voxel_centroids",
]
