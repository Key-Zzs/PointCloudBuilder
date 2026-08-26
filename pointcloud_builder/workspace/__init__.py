"""Frame-explicit point-cloud transforms and single-camera workspace pipelines."""

from pointcloud_builder.workspace.crop import crop_workspace_cloud
from pointcloud_builder.workspace.pipeline import (
    SingleCameraWorkspacePipeline,
    SingleCameraWorkspaceStages,
)
from pointcloud_builder.workspace.transform import transform_point_cloud
from pointcloud_builder.workspace.types import FramedPointCloud, WorkspacePointCloud
from pointcloud_builder.workspace.validation import (
    ExpectedPlaneRegion,
    PlaneMetrics,
    evaluate_expected_plane,
    select_expected_plane_points,
)

__all__ = [
    "ExpectedPlaneRegion",
    "FramedPointCloud",
    "PlaneMetrics",
    "SingleCameraWorkspacePipeline",
    "SingleCameraWorkspaceStages",
    "WorkspacePointCloud",
    "crop_workspace_cloud",
    "evaluate_expected_plane",
    "select_expected_plane_points",
    "transform_point_cloud",
]
