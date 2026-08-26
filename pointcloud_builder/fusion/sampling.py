"""Global post-fusion sampling."""

from __future__ import annotations

from pointcloud_builder.config import SamplingConfig
from pointcloud_builder.sampling import sample_point_cloud
from pointcloud_builder.workspace.types import WorkspacePointCloud


def sample_fused_cloud(
    cloud: WorkspacePointCloud, config: SamplingConfig
) -> WorkspacePointCloud:
    points, metadata = sample_point_cloud(cloud.points, config)
    return WorkspacePointCloud(
        points=points,
        frame=cloud.frame,
        metadata={**cloud.metadata, "global_sampling": metadata},
    )
