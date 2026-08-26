"""Workspace-frame cropping for framed point clouds."""

from __future__ import annotations

from pointcloud_builder.config import CropConfig
from pointcloud_builder.crop import crop_point_cloud
from pointcloud_builder.workspace.types import WorkspacePointCloud


def crop_workspace_cloud(
    cloud: WorkspacePointCloud,
    config: CropConfig,
) -> WorkspacePointCloud:
    """Crop XYZ bounds only after confirming the configured frame."""

    if config.frame != cloud.frame:
        raise ValueError(
            f"workspace crop frame {config.frame!r} does not match cloud frame {cloud.frame!r}"
        )
    cropped, _ = crop_point_cloud(cloud.points, config)
    metadata = dict(cloud.metadata)
    metadata["workspace_crop"] = {
        "enabled": config.enabled,
        "frame": config.frame,
        "x": config.x,
        "y": config.y,
        "z": config.z,
        "input_count": int(cloud.points.shape[0]),
        "output_count": int(cropped.shape[0]),
    }
    return WorkspacePointCloud(points=cropped, frame=cloud.frame, metadata=metadata)
