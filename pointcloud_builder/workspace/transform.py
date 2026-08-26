"""Pure-Torch transforms for frame-explicit point clouds."""

from __future__ import annotations

import torch

from pointcloud_builder.integrations.camera_rig.types import FrameExplicitTransform
from pointcloud_builder.workspace.types import FramedPointCloud, WorkspacePointCloud


def transform_point_cloud(
    cloud: FramedPointCloud,
    transform: FrameExplicitTransform,
) -> WorkspacePointCloud:
    """Apply ``xyz @ R.T + t`` on the input tensor's existing device."""

    if cloud.frame != transform.source_frame:
        raise ValueError(
            f"point-cloud frame {cloud.frame!r} does not match transform source "
            f"{transform.source_frame!r}"
        )
    matrix = torch.tensor(
        transform.matrix,
        dtype=cloud.points.dtype,
        device=cloud.points.device,
    )
    xyz = cloud.points[:, :3]
    transformed_xyz = xyz @ matrix[:3, :3].T + matrix[:3, 3]
    transformed = (
        transformed_xyz
        if cloud.points.shape[1] == 3
        else torch.cat((transformed_xyz, cloud.points[:, 3:]), dim=1)
    )
    metadata = dict(cloud.metadata)
    metadata["transform"] = {
        "source_frame": transform.source_frame,
        "target_frame": transform.target_frame,
        "convention": "T_target_from_source",
    }
    return WorkspacePointCloud(
        points=transformed,
        frame=transform.target_frame,
        metadata=metadata,
    )
