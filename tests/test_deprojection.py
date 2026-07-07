from __future__ import annotations

import torch

from pointcloud_builder.camera_model import CameraModel
from pointcloud_builder.config import CameraConfig, CameraIntrinsics
from pointcloud_builder.deprojection import deproject_depth


def test_deprojection_outputs_camera_points() -> None:
    camera = CameraModel.from_config(
        CameraConfig(
            width=2,
            height=2,
            depth_scale=1.0,
            aligned_depth_to_color=False,
            intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0),
        )
    )
    points, mask = deproject_depth(torch.ones((2, 2)), camera)
    assert points.shape == (4, 3)
    assert mask.shape == (4,)
