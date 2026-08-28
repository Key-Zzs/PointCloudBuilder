"""Joint-calibration observation projection with explicit transform directions."""

from __future__ import annotations

import numpy as np
import torch

from pointcloud_builder.camera_model import CameraIntrinsics
from pointcloud_builder.projection import project_points
from pointcloud_builder.rig_calibration.se3 import inverse, transform_points


def project_target_points(
    object_points_target: np.ndarray,
    T_workspace_from_target: np.ndarray,
    T_workspace_from_camera: np.ndarray,
    projection_model: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Project target points and return pixels plus positive-depth mask."""

    points_workspace = transform_points(T_workspace_from_target, object_points_target)
    points_camera = transform_points(inverse(T_workspace_from_camera), points_workspace)
    result = project_points(torch.from_numpy(points_camera), projection_model)
    return (
        result.pixels_px.detach().cpu().numpy(),
        result.in_front.detach().cpu().numpy(),
    )
