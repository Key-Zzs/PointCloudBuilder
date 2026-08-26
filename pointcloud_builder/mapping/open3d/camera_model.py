"""Open3D camera conversion with explicit transform direction."""

from __future__ import annotations

import numpy as np

from pointcloud_builder.mapping.types import RigDepthObservation


def intrinsic_matrix(observation: RigDepthObservation) -> np.ndarray:
    intrinsics = observation.intrinsics
    return np.asarray(
        (
            (intrinsics.fx, 0.0, intrinsics.cx),
            (0.0, intrinsics.fy, intrinsics.cy),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def T_camera_from_workspace(observation: RigDepthObservation) -> np.ndarray:
    """Open3D world-to-camera extrinsic, proven by synthetic parity tests."""

    return np.linalg.inv(observation.T_workspace_from_camera)


def open3d_depth_scale(observation: RigDepthObservation) -> float:
    """Open3D divides stored values by depth_scale; PCB stores meters/unit."""

    return 1.0 / float(observation.depth_scale_m_per_unit)
