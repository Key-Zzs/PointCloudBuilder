"""Lazy dependency boundary for the stable CameraRig v1 consumer API."""

from __future__ import annotations


class CameraRigDependencyError(ImportError):
    """Raised when the optional CameraRig consumer dependency is unavailable."""


try:
    from camera_rig.api import (
        CameraBundle,
        CameraConfig,
        CameraFrame,
        CameraIntrinsics,
        CameraSession,
        ReplayCameraSession,
        RigidTransform,
        StreamFrame,
        load_camera_bundle,
        load_camera_config,
        load_provisioned_camera_bundle,
    )
except ImportError as error:  # pragma: no cover - exercised in an isolated subprocess
    raise CameraRigDependencyError(
        "PointCloudBuilder's CameraRig integration requires the pinned CameraRig "
        "consumer package. Initialize third_party/CameraRig and install it with "
        "`python -m pip install -e third_party/CameraRig --no-deps`."
    ) from error


__all__ = [
    "CameraBundle",
    "CameraConfig",
    "CameraFrame",
    "CameraIntrinsics",
    "CameraRigDependencyError",
    "CameraSession",
    "ReplayCameraSession",
    "RigidTransform",
    "StreamFrame",
    "load_camera_bundle",
    "load_camera_config",
    "load_provisioned_camera_bundle",
]
