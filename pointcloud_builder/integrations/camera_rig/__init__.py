"""Optional CameraRig v1 integration using only ``camera_rig.api``."""

from pointcloud_builder.integrations.camera_rig.builder_factory import (
    create_ffs_builder,
    create_native_builder,
)
from pointcloud_builder.integrations.camera_rig.calibration_adapter import (
    calibration_from_camera_bundle,
    camera_intrinsics_to_pcb,
    resolve_bundle_transform,
    rigid_transform_to_frame_explicit,
)
from pointcloud_builder.integrations.camera_rig.dependencies import CameraRigDependencyError
from pointcloud_builder.integrations.camera_rig.frame_adapter import CameraRigFrameAdapter
from pointcloud_builder.integrations.camera_rig.transform_resolver import (
    TransformResolutionError,
    resolve_transform,
)
from pointcloud_builder.integrations.camera_rig.types import (
    CameraRigBuilderContext,
    CameraRigCalibration,
    FrameExplicitTransform,
)

__all__ = [
    "CameraRigBuilderContext",
    "CameraRigCalibration",
    "CameraRigDependencyError",
    "CameraRigFrameAdapter",
    "FrameExplicitTransform",
    "TransformResolutionError",
    "calibration_from_camera_bundle",
    "camera_intrinsics_to_pcb",
    "create_ffs_builder",
    "create_native_builder",
    "resolve_bundle_transform",
    "resolve_transform",
    "rigid_transform_to_frame_explicit",
]
