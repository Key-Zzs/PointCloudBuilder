"""PCB-owned offline multi-camera rig contracts."""

from pointcloud_builder.rig.config import (
    RIG_SCHEMA_VERSION,
    RigCameraConfig,
    RigCalibrationDeploymentConfig,
    RigConfig,
    RigDepthConfig,
    RigPointCloudConfig,
    RigSourceConfig,
    RigTimingConfig,
    load_rig_config,
    parse_rig_config,
)
from pointcloud_builder.rig.offline import build_replay_rig
from pointcloud_builder.rig.live import (
    LiveRigAcquisition,
    LiveRigBuild,
    LiveRigPipeline,
    LiveRigWorkerFailure,
    build_live_rig,
)
from pointcloud_builder.rig.pipeline import OfflineRigPipeline, RigCameraRuntime
from pointcloud_builder.rig.processor import RigFrameProcessor
from pointcloud_builder.rig.sources import CameraRigReplaySource, SyntheticCameraSource
from pointcloud_builder.rig.synthetic import (
    SyntheticScene,
    build_synthetic_rig,
    create_synthetic_scene,
)
from pointcloud_builder.rig.types import (
    CameraFrameEnvelope,
    PerCameraCloud,
    PerCameraFramedCloud,
    RigBuildResult,
    RigFrameSet,
    WorkspaceCloud,
)

__all__ = [
    "RIG_SCHEMA_VERSION",
    "CameraFrameEnvelope",
    "CameraRigReplaySource",
    "OfflineRigPipeline",
    "LiveRigAcquisition",
    "LiveRigBuild",
    "LiveRigPipeline",
    "LiveRigWorkerFailure",
    "PerCameraCloud",
    "PerCameraFramedCloud",
    "RigBuildResult",
    "RigCalibrationDeploymentConfig",
    "RigCameraConfig",
    "RigCameraRuntime",
    "RigFrameProcessor",
    "RigConfig",
    "RigDepthConfig",
    "RigPointCloudConfig",
    "RigFrameSet",
    "RigSourceConfig",
    "RigTimingConfig",
    "SyntheticCameraSource",
    "SyntheticScene",
    "WorkspaceCloud",
    "build_replay_rig",
    "build_live_rig",
    "build_synthetic_rig",
    "create_synthetic_scene",
    "load_rig_config",
    "parse_rig_config",
]
