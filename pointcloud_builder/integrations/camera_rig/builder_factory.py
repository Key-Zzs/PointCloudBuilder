"""Construct PCB builders directly from an authoritative CameraRig bundle."""

from __future__ import annotations

from typing import Any

from pointcloud_builder.builder import PointCloudBuilder
from pointcloud_builder.config import (
    CameraConfig,
    CropConfig,
    DepthSourceConfig,
    PointCloudBuilderConfig,
    PointCloudConfig,
    SamplingConfig,
)
from pointcloud_builder.integrations.camera_rig.calibration_adapter import (
    calibration_from_camera_bundle,
)
from pointcloud_builder.integrations.camera_rig.dependencies import CameraBundle
from pointcloud_builder.integrations.camera_rig.frame_adapter import CameraRigFrameAdapter
from pointcloud_builder.integrations.camera_rig.types import CameraRigBuilderContext


def create_native_builder(
    bundle: CameraBundle,
    *,
    camera_name: str | None = None,
    device: str = "auto",
    crop: CropConfig | None = None,
    sampling: SamplingConfig | None = None,
) -> CameraRigBuilderContext:
    """Create an unaligned raw-depth XYZ builder without hand-copied calibration."""

    calibration = calibration_from_camera_bundle(
        bundle,
        camera_name=camera_name,
        required_streams=("color", "depth", "ir_left"),
    )
    T_color_from_depth = calibration.transform(
        calibration.intrinsic_frames["depth"], calibration.intrinsic_frames["color"]
    )
    source_frame = calibration.intrinsic_frames["depth"]
    config = PointCloudBuilderConfig(
        camera=CameraConfig(
            name=calibration.camera_name,
            depth_scale=calibration.depth_scale_m_per_unit,
            aligned_depth_to_color=False,
            color_intrinsics=calibration.intrinsics["color"],
            depth_intrinsics=calibration.intrinsics["depth"],
            depth_to_color_extrinsics=T_color_from_depth.extrinsics,
        ),
        pointcloud=PointCloudConfig(
            use_rgb=False,
            output_format="xyz",
            xyz_frame="depth",
        ),
        device=device,
        crop=crop or _disabled_crop(),
        sampling=sampling or SamplingConfig(mode="voxel_random", num_points=1024),
        depth_source=DepthSourceConfig(mode="frame"),
    )
    return CameraRigBuilderContext(
        builder=PointCloudBuilder(config),
        calibration=calibration,
        source_frame=source_frame,
        workspace_frame=calibration.workspace_frame,
        T_workspace_from_source=calibration.transform(source_frame, calibration.workspace_frame),
        depth_mode="native",
        frame_adapter=CameraRigFrameAdapter(
            bundle,
            required_streams=("depth",),
            timestamp_stream="depth",
        ),
    )


def create_ffs_builder(
    bundle: CameraBundle,
    *,
    ffs_config: Any,
    device: str = "auto",
    crop: CropConfig | None = None,
    sampling: SamplingConfig | None = None,
) -> CameraRigBuilderContext:
    """Reserved M3 interface for bundle-derived FFS calibration."""

    raise NotImplementedError(
        "CameraRig FFS builder construction is introduced in M3; use create_native_builder in M1/M2"
    )


def _disabled_crop() -> CropConfig:
    return CropConfig(
        enabled=False,
        x=(-float("inf"), float("inf")),
        y=(-float("inf"), float("inf")),
        z=(-float("inf"), float("inf")),
        frame="camera",
    )
