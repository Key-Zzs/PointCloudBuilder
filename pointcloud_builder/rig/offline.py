"""Factory helpers for validated offline CameraRig replay runtimes."""

from __future__ import annotations

from camera_rig.api import load_provisioned_camera_bundle

from pointcloud_builder.config import SamplingConfig, load_config
from pointcloud_builder.integrations.camera_rig import create_ffs_builder, create_native_builder
from pointcloud_builder.rig.config import RigConfig
from pointcloud_builder.rig.pipeline import OfflineRigPipeline, RigCameraRuntime
from pointcloud_builder.rig.sources import CameraRigReplaySource
from pointcloud_builder.workspace import SingleCameraWorkspacePipeline


def build_replay_rig(config: RigConfig, *, device: str = "auto") -> OfflineRigPipeline:
    runtimes: dict[str, RigCameraRuntime] = {}
    no_sampling = SamplingConfig(mode="stride", num_points=1, enabled=False)
    for camera in config.enabled_cameras:
        if camera.source.type != "camera_rig_replay":
            raise ValueError(f"camera {camera.name!r} requires an injected synthetic runtime")
        bundle = load_provisioned_camera_bundle(camera.source.provision_artifact)
        if camera.depth.mode == "native":
            context = create_native_builder(bundle, camera_name=camera.name, device=device, crop=camera.local_crop, sampling=no_sampling)
        else:
            if camera.pipeline_config is None:
                raise ValueError(f"camera {camera.name!r} FFS mode requires pipeline_config")
            pipeline_config = load_config(camera.pipeline_config)
            if pipeline_config.depth_source.ffs is None:
                raise ValueError(f"camera {camera.name!r} pipeline_config has no FFS section")
            context = create_ffs_builder(bundle, ffs_config=pipeline_config.depth_source.ffs, device=device, crop=camera.local_crop, sampling=no_sampling)
        if context.workspace_frame != config.output_frame:
            raise ValueError(
                f"camera {camera.name!r} output frame differs from bundle parent frame"
            )
        runtimes[camera.name] = RigCameraRuntime(
            source=CameraRigReplaySource(camera.name, camera.source.capture_artifact),
            pipeline=SingleCameraWorkspacePipeline(context, workspace_crop=config.workspace_crop),
            provenance={"source_type": "camera_rig_replay", "depth_mode": camera.depth.mode},
        )
    return OfflineRigPipeline(config, runtimes)
