"""Single-camera CameraRig-to-workspace point-cloud pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from pointcloud_builder.config import CropConfig
from pointcloud_builder.integrations.camera_rig.types import CameraRigBuilderContext
from pointcloud_builder.sampling import sample_point_cloud
from pointcloud_builder.workspace.crop import crop_workspace_cloud
from pointcloud_builder.workspace.transform import transform_point_cloud
from pointcloud_builder.workspace.types import FramedPointCloud, WorkspacePointCloud


@dataclass(frozen=True)
class SingleCameraWorkspaceStages:
    """Camera and workspace stages from one deprojection pass."""

    camera_raw: FramedPointCloud
    camera_cropped: FramedPointCloud
    camera_sampled: FramedPointCloud
    workspace_raw: WorkspacePointCloud
    workspace_cropped: WorkspacePointCloud
    workspace_sampled: WorkspacePointCloud
    metadata: dict[str, Any]


class SingleCameraWorkspacePipeline:
    """Build native/FFS camera clouds and transform them into one workspace."""

    def __init__(
        self,
        context: CameraRigBuilderContext,
        *,
        workspace_crop: CropConfig,
    ) -> None:
        if workspace_crop.frame != context.workspace_frame:
            raise ValueError("workspace crop frame must match the CameraRig bundle parent frame")
        self.context = context
        self.workspace_crop = workspace_crop

    def process(self, camera_frame: Any) -> SingleCameraWorkspaceStages:
        """Adapt, deproject once, and expose camera/workspace stages."""

        pipeline_start = time.perf_counter()
        adapter_start = pipeline_start
        mapping = self.context.frame_adapter.adapt(camera_frame)
        adapter_ms = (time.perf_counter() - adapter_start) * 1000.0
        stages, builder_meta = self.context.builder.build_stages(mapping)
        camera_clouds = {
            name: FramedPointCloud(
                points=points,
                frame=self.context.source_frame,
                metadata={"camera_name": mapping["camera_name"], "stage": f"camera_{name}"},
            )
            for name, points in stages.items()
        }
        transform_start = time.perf_counter()
        workspace_raw = transform_point_cloud(
            camera_clouds["cropped"], self.context.T_workspace_from_source
        )
        _sync_if_cuda(workspace_raw.points)
        transform_ms = (time.perf_counter() - transform_start) * 1000.0
        workspace_crop_start = time.perf_counter()
        workspace_cropped = crop_workspace_cloud(workspace_raw, self.workspace_crop)
        _sync_if_cuda(workspace_cropped.points)
        workspace_crop_ms = (time.perf_counter() - workspace_crop_start) * 1000.0
        workspace_sampling_start = time.perf_counter()
        sampled, sampling_meta = sample_point_cloud(
            workspace_cropped.points,
            self.context.builder.config.sampling,
        )
        workspace_sampled = WorkspacePointCloud(
            points=sampled,
            frame=self.context.workspace_frame,
            metadata={**workspace_cropped.metadata, "sampling": sampling_meta},
        )
        _sync_if_cuda(workspace_sampled.points)
        workspace_sampling_ms = (time.perf_counter() - workspace_sampling_start) * 1000.0
        builder_timing = dict(builder_meta.get("timing_ms", {}))
        ffs_timing = dict(builder_meta.get("ffs", {}).get("timing_ms", {}))
        timing_ms = {
            "frame_adapter": adapter_ms,
            "depth_inference": float(ffs_timing.get("inference", 0.0)),
            "deprojection": float(builder_timing.get("deprojection", 0.0)),
            "local_crop": float(builder_timing.get("crop", 0.0)),
            "camera_sampling": float(builder_timing.get("sampling", 0.0)),
            "workspace_transform": transform_ms,
            "workspace_crop": workspace_crop_ms,
            "sampling": workspace_sampling_ms,
            "total_workspace_pipeline": (time.perf_counter() - pipeline_start) * 1000.0,
        }
        return SingleCameraWorkspaceStages(
            camera_raw=camera_clouds["raw"],
            camera_cropped=camera_clouds["cropped"],
            camera_sampled=camera_clouds["sampled"],
            workspace_raw=workspace_raw,
            workspace_cropped=workspace_cropped,
            workspace_sampled=workspace_sampled,
            metadata={
                "camera_name": mapping["camera_name"],
                "timestamp": mapping["timestamp"],
                "timestamp_ns": mapping["timestamp_ns"],
                "host_receive_timestamp_ns": mapping["host_receive_timestamp_ns"],
                "frame_numbers": mapping["frame_numbers"],
                "source_frame": self.context.source_frame,
                "workspace_frame": self.context.workspace_frame,
                "depth_mode": self.context.depth_mode,
                "builder": builder_meta,
                "timing_ms": timing_ms,
            },
        )


def _sync_if_cuda(points: Any) -> None:
    if getattr(points, "is_cuda", False):
        import torch

        torch.cuda.current_stream(points.device).synchronize()
