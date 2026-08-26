"""Deterministic offline rig orchestration with optional snapshot voxel fusion."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch

from pointcloud_builder.fusion import sample_fused_cloud, voxel_fuse_workspace_clouds
from pointcloud_builder.rig.config import RigConfig
from pointcloud_builder.rig.frame_matcher import match_exact_index, match_nearest_host_timestamp
from pointcloud_builder.rig.types import (
    PerCameraCloud,
    PerCameraFramedCloud,
    RigBuildResult,
    WorkspaceCloud,
)
from pointcloud_builder.rig.validation import validate_rig_runtimes
from pointcloud_builder.workspace.crop import crop_workspace_cloud
from pointcloud_builder.workspace.types import WorkspacePointCloud


@dataclass(frozen=True)
class RigCameraRuntime:
    source: Any
    pipeline: Any
    provenance: dict[str, Any]


class OfflineRigPipeline:
    """Match, independently deproject, and canonically concatenate rig frames."""

    def __init__(self, config: RigConfig, runtimes: dict[str, RigCameraRuntime]) -> None:
        validate_rig_runtimes(config, runtimes)
        self.config = config
        self.runtimes = dict(runtimes)
        self.canonical_order = tuple(sorted(self.runtimes))

    def build(self, index: int) -> RigBuildResult:
        total_start = time.perf_counter()
        sources = {name: runtime.source for name, runtime in self.runtimes.items()}
        reference = self.config.timing.reference_camera or self.canonical_order[0]
        match_start = time.perf_counter()
        if self.config.timing.mode == "exact_index":
            frame_set = match_exact_index(sources, index, reference_camera=reference)
        else:
            frame_set = match_nearest_host_timestamp(
                sources,
                index,
                reference_camera=reference,
                maximum_skew_ms=self.config.timing.maximum_skew_ms,
            )
        match_ms = (time.perf_counter() - match_start) * 1000.0
        if frame_set.unmatched_cameras:
            raise ValueError(
                f"nearest_host_timestamp unmatched cameras: {list(frame_set.unmatched_cameras)}"
            )
        per_camera: list[PerCameraCloud] = []
        per_camera_camera: list[PerCameraFramedCloud] = []
        per_camera_workspace_raw: list[WorkspacePointCloud] = []
        timing: dict[str, Any] = {"frame_match": match_ms, "per_camera": {}}
        for name in self.canonical_order:
            runtime = self.runtimes[name]
            envelope = frame_set.envelopes[name]
            start = time.perf_counter()
            stages = runtime.pipeline.process(envelope.frame)
            timing["per_camera"][name] = {
                **stages.metadata["timing_ms"],
                "rig_camera_total": (time.perf_counter() - start) * 1000.0,
            }
            cloud = stages.workspace_cropped
            per_camera_camera.append(
                PerCameraFramedCloud(camera_name=name, cloud=stages.camera_cropped)
            )
            per_camera_workspace_raw.append(stages.workspace_raw)
            per_camera.append(
                PerCameraCloud(
                    camera_name=name,
                    cloud=cloud,
                    source_frame=runtime.pipeline.context.source_frame,
                    depth_mode=runtime.pipeline.context.depth_mode,
                    frame_index=envelope.frame_index,
                    host_receive_timestamp_ns=envelope.host_receive_timestamp_ns,
                    provenance=runtime.provenance,
                )
            )
        concat_start = time.perf_counter()
        tensors = [cloud.points for cloud in per_camera_workspace_raw]
        if len({int(points.shape[1]) for points in tensors}) != 1:
            raise ValueError("per-camera clouds must have the same channel count")
        concatenated = torch.cat(tensors, dim=0)
        pre_sampling_counts = {item.camera_name: int(item.cloud.points.shape[0]) for item in per_camera}
        concatenated_cloud = WorkspacePointCloud(
            points=concatenated,
            frame=self.config.output_frame,
            metadata={
                "schema_version": self.config.schema_version,
                "canonical_camera_order": list(self.canonical_order),
                "timing_mode": self.config.timing.mode,
            },
        )
        workspace_cropped = crop_workspace_cloud(concatenated_cloud, self.config.workspace_crop)
        fusion_inputs = [WorkspaceCloud(item.camera_name, item.cloud) for item in per_camera]
        if self.config.fusion.enabled:
            fusion_result = voxel_fuse_workspace_clouds(fusion_inputs, self.config.fusion)
            fused = fusion_result.cloud
            fusion_provenance = fusion_result.provenance
        else:
            fused = workspace_cropped
            fusion_provenance = None
        sampled = sample_fused_cloud(fused, self.config.sampling)
        stable_metadata = {
            "schema_version": self.config.schema_version,
            "canonical_camera_order": list(self.canonical_order),
            "per_camera_pre_sampling_counts": pre_sampling_counts,
            "pre_sampling_count": int(workspace_cropped.points.shape[0]),
            "fusion_enabled": self.config.fusion.enabled,
            "timing_mode": self.config.timing.mode,
        }
        sampled = WorkspacePointCloud(
            points=sampled.points,
            frame=sampled.frame,
            metadata={**sampled.metadata, **stable_metadata},
        )
        concat_ms = (time.perf_counter() - concat_start) * 1000.0
        timing["concatenate_crop_fuse_and_global_sampling"] = concat_ms
        timing["total"] = (time.perf_counter() - total_start) * 1000.0
        provenance = {item.camera_name: item.provenance for item in per_camera}
        return RigBuildResult(
            per_camera_camera_frame=tuple(per_camera_camera),
            per_camera_workspace=tuple(per_camera),
            concatenated=concatenated_cloud,
            workspace_cropped=workspace_cropped,
            fused=fused,
            sampled=sampled,
            fusion_provenance=fusion_provenance,
            timing_report_ms=timing,
            per_camera_provenance=provenance,
            canonical_camera_order=self.canonical_order,
            frame_match=frame_set,
        )
