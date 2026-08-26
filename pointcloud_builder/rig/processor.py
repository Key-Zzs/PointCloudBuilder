"""Shared deterministic processing for an already-matched rig frame set."""

from __future__ import annotations

from collections.abc import Mapping
import time
from typing import Any

import torch

from pointcloud_builder.fusion import sample_fused_cloud, voxel_fuse_workspace_clouds
from pointcloud_builder.rig.config import RigConfig
from pointcloud_builder.rig.types import (
    PerCameraCloud,
    PerCameraFramedCloud,
    RigBuildResult,
    RigFrameSet,
    WorkspaceCloud,
)
from pointcloud_builder.mapping.types import RigDepthFrameSet
from pointcloud_builder.workspace.crop import crop_workspace_cloud
from pointcloud_builder.workspace.types import WorkspacePointCloud


class RigFrameProcessor:
    """Apply the M6 per-camera, crop, fusion, and global-sampling contract."""

    def __init__(self, config: RigConfig, runtimes: Mapping[str, Any]) -> None:
        expected = {camera.name for camera in config.enabled_cameras}
        actual = set(runtimes)
        if actual != expected:
            raise ValueError(
                "rig processor cameras mismatch: "
                f"expected {sorted(expected)}, got {sorted(actual)}"
            )
        self.config = config
        self.runtimes = dict(runtimes)
        self.canonical_order = tuple(sorted(self.runtimes))

    def process_frame_set(
        self,
        frame_set: RigFrameSet,
        *,
        frame_match_ms: float = 0.0,
        total_start_s: float | None = None,
    ) -> RigBuildResult:
        """Process one complete frame set without retaining tensors in metadata."""

        processing_start = time.perf_counter()
        total_start = processing_start if total_start_s is None else total_start_s
        if frame_set.unmatched_cameras:
            raise ValueError(
                "nearest_host_timestamp unmatched cameras: "
                f"{list(frame_set.unmatched_cameras)}"
            )
        actual_cameras = set(frame_set.envelopes)
        expected_cameras = set(self.canonical_order)
        if actual_cameras != expected_cameras:
            raise ValueError(
                "rig frame set cameras mismatch: "
                f"expected {sorted(expected_cameras)}, got {sorted(actual_cameras)}"
            )

        per_camera: list[PerCameraCloud] = []
        per_camera_camera: list[PerCameraFramedCloud] = []
        per_camera_workspace_raw: list[WorkspacePointCloud] = []
        per_camera_stage_statistics: dict[str, dict[str, Any]] = {}
        timing: dict[str, Any] = {"frame_match": frame_match_ms, "per_camera": {}}
        depth_observations = []
        for name in self.canonical_order:
            runtime = self.runtimes[name]
            envelope = frame_set.envelopes[name]
            start = time.perf_counter()
            stages = runtime.pipeline.process(envelope.frame)
            depth_observations.append(stages.depth_observation)
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
            per_camera_stage_statistics[name] = _stage_statistics(
                stages,
                image_width=int(runtime.pipeline.context.builder.camera.width),
                image_height=int(runtime.pipeline.context.builder.camera.height),
                source_frame=runtime.pipeline.context.source_frame,
                depth_mode=runtime.pipeline.context.depth_mode,
                frame_index=envelope.frame_index,
                host_receive_timestamp_ns=envelope.host_receive_timestamp_ns,
            )

        concat_start = time.perf_counter()
        tensors = [cloud.points for cloud in per_camera_workspace_raw]
        if len({int(points.shape[1]) for points in tensors}) != 1:
            raise ValueError("per-camera clouds must have the same channel count")
        concatenated = torch.cat(tensors, dim=0)
        pre_sampling_counts = {
            item.camera_name: int(item.cloud.points.shape[0]) for item in per_camera
        }
        concatenated_cloud = WorkspacePointCloud(
            points=concatenated,
            frame=self.config.output_frame,
            metadata={
                "schema_version": self.config.schema_version,
                "canonical_camera_order": list(self.canonical_order),
                "timing_mode": self.config.timing.mode,
            },
        )
        workspace_cropped = crop_workspace_cloud(
            concatenated_cloud, self.config.workspace_crop
        )
        fusion_inputs = [
            WorkspaceCloud(item.camera_name, item.cloud) for item in per_camera
        ]
        if self.config.fusion.enabled:
            fusion_result = voxel_fuse_workspace_clouds(
                fusion_inputs, self.config.fusion
            )
            fused = fusion_result.cloud
            fusion_provenance = fusion_result.provenance
        else:
            fused = workspace_cropped
            fusion_provenance = None
        sampled = sample_fused_cloud(fused, self.config.sampling)
        processing_metadata = {
            "processor": "RigFrameProcessor",
            "camera_count": len(self.canonical_order),
            "canonical_camera_order": list(self.canonical_order),
            "per_camera_processing": "sequential_canonical_order",
            "concatenation_input_stage": "per_camera_workspace_raw",
            "workspace_crop_stage": "after_concatenation",
            "fusion_input_stage": "per_camera_workspace_cropped",
            "global_sampling_input_stage": (
                "fused" if self.config.fusion.enabled else "workspace_cropped"
            ),
            "point_counts": {
                "concatenated": int(concatenated_cloud.points.shape[0]),
                "workspace_cropped": int(workspace_cropped.points.shape[0]),
                "fused": int(fused.points.shape[0]),
                "sampled": int(sampled.points.shape[0]),
            },
        }
        stable_metadata = {
            "schema_version": self.config.schema_version,
            "canonical_camera_order": list(self.canonical_order),
            "per_camera_pre_sampling_counts": pre_sampling_counts,
            "pre_sampling_count": int(workspace_cropped.points.shape[0]),
            "fusion_enabled": self.config.fusion.enabled,
            "timing_mode": self.config.timing.mode,
            "per_camera_stage_statistics": per_camera_stage_statistics,
            "processing_metadata": processing_metadata,
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
        if frame_set.match_timestamp_ns is None:
            raise ValueError("rig depth output requires a matched-set timestamp")
        matched_set_index = frame_set.match_sequence_index
        if matched_set_index is None:
            matched_set_index = frame_set.envelopes[frame_set.reference_camera].frame_index
        depth_frame_set = RigDepthFrameSet(
            matched_set_index=matched_set_index,
            host_timestamp_ns=frame_set.match_timestamp_ns,
            maximum_skew_ms=frame_set.maximum_skew_ms,
            observations=tuple(depth_observations),
        )
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
            depth_frame_set=depth_frame_set,
            per_camera_stage_statistics=per_camera_stage_statistics,
            processing_metadata=processing_metadata,
        )


def _stage_statistics(
    stages: Any,
    *,
    image_width: int,
    image_height: int,
    source_frame: str,
    depth_mode: str,
    frame_index: int,
    host_receive_timestamp_ns: int,
) -> dict[str, Any]:
    builder = stages.metadata.get("builder", {})
    ffs = builder.get("ffs", {}) if isinstance(builder, dict) else {}
    total_depth_pixels = image_width * image_height
    valid_depth_count = int(builder.get("num_raw_points", stages.camera_raw.points.shape[0]))
    valid_disparity_count = ffs.get("valid_disparity_count")
    invalid_disparity_count = ffs.get("invalid_disparity_count")
    disparity_total = (
        int(valid_disparity_count) + int(invalid_disparity_count)
        if valid_disparity_count is not None and invalid_disparity_count is not None
        else None
    )
    return {
        "source_frame": source_frame,
        "workspace_frame": stages.workspace_raw.frame,
        "depth_mode": depth_mode,
        "frame_index": frame_index,
        "host_receive_timestamp_ns": host_receive_timestamp_ns,
        "camera_raw_point_count": int(stages.camera_raw.points.shape[0]),
        "camera_cropped_point_count": int(stages.camera_cropped.points.shape[0]),
        "camera_sampled_point_count": int(stages.camera_sampled.points.shape[0]),
        "workspace_raw_point_count": int(stages.workspace_raw.points.shape[0]),
        "workspace_cropped_point_count": int(stages.workspace_cropped.points.shape[0]),
        "workspace_sampled_point_count": int(stages.workspace_sampled.points.shape[0]),
        "camera_point_channels": int(stages.camera_raw.points.shape[1]),
        "workspace_point_channels": int(stages.workspace_raw.points.shape[1]),
        "total_depth_pixel_count": total_depth_pixels,
        "valid_depth_count": valid_depth_count,
        "valid_depth_ratio": valid_depth_count / total_depth_pixels,
        "valid_disparity_count": (
            None if valid_disparity_count is None else int(valid_disparity_count)
        ),
        "valid_disparity_ratio": (
            None
            if valid_disparity_count is None or not disparity_total
            else int(valid_disparity_count) / disparity_total
        ),
        "ffs_backend": ffs.get("backend"),
    }
