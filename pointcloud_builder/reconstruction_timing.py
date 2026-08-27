"""Stable, JSON-safe timing contracts for snapshot and persistent reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics
from typing import Any, Literal

import numpy as np

TimingPath = Literal["current_snapshot", "persistent_tsdf_update", "tsdf_extraction"]

CURRENT_SNAPSHOT_STAGES = frozenset(
    {
        "combined_per_camera_sequential_ms",
        "concatenate_ms",
        "workspace_crop_ms",
        "voxel_fusion_ms",
        "global_sampling_ms",
        "raw_to_world_concatenated_ms",
        "raw_to_world_fused_ms",
        "raw_to_world_cropped_ms",
        "raw_to_world_sampled_ms",
    }
)
CURRENT_CAMERA_STAGES = frozenset(
    {
        "frame_adapter_ms",
        "depth_resolution_ms",
        "deprojection_ms",
        "local_crop_ms",
        "workspace_transform_ms",
        "raw_to_workspace_per_camera_ms",
    }
)
TSDF_UPDATE_STAGES = frozenset(
    {
        "block_activation_plus_coordinate_generation_ms",
        "volume_integrate_ms",
        "map_update_total_ms",
        "raw_to_tsdf_update_ms",
    }
)
TSDF_EXTRACTION_STAGES = frozenset(
    {
        "extract_point_cloud_ms",
        "extract_mesh_ms",
        "post_crop_ms",
        "post_sampling_ms",
        "map_to_raw_cloud_ms",
        "map_to_cropped_cloud_ms",
        "map_to_sampled_cloud_ms",
    }
)


@dataclass(frozen=True)
class ReconstructionTiming:
    """One measured reconstruction sample with stable stage names."""

    path: TimingPath
    stages_ms: dict[str, float]
    per_camera_ms: dict[str, dict[str, float]] = field(default_factory=dict)
    frame_match_ms: float | None = None
    capture_ms: float | None = None
    capture_inclusive_total_ms: float | None = None
    schema_version: str = "pointcloud-builder.reconstruction-timing.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "pointcloud-builder.reconstruction-timing.v1":
            raise ValueError("unsupported reconstruction timing schema")
        allowed = {
            "current_snapshot": CURRENT_SNAPSHOT_STAGES,
            "persistent_tsdf_update": TSDF_UPDATE_STAGES,
            "tsdf_extraction": TSDF_EXTRACTION_STAGES,
        }[self.path]
        unknown = sorted(set(self.stages_ms) - allowed)
        if unknown:
            raise ValueError(f"unknown {self.path} timing stages: {unknown}")
        _validate_values(self.stages_ms, "stage")
        for camera, values in self.per_camera_ms.items():
            if not camera.strip():
                raise ValueError("timing camera name must be non-empty")
            unknown_camera = sorted(set(values) - CURRENT_CAMERA_STAGES)
            if unknown_camera:
                raise ValueError(f"unknown per-camera timing stages: {unknown_camera}")
            _validate_values(values, f"camera {camera}")
        for name in ("frame_match_ms", "capture_ms", "capture_inclusive_total_ms"):
            value = getattr(self, name)
            if value is not None:
                _validate_value(value, name)
        object.__setattr__(self, "stages_ms", dict(self.stages_ms))
        object.__setattr__(
            self,
            "per_camera_ms",
            {name: dict(values) for name, values in self.per_camera_ms.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "path": self.path,
            "processing_only": {
                "stages_ms": dict(self.stages_ms),
                "per_camera_ms": {
                    name: dict(values)
                    for name, values in sorted(self.per_camera_ms.items())
                },
            },
        }
        capture = {
            "capture_ms": self.capture_ms,
            "frame_match_ms": self.frame_match_ms,
            "capture_inclusive_total_ms": self.capture_inclusive_total_ms,
        }
        if any(value is not None for value in capture.values()):
            result["capture_inclusive"] = capture
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ReconstructionTiming:
        if value.get("schema_version") != "pointcloud-builder.reconstruction-timing.v1":
            raise ValueError("unsupported reconstruction timing schema")
        processing = value.get("processing_only")
        if not isinstance(processing, dict):
            raise ValueError("timing processing_only must be a mapping")
        capture = value.get("capture_inclusive", {})
        if not isinstance(capture, dict):
            raise ValueError("timing capture_inclusive must be a mapping")
        return cls(
            path=value.get("path"),
            stages_ms=processing.get("stages_ms", {}),
            per_camera_ms=processing.get("per_camera_ms", {}),
            frame_match_ms=capture.get("frame_match_ms"),
            capture_ms=capture.get("capture_ms"),
            capture_inclusive_total_ms=capture.get("capture_inclusive_total_ms"),
        )


def aggregate_reconstruction_timings(
    samples: list[ReconstructionTiming] | tuple[ReconstructionTiming, ...],
) -> dict[str, Any]:
    if not samples:
        raise ValueError("at least one reconstruction timing sample is required")
    path = samples[0].path
    if any(sample.path != path for sample in samples):
        raise ValueError("cannot aggregate different reconstruction timing paths")
    expected_stages = set(samples[0].stages_ms)
    expected_cameras = set(samples[0].per_camera_ms)
    if any(set(item.stages_ms) != expected_stages for item in samples[1:]):
        raise ValueError("timing samples have inconsistent stage keys")
    if any(set(item.per_camera_ms) != expected_cameras for item in samples[1:]):
        raise ValueError("timing samples have inconsistent camera keys")
    stage_names = sorted(expected_stages)
    cameras = sorted(expected_cameras)
    camera_stages = {
        camera: sorted(samples[0].per_camera_ms[camera])
        for camera in cameras
    }
    if any(
        set(item.per_camera_ms[camera]) != set(camera_stages[camera])
        for item in samples[1:]
        for camera in cameras
    ):
        raise ValueError("timing samples have inconsistent per-camera stage keys")
    capture_summary = {
        name: summarize_ms(values)
        for name in (
            "capture_ms",
            "frame_match_ms",
            "capture_inclusive_total_ms",
        )
        if (
            values := [
                float(getattr(item, name))
                for item in samples
                if getattr(item, name) is not None
            ]
        )
    }
    result = {
        "schema_version": "pointcloud-builder.reconstruction-timing-summary.v1",
        "path": path,
        "sample_count": len(samples),
        "processing_only": {
            "stages_ms": {
                name: summarize_ms([item.stages_ms[name] for item in samples])
                for name in stage_names
            },
            "per_camera_ms": {
                camera: {
                    name: summarize_ms(
                        [item.per_camera_ms[camera][name] for item in samples]
                    )
                    for name in camera_stages[camera]
                }
                for camera in cameras
            },
        },
    }
    if capture_summary:
        result["capture_inclusive"] = capture_summary
    return result


def summarize_ms(values: list[float] | tuple[float, ...]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize empty timing values")
    checked = [float(value) for value in values]
    _validate_values({str(index): value for index, value in enumerate(checked)}, "sample")
    return {
        "p50": float(statistics.median(checked)),
        "p95": float(np.quantile(checked, 0.95)),
        "mean": float(statistics.mean(checked)),
        "max": float(max(checked)),
    }


def _validate_values(values: dict[str, float], label: str) -> None:
    for name, value in values.items():
        _validate_value(value, f"{label} {name}")


def _validate_value(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} timing must be numeric")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{label} timing must be finite and non-negative")
