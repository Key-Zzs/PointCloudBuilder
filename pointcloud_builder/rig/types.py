"""Frame-explicit contracts for offline multi-camera orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pointcloud_builder.workspace.types import WorkspacePointCloud


@dataclass(frozen=True)
class CameraFrameEnvelope:
    camera_name: str
    frame_index: int
    host_receive_timestamp_ns: int
    frame: Any


@dataclass(frozen=True)
class RigFrameSet:
    envelopes: dict[str, CameraFrameEnvelope]
    reference_camera: str
    per_camera_delta_ms: dict[str, float]
    maximum_skew_ms: float
    unmatched_cameras: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "envelopes", dict(self.envelopes))
        object.__setattr__(self, "per_camera_delta_ms", dict(self.per_camera_delta_ms))


@dataclass(frozen=True)
class WorkspaceCloud:
    camera_name: str
    cloud: WorkspacePointCloud


@dataclass(frozen=True)
class PerCameraCloud(WorkspaceCloud):
    source_frame: str = ""
    depth_mode: str = "native"
    frame_index: int = 0
    host_receive_timestamp_ns: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", dict(self.provenance))


@dataclass(frozen=True)
class RigBuildResult:
    per_camera_workspace_clouds: tuple[PerCameraCloud, ...]
    concatenated_workspace_cloud: WorkspacePointCloud
    timing_report_ms: dict[str, Any]
    per_camera_provenance: dict[str, dict[str, Any]]
    canonical_camera_order: tuple[str, ...]
    frame_match: RigFrameSet
