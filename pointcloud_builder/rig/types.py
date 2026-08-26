"""Frame-explicit contracts for offline multi-camera orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pointcloud_builder.workspace.types import FramedPointCloud, WorkspacePointCloud


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
    match_sequence_index: int | None = None
    match_timestamp_ns: int | None = None
    per_camera_absolute_delta_ms: dict[str, float] = field(default_factory=dict)
    matching_policy: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        envelopes = dict(self.envelopes)
        deltas = dict(self.per_camera_delta_ms)
        absolute_deltas = dict(self.per_camera_absolute_delta_ms)
        if not absolute_deltas:
            absolute_deltas = {name: abs(value) for name, value in deltas.items()}
        match_timestamp_ns = self.match_timestamp_ns
        reference = envelopes.get(self.reference_camera)
        if match_timestamp_ns is None and reference is not None:
            match_timestamp_ns = reference.host_receive_timestamp_ns
        object.__setattr__(self, "envelopes", envelopes)
        object.__setattr__(self, "per_camera_delta_ms", deltas)
        object.__setattr__(self, "per_camera_absolute_delta_ms", absolute_deltas)
        object.__setattr__(self, "match_timestamp_ns", match_timestamp_ns)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def matched_camera_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.envelopes))


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
class PerCameraFramedCloud:
    camera_name: str
    cloud: FramedPointCloud


@dataclass(frozen=True)
class RigBuildResult:
    per_camera_camera_frame: tuple[PerCameraFramedCloud, ...]
    per_camera_workspace: tuple[PerCameraCloud, ...]
    concatenated: WorkspacePointCloud
    workspace_cropped: WorkspacePointCloud
    fused: WorkspacePointCloud
    sampled: WorkspacePointCloud
    fusion_provenance: Any | None
    timing_report_ms: dict[str, Any]
    per_camera_provenance: dict[str, dict[str, Any]]
    canonical_camera_order: tuple[str, ...]
    frame_match: RigFrameSet
    per_camera_stage_statistics: dict[str, dict[str, Any]] = field(default_factory=dict)
    processing_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def per_camera_workspace_clouds(self) -> tuple[PerCameraCloud, ...]:
        return self.per_camera_workspace

    @property
    def concatenated_workspace_cloud(self) -> WorkspacePointCloud:
        return self.workspace_cropped
