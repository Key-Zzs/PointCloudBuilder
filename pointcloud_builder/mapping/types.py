"""Dependency-light contracts shared by recording, mapping, and visualization."""

from __future__ import annotations

from dataclasses import dataclass, field
import numbers
from pathlib import Path
import re
from typing import Any, Literal

import numpy as np

from pointcloud_builder.camera_model import CameraIntrinsics

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _array(value: object, name: str, *, ndim: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    result = np.ascontiguousarray(value).copy()
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if result.dtype != np.bool_ and (
        not np.issubdtype(result.dtype, np.number) or not np.isfinite(result).all()
    ):
        raise ValueError(f"{name} must contain finite numeric values")
    result.setflags(write=False)
    return result


def _transform(value: object) -> np.ndarray:
    result = _array(value, "T_workspace_from_camera", ndim=2).astype(
        np.float64, copy=False
    )
    if result.shape != (4, 4) or not np.allclose(result[3], (0, 0, 0, 1), atol=1e-9):
        raise ValueError("T_workspace_from_camera must be homogeneous 4x4")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError("T_workspace_from_camera rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("T_workspace_from_camera rotation determinant must be +1")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RigDepthObservation:
    camera_name: str
    depth: np.ndarray
    depth_unit: Literal["raw_units", "meters"]
    depth_scale_m_per_unit: float
    valid_mask: np.ndarray
    intrinsics: CameraIntrinsics
    T_workspace_from_camera: np.ndarray
    timestamp_ns: int
    depth_source: Literal["native", "ffs_stereo"]
    source_frame: str
    workspace_frame: str
    bundle_identity: str
    provision_sha256: str
    distortion_model: str
    distortion_coeffs: tuple[float, ...]
    rectified: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.camera_name, str)
            or not self.camera_name.strip()
            or any(x in self.camera_name for x in "/\\")
        ):
            raise ValueError("camera_name must be a portable entity segment")
        depth = _array(self.depth, "depth", ndim=2)
        mask = _array(self.valid_mask, "valid_mask", ndim=2)
        if mask.dtype != np.bool_ or mask.shape != depth.shape:
            raise ValueError("valid_mask must be bool with the depth shape")
        if (self.intrinsics.height, self.intrinsics.width) != depth.shape:
            raise ValueError("depth shape must match intrinsics")
        if (
            isinstance(self.depth_scale_m_per_unit, bool)
            or not isinstance(self.depth_scale_m_per_unit, numbers.Real)
            or not np.isfinite(self.depth_scale_m_per_unit)
        ):
            raise ValueError("depth scale must be a finite number")
        if self.depth_unit == "raw_units":
            if depth.dtype != np.uint16 or self.depth_scale_m_per_unit <= 0:
                raise ValueError(
                    "raw depth must be uint16 with a positive device scale"
                )
        elif self.depth_unit == "meters":
            if not np.issubdtype(depth.dtype, np.floating):
                raise ValueError("metric depth must be floating point")
            if self.depth_scale_m_per_unit != 1.0:
                raise ValueError("metric depth must use depth scale 1")
        else:
            raise ValueError("unsupported depth_unit")
        expected_valid = np.isfinite(depth) & (depth > 0)
        if not np.array_equal(mask, expected_valid):
            raise ValueError("valid_mask must exactly identify finite positive depth")
        if (
            isinstance(self.timestamp_ns, bool)
            or not isinstance(self.timestamp_ns, numbers.Integral)
            or self.timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be a non-negative integer")
        if self.depth_source not in {"native", "ffs_stereo"}:
            raise ValueError("unsupported depth_source")
        if (
            not isinstance(self.source_frame, str)
            or not self.source_frame.strip()
            or not isinstance(self.workspace_frame, str)
            or not self.workspace_frame.strip()
        ):
            raise ValueError("depth source and workspace frames must be non-empty")
        if (
            not isinstance(self.bundle_identity, str)
            or not self.bundle_identity.strip()
        ):
            raise ValueError("bundle_identity must be non-empty")
        if (
            not isinstance(self.provision_sha256, str)
            or _SHA256.fullmatch(self.provision_sha256) is None
        ):
            raise ValueError("provision_sha256 must be a lowercase SHA-256")
        if (
            not isinstance(self.distortion_model, str)
            or not self.distortion_model.strip()
        ):
            raise ValueError("distortion_model must be non-empty")
        if not isinstance(self.rectified, bool):
            raise ValueError("rectified must be a boolean")
        if any(
            isinstance(value, bool) or not isinstance(value, numbers.Real)
            for value in self.distortion_coeffs
        ):
            raise ValueError("distortion coefficients must be numeric")
        coeffs = tuple(float(value) for value in self.distortion_coeffs)
        if not np.isfinite(coeffs).all():
            raise ValueError("distortion coefficients must be finite")
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(
            self, "T_workspace_from_camera", _transform(self.T_workspace_from_camera)
        )
        object.__setattr__(self, "distortion_coeffs", coeffs)

    @property
    def metric_depth(self) -> np.ndarray:
        result = self.depth.astype(np.float32) * float(self.depth_scale_m_per_unit)
        result[~self.valid_mask] = 0.0
        return result


@dataclass(frozen=True)
class RigDepthFrameSet:
    matched_set_index: int
    host_timestamp_ns: int
    maximum_skew_ms: float
    observations: tuple[RigDepthObservation, ...]
    schema_version: str = "pointcloud-builder.rig-depth-frame-set.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "pointcloud-builder.rig-depth-frame-set.v1":
            raise ValueError("unsupported rig depth frame-set schema")
        if (
            isinstance(self.matched_set_index, bool)
            or not isinstance(self.matched_set_index, numbers.Integral)
            or self.matched_set_index < 0
        ):
            raise ValueError("matched_set_index must be a non-negative integer")
        if (
            isinstance(self.host_timestamp_ns, bool)
            or not isinstance(self.host_timestamp_ns, numbers.Integral)
            or self.host_timestamp_ns < 0
            or isinstance(self.maximum_skew_ms, bool)
            or not isinstance(self.maximum_skew_ms, numbers.Real)
            or not np.isfinite(self.maximum_skew_ms)
            or self.maximum_skew_ms < 0
        ):
            raise ValueError("frame-set time/skew must be finite and non-negative")
        observations = tuple(self.observations)
        if any(not isinstance(item, RigDepthObservation) for item in observations):
            raise TypeError("depth observations must be RigDepthObservation values")
        names = [item.camera_name for item in observations]
        if not observations or names != sorted(names) or len(set(names)) != len(names):
            raise ValueError(
                "depth observations must be non-empty and canonically ordered"
            )
        if len({item.workspace_frame for item in observations}) != 1:
            raise ValueError("all depth observations must share one workspace frame")
        object.__setattr__(self, "observations", observations)


@dataclass(frozen=True)
class TsdfIntegrationResult:
    matched_set_index: int
    integrated_cameras: tuple[str, ...]
    active_block_count: int
    integration_ms: float
    skipped: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class MapExtraction:
    points: np.ndarray
    vertices: np.ndarray
    triangles: np.ndarray
    point_count: int
    vertex_count: int
    triangle_count: int
    extraction_ms: float

    def __post_init__(self) -> None:
        points = _array(self.points, "map points", ndim=2)
        vertices = _array(self.vertices, "mesh vertices", ndim=2)
        triangles = _array(self.triangles, "mesh triangles", ndim=2)
        if (
            points.shape[1:] != (3,)
            or vertices.shape[1:] != (3,)
            or triangles.shape[1:] != (3,)
        ):
            raise ValueError("map extraction arrays must be Nx3")
        if not np.issubdtype(triangles.dtype, np.integer):
            raise ValueError("mesh triangles must contain integer indices")
        if (self.point_count, self.vertex_count, self.triangle_count) != (
            len(points),
            len(vertices),
            len(triangles),
        ):
            raise ValueError("map extraction counts differ from arrays")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "triangles", triangles)


@dataclass(frozen=True)
class TsdfMapState:
    lifecycle: Literal["created", "integrating", "frozen", "closed"]
    workspace_frame: str
    integrated_frame_sets: int
    integrated_observations: int
    active_block_count: int
    last_matched_set_index: int | None
    map_revision: int


@dataclass(frozen=True)
class TsdfMapArtifact:
    root: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class DynamicMaskReport:
    camera_name: str
    total_pixels: int
    background_consistent_pixels: int
    transient_dynamic_pixels: int
    persistent_candidate_pixels: int
    newly_persistent_pixels: int
    integrated_pixels: int
    residual_median_m: float | None
    residual_p95_m: float | None
    metrics: dict[str, float] = field(default_factory=dict)
