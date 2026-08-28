"""Typed contracts for PCB rig-calibration observations and solutions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pointcloud_builder.camera_model import CameraIntrinsics
from pointcloud_builder.rig_calibration.se3 import validate_transform

OBSERVATIONS_SCHEMA_VERSION = "pointcloud-builder.rig-calibration-observations.v1"
SOLUTION_SCHEMA_VERSION = "pointcloud-builder.rig-calibration-solution.v1"
_SAFE_CAMERA_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _validate_camera_id(value: object) -> str:
    """Return one portable, non-traversing camera identifier."""

    if not isinstance(value, str) or not _SAFE_CAMERA_ID.fullmatch(value):
        raise ValueError(
            "camera_id must be a safe single path component containing only "
            "letters, digits, dot, underscore, or hyphen"
        )
    if value in {".", ".."}:
        raise ValueError("camera_id must not be '.' or '..'")
    return value


@dataclass(frozen=True)
class RigTargetObservation:
    """One camera's target corners for one stationary target pose."""

    observation_id: str
    camera_id: str
    pose_id: str
    point_ids: tuple[int, ...]
    object_points_m: np.ndarray
    image_points_px: np.ndarray
    timestamp_ns: int | None = None
    quality: dict[str, Any] = field(default_factory=dict)
    split: str = "solve"

    def __post_init__(self) -> None:
        for name in ("observation_id", "camera_id", "pose_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.split not in {"solve", "holdout"}:
            raise ValueError("observation split must be 'solve' or 'holdout'")
        _validate_camera_id(self.camera_id)
        object_points = np.asarray(self.object_points_m, dtype=np.float64).copy()
        image_points = np.asarray(self.image_points_px, dtype=np.float64).copy()
        if object_points.ndim != 2 or object_points.shape[1] != 3:
            raise ValueError("object_points_m must have shape [K,3]")
        if image_points.shape != (len(object_points), 2):
            raise ValueError("image_points_px must have shape [K,2]")
        if len(self.point_ids) != len(object_points):
            raise ValueError("point_ids must align with object and image points")
        if len(set(self.point_ids)) != len(self.point_ids):
            raise ValueError("point_ids must be unique within one observation")
        if not np.isfinite(object_points).all() or not np.isfinite(image_points).all():
            raise ValueError("target observation points must be finite")
        if self.timestamp_ns is not None and self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        object_points.setflags(write=False)
        image_points.setflags(write=False)
        object.__setattr__(self, "object_points_m", object_points)
        object.__setattr__(self, "image_points_px", image_points)
        object.__setattr__(self, "point_ids", tuple(int(value) for value in self.point_ids))
        object.__setattr__(self, "quality", dict(self.quality))


@dataclass(frozen=True)
class RigCalibrationObservations:
    """Versioned, image-independent multi-camera observation artifact."""

    target_identity: dict[str, Any]
    camera_bundle_hashes: dict[str, str]
    camera_identities: dict[str, dict[str, Any]]
    projection_models: dict[str, CameraIntrinsics]
    observations: tuple[RigTargetObservation, ...]
    workspace_frame: str = "workspace"
    initial_camera_poses: dict[str, np.ndarray] = field(default_factory=dict)
    schema_version: str = OBSERVATIONS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OBSERVATIONS_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {OBSERVATIONS_SCHEMA_VERSION!r}")
        if not self.workspace_frame.strip():
            raise ValueError("workspace_frame must be non-empty")
        models = dict(self.projection_models)
        if len(models) < 2:
            raise ValueError("rig calibration requires at least two cameras")
        for camera_id in models:
            _validate_camera_id(camera_id)
        missing_frames = sorted(
            camera_id for camera_id, model in models.items() if not model.frame.strip()
        )
        if missing_frames:
            raise ValueError(
                "rig calibration projection models require explicit frames: "
                f"{missing_frames}"
            )
        bundle_hashes = dict(self.camera_bundle_hashes)
        if set(bundle_hashes) != set(models):
            raise ValueError("camera_bundle_hashes and projection_models must have identical keys")
        if any(not str(value).strip() for value in bundle_hashes.values()):
            raise ValueError("camera_bundle_hashes must be non-empty")
        if not self.target_identity:
            raise ValueError("target_identity must be non-empty")
        identities = {
            str(camera_id): dict(identity)
            for camera_id, identity in self.camera_identities.items()
        }
        if set(identities) != set(models):
            raise ValueError("camera_identities and projection_models must have identical keys")
        for camera_id, identity in identities.items():
            if str(identity.get("camera_name", "")) != camera_id:
                raise ValueError(
                    f"camera identity for {camera_id!r} must carry the same camera_name"
                )
        observations = tuple(self.observations)
        observation_ids = [item.observation_id for item in observations]
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("observation_id values must be unique")
        unknown = sorted({item.camera_id for item in observations} - set(models))
        if unknown:
            raise ValueError(f"observations reference unknown cameras: {unknown}")
        canonical_points: dict[int, np.ndarray] = {}
        for observation in observations:
            for point_id, point in zip(
                observation.point_ids, observation.object_points_m, strict=True
            ):
                previous = canonical_points.setdefault(point_id, point)
                if not np.allclose(previous, point, atol=1e-12, rtol=0.0):
                    raise ValueError(
                        "inconsistent target geometry for point_id "
                        f"{point_id} in observation {observation.observation_id!r}"
                    )
        initial: dict[str, np.ndarray] = {}
        for camera_id, matrix in self.initial_camera_poses.items():
            if camera_id not in models:
                raise ValueError(f"initial pose references unknown camera {camera_id!r}")
            initial[camera_id] = validate_transform(
                matrix, name=f"initial_camera_poses[{camera_id!r}]"
            )
        object.__setattr__(self, "target_identity", dict(self.target_identity))
        object.__setattr__(self, "camera_bundle_hashes", bundle_hashes)
        object.__setattr__(self, "camera_identities", identities)
        object.__setattr__(self, "projection_models", models)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "initial_camera_poses", initial)

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.projection_models))

    @property
    def pose_ids(self) -> tuple[str, ...]:
        return tuple(sorted({item.pose_id for item in self.observations}))


@dataclass(frozen=True)
class RigCalibrationSolution:
    """Candidate-only joint calibration result; never an applied CameraBundle."""

    workspace_frame: str
    anchor_pose_id: str
    camera_frames: dict[str, str]
    target_identity: dict[str, Any]
    camera_bundle_hashes: dict[str, str]
    camera_identities: dict[str, dict[str, Any]]
    T_workspace_from_camera: dict[str, np.ndarray]
    T_workspace_from_target: dict[str, np.ndarray]
    initial_camera_poses: dict[str, np.ndarray]
    camera_corrections: dict[str, dict[str, Any]]
    quality_counts: dict[str, int]
    reprojection: dict[str, Any]
    per_camera_metrics: dict[str, dict[str, Any]]
    per_pose_metrics: dict[str, dict[str, Any]]
    pose_diversity: dict[str, Any]
    graph: dict[str, Any]
    optimizer: dict[str, Any]
    observability: dict[str, Any]
    validation: dict[str, Any]
    config: dict[str, Any]
    schema_version: str = SOLUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SOLUTION_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SOLUTION_SCHEMA_VERSION!r}")
        if not self.workspace_frame.strip() or not self.anchor_pose_id.strip():
            raise ValueError("workspace_frame and anchor_pose_id must be non-empty")
        camera_frames = {str(key): str(value) for key, value in self.camera_frames.items()}
        for camera_id in camera_frames:
            _validate_camera_id(camera_id)
        if set(camera_frames) != set(self.T_workspace_from_camera) or any(
            not value.strip() for value in camera_frames.values()
        ):
            raise ValueError("camera_frames must name every optimized camera frame")
        object.__setattr__(self, "camera_frames", camera_frames)
        bundle_hashes = {str(key): str(value) for key, value in self.camera_bundle_hashes.items()}
        identities = {
            str(camera_id): dict(identity)
            for camera_id, identity in self.camera_identities.items()
        }
        if set(bundle_hashes) != set(camera_frames) or set(identities) != set(camera_frames):
            raise ValueError(
                "solution provenance must identify every optimized camera frame"
            )
        if any(not value.strip() for value in bundle_hashes.values()):
            raise ValueError("solution camera_bundle_hashes must be non-empty")
        for camera_id, identity in identities.items():
            if str(identity.get("camera_name", "")) != camera_id:
                raise ValueError(
                    f"solution camera identity for {camera_id!r} must carry "
                    "the same camera_name"
                )
        object.__setattr__(self, "target_identity", dict(self.target_identity))
        object.__setattr__(self, "camera_bundle_hashes", bundle_hashes)
        object.__setattr__(self, "camera_identities", identities)
        for field_name in (
            "T_workspace_from_camera",
            "T_workspace_from_target",
            "initial_camera_poses",
        ):
            validated = {
                key: validate_transform(value, name=f"{field_name}[{key!r}]")
                for key, value in getattr(self, field_name).items()
            }
            object.__setattr__(self, field_name, validated)

    @property
    def passed(self) -> bool:
        return bool(self.validation.get("passed", False))
