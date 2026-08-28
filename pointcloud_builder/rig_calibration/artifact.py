"""JSON serialization for versioned rig-calibration artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from pointcloud_builder.camera_model import CameraIntrinsics
from pointcloud_builder.rig_calibration.types import (
    RigCalibrationObservations,
    RigCalibrationSolution,
    RigTargetObservation,
)


def write_observations(
    observations: RigCalibrationObservations, path: str | Path
) -> None:
    _write_json(path, observations_to_dict(observations))


def load_observations(path: str | Path) -> RigCalibrationObservations:
    raw = _read_json(path)
    return RigCalibrationObservations(
        schema_version=str(raw["schema_version"]),
        target_identity=dict(raw["target_identity"]),
        camera_identities={
            str(key): dict(value) for key, value in raw["camera_identities"].items()
        },
        camera_bundle_hashes={
            str(key): str(value) for key, value in raw["camera_bundle_hashes"].items()
        },
        projection_models={
            str(key): _projection_from_dict(value)
            for key, value in raw["projection_models"].items()
        },
        workspace_frame=str(raw["workspace_frame"]),
        initial_camera_poses={
            str(key): np.asarray(value, dtype=np.float64)
            for key, value in raw.get("initial_camera_poses", {}).items()
        },
        observations=tuple(
            RigTargetObservation(
                observation_id=str(value["observation_id"]),
                camera_id=str(value["camera_id"]),
                pose_id=str(value["pose_id"]),
                point_ids=tuple(int(item) for item in value["point_ids"]),
                object_points_m=np.asarray(value["object_points_m"], dtype=np.float64),
                image_points_px=np.asarray(value["image_points_px"], dtype=np.float64),
                timestamp_ns=int(value["timestamp_ns"])
                if value.get("timestamp_ns") is not None
                else None,
                quality=dict(value.get("quality", {})),
                split=str(value.get("split", "solve")),
            )
            for value in raw["observations"]
        ),
    )


def observations_to_dict(data: RigCalibrationObservations) -> dict[str, Any]:
    return {
        "schema_version": data.schema_version,
        "workspace_frame": data.workspace_frame,
        "target_identity": data.target_identity,
        "camera_identities": {
            key: value for key, value in sorted(data.camera_identities.items())
        },
        "camera_bundle_hashes": dict(sorted(data.camera_bundle_hashes.items())),
        "projection_models": {
            key: _projection_to_dict(value)
            for key, value in sorted(data.projection_models.items())
        },
        "initial_camera_poses": {
            key: value.tolist() for key, value in sorted(data.initial_camera_poses.items())
        },
        "observations": [
            {
                "observation_id": item.observation_id,
                "camera_id": item.camera_id,
                "pose_id": item.pose_id,
                "point_ids": list(item.point_ids),
                "object_points_m": item.object_points_m.tolist(),
                "image_points_px": item.image_points_px.tolist(),
                "timestamp_ns": item.timestamp_ns,
                "quality": item.quality,
                "split": item.split,
            }
            for item in sorted(data.observations, key=lambda value: value.observation_id)
        ],
    }


def write_solution(solution: RigCalibrationSolution, path: str | Path) -> None:
    _write_json(path, solution_to_dict(solution))


def load_solution(path: str | Path) -> RigCalibrationSolution:
    raw = _read_json(path)
    return RigCalibrationSolution(
        schema_version=str(raw["schema_version"]),
        workspace_frame=str(raw["workspace_frame"]),
        anchor_pose_id=str(raw["anchor_pose_id"]),
        camera_frames={str(key): str(value) for key, value in raw["camera_frames"].items()},
        target_identity=dict(raw["target_identity"]),
        camera_bundle_hashes={
            str(key): str(value) for key, value in raw["camera_bundle_hashes"].items()
        },
        camera_identities={
            str(key): dict(value) for key, value in raw["camera_identities"].items()
        },
        T_workspace_from_camera=_matrix_mapping(raw["T_workspace_from_camera"]),
        T_workspace_from_target=_matrix_mapping(raw["T_workspace_from_target"]),
        initial_camera_poses=_matrix_mapping(raw["initial_camera_poses"]),
        camera_corrections=dict(raw["camera_corrections"]),
        quality_counts={str(key): int(value) for key, value in raw["quality_counts"].items()},
        reprojection=dict(raw["reprojection"]),
        per_camera_metrics=dict(raw["per_camera_metrics"]),
        per_pose_metrics=dict(raw["per_pose_metrics"]),
        pose_diversity=dict(raw["pose_diversity"]),
        graph=dict(raw["graph"]),
        optimizer=dict(raw["optimizer"]),
        observability=dict(raw["observability"]),
        validation=dict(raw["validation"]),
        config=dict(raw["config"]),
    )


def solution_to_dict(solution: RigCalibrationSolution) -> dict[str, Any]:
    return {
        "schema_version": solution.schema_version,
        "workspace_frame": solution.workspace_frame,
        "anchor_pose_id": solution.anchor_pose_id,
        "camera_frames": dict(sorted(solution.camera_frames.items())),
        "target_identity": solution.target_identity,
        "camera_bundle_hashes": dict(sorted(solution.camera_bundle_hashes.items())),
        "camera_identities": {
            key: value for key, value in sorted(solution.camera_identities.items())
        },
        "T_workspace_from_camera": {
            key: value.tolist()
            for key, value in sorted(solution.T_workspace_from_camera.items())
        },
        "T_workspace_from_target": {
            key: value.tolist()
            for key, value in sorted(solution.T_workspace_from_target.items())
        },
        "initial_camera_poses": {
            key: value.tolist()
            for key, value in sorted(solution.initial_camera_poses.items())
        },
        "camera_corrections": solution.camera_corrections,
        "quality_counts": solution.quality_counts,
        "reprojection": solution.reprojection,
        "per_camera_metrics": solution.per_camera_metrics,
        "per_pose_metrics": solution.per_pose_metrics,
        "pose_diversity": solution.pose_diversity,
        "graph": solution.graph,
        "optimizer": solution.optimizer,
        "observability": solution.observability,
        "validation": solution.validation,
        "config": solution.config,
    }


def solution_fingerprint(solution: RigCalibrationSolution) -> str:
    """Bind validation/export artifacts to one exact candidate solution."""

    canonical = json.dumps(
        solution_to_dict(solution), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _projection_to_dict(value: CameraIntrinsics) -> dict[str, Any]:
    return {
        "width": value.width,
        "height": value.height,
        "fx": value.fx,
        "fy": value.fy,
        "cx": value.cx,
        "cy": value.cy,
        "distortion_model": value.distortion_model,
        "distortion_coeffs": list(value.distortion_coeffs),
        "pixel_geometry": value.pixel_geometry,
        "frame": value.frame,
    }


def _projection_from_dict(value: dict[str, Any]) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=int(value["width"]),
        height=int(value["height"]),
        fx=float(value["fx"]),
        fy=float(value["fy"]),
        cx=float(value["cx"]),
        cy=float(value["cy"]),
        distortion_model=str(value.get("distortion_model", "none")),
        distortion_coeffs=tuple(float(item) for item in value.get("distortion_coeffs", ())),
        pixel_geometry=str(value.get("pixel_geometry", "rectified")),
        frame=str(value.get("frame", "")),
    )


def _matrix_mapping(value: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        str(key): np.asarray(matrix, dtype=np.float64)
        for key, matrix in value.items()
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("artifact root must be a JSON object")
    return raw


def _write_json(path: str | Path, value: dict[str, Any]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
