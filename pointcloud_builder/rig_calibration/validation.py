"""Candidate solution validation, including held-out target poses."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from pointcloud_builder.rig_calibration.artifact import solution_fingerprint
from pointcloud_builder.rig_calibration.config import RigCalibrationConfig
from pointcloud_builder.rig_calibration.initialization import (
    aggregate_transforms,
    estimate_camera_from_target,
)
from pointcloud_builder.rig_calibration.projection import project_target_points
from pointcloud_builder.rig_calibration.quality import reprojection_metrics
from pointcloud_builder.rig_calibration.se3 import compose
from pointcloud_builder.rig_calibration.types import (
    RigCalibrationObservations,
    RigCalibrationSolution,
)


def validate_rig_calibration_solution(
    solution: RigCalibrationSolution,
    data: RigCalibrationObservations,
    config: RigCalibrationConfig | None = None,
) -> dict[str, Any]:
    """Recompute solve residuals and optionally validate held-out target poses."""

    if config is not None and config.to_dict() != solution.config:
        raise ValueError("validation config must exactly match the candidate solution config")
    config = RigCalibrationConfig(**solution.config)
    _require_matching_provenance(solution, data)
    failed_quality = [
        item.observation_id
        for item in data.observations
        if item.quality.get("passed") is not True
    ]
    if failed_quality:
        raise ValueError(
            "validation observations require explicit passed quality: "
            + ", ".join(sorted(failed_quality))
        )
    solve_errors = _errors_for_known_poses(solution, data, split="solve")
    solve_metrics = reprojection_metrics(solve_errors)
    holdout = _validate_holdout(solution, data, config)
    holdout_pass = holdout["status"] in {"PASS", "NOT_RUN"}
    passed = bool(
        solution.passed
        and solve_metrics["p95_px"] is not None
        and solve_metrics["p95_px"] <= config.final_reprojection_p95_px
        and holdout_pass
    )
    return {
        "schema_version": "pointcloud-builder.rig-calibration-validation.v1",
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
        "candidate_only": True,
        "production_applied": False,
        "solution_fingerprint": solution_fingerprint(solution),
        "workspace_frame": solution.workspace_frame,
        "target_identity": solution.target_identity,
        "camera_bundle_hashes": solution.camera_bundle_hashes,
        "camera_identities": solution.camera_identities,
        "config": config.to_dict(),
        "solve_reprojection": solve_metrics,
        "holdout": holdout,
    }


def _validate_holdout(
    solution: RigCalibrationSolution,
    data: RigCalibrationObservations,
    config: RigCalibrationConfig,
) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for observation in data.observations:
        if observation.split == "holdout":
            grouped[observation.pose_id].append(observation)
    if not grouped:
        return {"status": "NOT_RUN", "pose_count": 0}
    too_few = sorted(
        observation.observation_id
        for observations in grouped.values()
        for observation in observations
        if len(observation.point_ids) < config.min_corners_per_observation
    )
    if too_few:
        return {
            "status": "FAIL",
            "pose_count": len(grouped),
            "failure_reason": "HOLDOUT_TOO_FEW_CORNERS",
            "failed_observation_ids": too_few,
            "minimum_corners_per_observation": config.min_corners_per_observation,
        }
    all_errors: list[float] = []
    per_pose: dict[str, dict[str, Any]] = {}
    for pose_id, observations in sorted(grouped.items()):
        camera_ids = {item.camera_id for item in observations}
        if len(camera_ids) < 2:
            return {
                "status": "FAIL",
                "pose_count": len(grouped),
                "failure_reason": "HOLDOUT_POSE_REQUIRES_MULTICAMERA_VISIBILITY",
                "failed_pose_id": pose_id,
            }
        candidates = []
        for observation in sorted(observations, key=lambda item: item.observation_id):
            T_camera_from_target, _rmse = estimate_camera_from_target(
                observation, data.projection_models[observation.camera_id]
            )
            candidates.append(
                compose(
                    solution.T_workspace_from_camera[observation.camera_id],
                    T_camera_from_target,
                )
            )
        T_workspace_from_target = aggregate_transforms(candidates)
        errors: list[float] = []
        for observation in observations:
            projected, in_front = project_target_points(
                observation.object_points_m,
                T_workspace_from_target,
                solution.T_workspace_from_camera[observation.camera_id],
                data.projection_models[observation.camera_id],
            )
            values = np.linalg.norm(projected - observation.image_points_px, axis=1)
            values[~in_front | ~np.isfinite(values)] = 1e4
            errors.extend(values.tolist())
        per_pose[pose_id] = reprojection_metrics(errors)
        all_errors.extend(errors)
    global_reprojection = reprojection_metrics(all_errors)
    passed = bool(
        global_reprojection["p95_px"] is not None
        and global_reprojection["p95_px"] <= config.final_reprojection_p95_px
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "pose_count": len(grouped),
        "global_reprojection": global_reprojection,
        "per_pose": per_pose,
        "p95_gate_px": config.final_reprojection_p95_px,
        "failure_reason": None if passed else "HOLDOUT_REPROJECTION_P95_EXCEEDS_GATE",
    }


def _require_matching_provenance(
    solution: RigCalibrationSolution,
    data: RigCalibrationObservations,
) -> None:
    expected_frames = {
        camera_id: model.frame for camera_id, model in data.projection_models.items()
    }
    checks = {
        "workspace_frame": solution.workspace_frame == data.workspace_frame,
        "camera_frames": solution.camera_frames == expected_frames,
        "target_identity": solution.target_identity == data.target_identity,
        "camera_bundle_hashes": (
            solution.camera_bundle_hashes == data.camera_bundle_hashes
        ),
        "camera_identities": solution.camera_identities == data.camera_identities,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("solution/observation provenance mismatch: " + ", ".join(failed))


def _errors_for_known_poses(
    solution: RigCalibrationSolution,
    data: RigCalibrationObservations,
    *,
    split: str,
) -> list[float]:
    errors: list[float] = []
    for observation in data.observations:
        if observation.split != split:
            continue
        projected, in_front = project_target_points(
            observation.object_points_m,
            solution.T_workspace_from_target[observation.pose_id],
            solution.T_workspace_from_camera[observation.camera_id],
            data.projection_models[observation.camera_id],
        )
        values = np.linalg.norm(projected - observation.image_points_px, axis=1)
        values[~in_front | ~np.isfinite(values)] = 1e4
        errors.extend(values.tolist())
    return errors
