"""Pose-diversity, reprojection, and observability metrics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from pointcloud_builder.rig_calibration.config import RigCalibrationConfig
from pointcloud_builder.rig_calibration.types import (
    RigCalibrationObservations,
    RigTargetObservation,
)


def reprojection_metrics(errors_px: Iterable[float]) -> dict[str, Any]:
    values = np.asarray(tuple(errors_px), dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "count": 0,
            "rmse_px": None,
            "p50_px": None,
            "p95_px": None,
            "max_px": None,
        }
    return {
        "count": len(values),
        "rmse_px": float(np.sqrt(np.mean(values**2))),
        "p50_px": float(np.percentile(values, 50)),
        "p95_px": float(np.percentile(values, 95)),
        "max_px": float(np.max(values)),
    }


def assess_pose_diversity(
    data: RigCalibrationObservations,
    camera_poses: dict[str, np.ndarray],
    target_poses: dict[str, np.ndarray],
    config: RigCalibrationConfig,
) -> dict[str, Any]:
    solve_observations = tuple(item for item in data.observations if item.split == "solve")
    poses = [target_poses[key] for key in sorted(target_poses)]
    translations = np.asarray([pose[:3, 3] for pose in poses])
    translation_span = _maximum_pairwise_distance(translations)
    normals = np.asarray([pose[:3, 2] for pose in poses])
    normal_span = _maximum_angle_deg(normals)
    yaw = np.degrees(np.arctan2(normals[:, 0], normals[:, 2]))
    pitch = np.degrees(
        np.arctan2(-normals[:, 1], np.sqrt(normals[:, 0] ** 2 + normals[:, 2] ** 2))
    )
    depth_values: list[float] = []
    for observation in solve_observations:
        T_camera_from_workspace = np.linalg.inv(camera_poses[observation.camera_id])
        T_camera_from_target = T_camera_from_workspace @ target_poses[observation.pose_id]
        depth_values.append(float(T_camera_from_target[2, 3]))
    depth_span = float(np.ptp(depth_values)) if depth_values else 0.0
    coverage_by_camera = {
        camera_id: _image_coverage(
            [item for item in solve_observations if item.camera_id == camera_id],
            data.projection_models[camera_id].width,
            data.projection_models[camera_id].height,
        )
        for camera_id in data.camera_ids
    }
    minimum_coverage = min(coverage_by_camera.values(), default=0.0)
    failures: list[str] = []
    if len(poses) < config.min_pose_count:
        failures.append("too_few_target_poses")
    if translation_span < config.min_translation_span_m:
        failures.append("insufficient_translation_diversity")
    if depth_span < config.min_depth_span_m:
        failures.append("insufficient_depth_diversity")
    if normal_span < config.min_normal_span_deg:
        failures.append("nearly_frontoparallel_board_normals")
    if float(np.ptp(yaw)) < config.min_yaw_span_deg:
        failures.append("insufficient_yaw_diversity")
    if float(np.ptp(pitch)) < config.min_pitch_span_deg:
        failures.append("insufficient_pitch_diversity")
    if minimum_coverage < config.min_image_coverage_fraction:
        failures.append("insufficient_image_coverage")
    return {
        "passed": not failures,
        "status": "PASS" if not failures else "INSUFFICIENT_POSE_DIVERSITY",
        "failure_reasons": failures,
        "pose_count": len(poses),
        "translation_span_m": translation_span,
        "depth_span_m": depth_span,
        "board_normal_span_deg": normal_span,
        "yaw_span_deg": float(np.ptp(yaw)),
        "pitch_span_deg": float(np.ptp(pitch)),
        "image_coverage_fraction_by_camera": coverage_by_camera,
        "minimum_image_coverage_fraction": minimum_coverage,
        "thresholds": {
            "min_pose_count": config.min_pose_count,
            "min_translation_span_m": config.min_translation_span_m,
            "min_depth_span_m": config.min_depth_span_m,
            "min_normal_span_deg": config.min_normal_span_deg,
            "min_yaw_span_deg": config.min_yaw_span_deg,
            "min_pitch_span_deg": config.min_pitch_span_deg,
            "min_image_coverage_fraction": config.min_image_coverage_fraction,
        },
    }


def jacobian_observability(jacobian: np.ndarray | None) -> dict[str, Any]:
    if jacobian is None:
        return {"rank": 0, "condition_number": None, "singular_values": []}
    values = np.linalg.svd(np.asarray(jacobian, dtype=np.float64), compute_uv=False)
    tolerance = max(jacobian.shape) * np.finfo(np.float64).eps * values[0]
    nonzero = values[values > tolerance]
    condition = float(nonzero[0] / nonzero[-1]) if len(nonzero) else None
    return {
        "rank": len(nonzero),
        "parameter_count": int(jacobian.shape[1]),
        "condition_number": condition,
        "singular_values": values.tolist(),
    }


def _maximum_pairwise_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    differences = points[:, None, :] - points[None, :, :]
    return float(np.max(np.linalg.norm(differences, axis=-1)))


def _maximum_angle_deg(vectors: np.ndarray) -> float:
    if len(vectors) < 2:
        return 0.0
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    cosine = np.clip(normalized @ normalized.T, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)).max())


def _image_coverage(
    observations: list[RigTargetObservation], width: int, height: int
) -> float:
    if not observations:
        return 0.0
    points = np.vstack([item.image_points_px for item in observations])
    span = np.ptp(points, axis=0)
    return float((span[0] / width) * (span[1] / height))
