"""Planar-PnP edge estimates and graph-propagated SE(3) initialization."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch

from pointcloud_builder.camera_model import CameraIntrinsics
from pointcloud_builder.projection import project_points
from pointcloud_builder.rig_calibration.graph import CalibrationPreflightError
from pointcloud_builder.rig_calibration.se3 import compose, inverse, validate_transform
from pointcloud_builder.rig_calibration.types import (
    RigCalibrationObservations,
    RigTargetObservation,
)


@dataclass(frozen=True)
class InitializationResult:
    camera_poses: dict[str, np.ndarray]
    target_poses: dict[str, np.ndarray]
    edge_camera_from_target: dict[tuple[str, str], np.ndarray]
    edge_reprojection_rmse_px: dict[tuple[str, str], float]


def initialize_rig(
    data: RigCalibrationObservations,
    *,
    anchor_pose_id: str,
) -> InitializationResult:
    """Initialize every connected node while fixing the canonical pose to identity."""

    solve_observations = tuple(item for item in data.observations if item.split == "solve")
    grouped: dict[tuple[str, str], list[tuple[str, np.ndarray, float]]] = defaultdict(list)
    for observation in sorted(solve_observations, key=lambda item: item.observation_id):
        transform, rmse = estimate_camera_from_target(
            observation, data.projection_models[observation.camera_id]
        )
        grouped[(observation.camera_id, observation.pose_id)].append(
            (observation.observation_id, transform, rmse)
        )
    edge_transforms = {
        key: aggregate_transforms([item[1] for item in sorted(values)])
        for key, values in sorted(grouped.items())
    }
    edge_rmse = {
        key: float(np.median([item[2] for item in values]))
        for key, values in sorted(grouped.items())
    }
    camera_poses = {
        key: np.asarray(value, dtype=np.float64).copy()
        for key, value in sorted(data.initial_camera_poses.items())
    }
    target_poses: dict[str, np.ndarray] = {anchor_pose_id: np.eye(4, dtype=np.float64)}
    expected_cameras = set(data.camera_ids)
    expected_poses = {item.pose_id for item in solve_observations}

    for _ in range(len(expected_cameras) + len(expected_poses) + 1):
        changed = False
        camera_candidates: dict[str, list[np.ndarray]] = defaultdict(list)
        pose_candidates: dict[str, list[np.ndarray]] = defaultdict(list)
        for (camera_id, pose_id), T_camera_from_target in sorted(edge_transforms.items()):
            if camera_id in camera_poses:
                pose_candidates[pose_id].append(
                    compose(camera_poses[camera_id], T_camera_from_target)
                )
            if pose_id in target_poses:
                camera_candidates[camera_id].append(
                    compose(target_poses[pose_id], inverse(T_camera_from_target))
                )
        for camera_id, candidates in sorted(camera_candidates.items()):
            if camera_id not in camera_poses:
                camera_poses[camera_id] = aggregate_transforms(candidates)
                changed = True
        for pose_id, candidates in sorted(pose_candidates.items()):
            if pose_id == anchor_pose_id:
                continue
            candidate = aggregate_transforms(candidates)
            if pose_id not in target_poses:
                target_poses[pose_id] = candidate
                changed = True
            else:
                target_poses[pose_id] = aggregate_transforms(
                    [target_poses[pose_id], candidate]
                )
        if expected_cameras <= set(camera_poses) and expected_poses <= set(target_poses):
            break
        if not changed:
            break
    missing_cameras = sorted(expected_cameras - set(camera_poses))
    missing_poses = sorted(expected_poses - set(target_poses))
    if missing_cameras or missing_poses:
        raise CalibrationPreflightError(
            "DISCONNECTED_CALIBRATION_GRAPH",
            f"could not initialize cameras={missing_cameras}, poses={missing_poses}",
        )
    target_poses[anchor_pose_id] = np.eye(4, dtype=np.float64)
    return InitializationResult(
        camera_poses={
            key: validate_transform(value, name=f"initial camera {key}")
            for key, value in sorted(camera_poses.items())
        },
        target_poses={
            key: validate_transform(value, name=f"initial target {key}")
            for key, value in sorted(target_poses.items())
        },
        edge_camera_from_target=edge_transforms,
        edge_reprojection_rmse_px=edge_rmse,
    )


def estimate_camera_from_target(
    observation: RigTargetObservation,
    model: CameraIntrinsics,
) -> tuple[np.ndarray, float]:
    """Estimate ``T_camera_from_target`` then refine through PCB projection."""

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("rig calibration initialization requires OpenCV") from error
    try:
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
    except ImportError as error:
        raise RuntimeError("rig calibration requires the optional scipy dependency") from error
    camera_matrix = np.asarray(
        [[model.fx, 0.0, model.cx], [0.0, model.fy, model.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if model.distortion_model == "brown-conrady" and len(model.distortion_coeffs) == 5:
        distortion = np.asarray(model.distortion_coeffs, dtype=np.float64)
    else:
        distortion = np.zeros(5, dtype=np.float64)
    success, rotation_vector, translation = cv2.solvePnP(
        observation.object_points_m,
        observation.image_points_px,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise CalibrationPreflightError(
            "PNP_INITIALIZATION_FAILED", observation.observation_id
        )
    initial = np.concatenate((rotation_vector.reshape(3), translation.reshape(3)))

    def residual(parameters: np.ndarray) -> np.ndarray:
        rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
        points_camera = observation.object_points_m @ rotation.T + parameters[3:]
        projected = (
            project_points(torch.from_numpy(points_camera), model)
            .pixels_px.detach()
            .cpu()
            .numpy()
        )
        result = (projected - observation.image_points_px).reshape(-1)
        return np.nan_to_num(result, nan=1e4, posinf=1e4, neginf=-1e4)

    refined = least_squares(
        residual,
        initial,
        loss="huber",
        f_scale=1.0,
        max_nfev=300,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_rotvec(refined.x[:3]).as_matrix()
    matrix[:3, 3] = refined.x[3:]
    depths = observation.object_points_m @ matrix[:3, :3].T + matrix[:3, 3]
    if not np.all(depths[:, 2] > 0.0):
        raise CalibrationPreflightError(
            "PNP_INITIALIZATION_FAILED",
            f"{observation.observation_id} has target corners behind the camera",
        )
    rmse = float(np.sqrt(np.mean(residual(refined.x) ** 2)))
    return validate_transform(matrix, name="T_camera_from_target"), rmse


def aggregate_transforms(transforms: Iterable[np.ndarray]) -> np.ndarray:
    """Deterministic robust SE(3) aggregate for initialization proposals."""

    try:
        from scipy.spatial.transform import Rotation
    except ImportError as error:
        raise RuntimeError("rig calibration requires the optional scipy dependency") from error
    values = [np.asarray(value, dtype=np.float64) for value in transforms]
    if not values:
        raise ValueError("cannot aggregate an empty transform collection")
    translations = np.asarray([value[:3, 3] for value in values])
    quaternions = Rotation.from_matrix(
        np.asarray([value[:3, :3] for value in values])
    ).as_quat()
    reference = quaternions[0]
    quaternions = np.asarray(
        [quaternion if np.dot(quaternion, reference) >= 0.0 else -quaternion for quaternion in quaternions]
    )
    center = np.median(quaternions, axis=0)
    center /= np.linalg.norm(center)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(center).as_matrix()
    result[:3, 3] = np.median(translations, axis=0)
    return result
