"""Independent OpenCV oracle for rig-calibration acceptance tests."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import numpy as np

from pointcloud_builder.camera_model import CameraIntrinsics
from pointcloud_builder.rig_calibration.types import (
    RigCalibrationObservations,
    RigTargetObservation,
)


def target_points() -> np.ndarray:
    return np.asarray(
        [
            [(column - 3.0) * 0.03, (row - 2.0) * 0.03, 0.0]
            for row in range(5)
            for column in range(7)
        ],
        dtype=np.float64,
    )


def camera_poses(camera_ids: Iterable[str]) -> dict[str, np.ndarray]:
    positions = {
        "camera_a": np.asarray([0.0, 0.0, -0.82]),
        "camera_b": np.asarray([0.34, 0.02, -0.78]),
        "camera_c": np.asarray([-0.32, -0.03, -0.80]),
        "camera_d": np.asarray([0.02, 0.32, -0.86]),
    }
    return {
        camera_id: _look_at(positions[camera_id], np.zeros(3))
        for camera_id in camera_ids
    }


def diverse_target_poses(count: int = 12) -> dict[str, np.ndarray]:
    translations = [
        (0.0, 0.0, 0.0),
        (-0.14, -0.08, 0.04),
        (0.14, -0.07, -0.03),
        (-0.12, 0.09, 0.10),
        (0.13, 0.08, -0.09),
        (0.0, -0.12, 0.13),
        (0.0, 0.12, -0.11),
        (-0.08, 0.0, -0.14),
        (0.09, 0.0, 0.15),
        (-0.15, 0.04, -0.05),
        (0.15, -0.03, 0.06),
        (0.04, 0.14, 0.02),
    ]
    angles = [
        (0.0, 0.0),
        (12.0, -16.0),
        (-15.0, 13.0),
        (19.0, 8.0),
        (-18.0, -10.0),
        (8.0, 20.0),
        (-9.0, -21.0),
        (16.0, -7.0),
        (-13.0, 18.0),
        (21.0, 5.0),
        (-20.0, -6.0),
        (6.0, -19.0),
    ]
    result = {}
    for index in range(count):
        result[f"pose_{index}"] = _pose(translations[index], *angles[index])
    return result


def make_scene(
    *,
    camera_ids: tuple[str, ...] = ("camera_a", "camera_b"),
    poses: dict[str, np.ndarray] | None = None,
    visibility: dict[str, set[str]] | None = None,
    noise_px: float = 0.25,
    distortion: bool = False,
    initial_perturbation: bool = True,
    holdout_pose_ids: set[str] | None = None,
    corrupt_observation_ids: set[str] | None = None,
) -> tuple[RigCalibrationObservations, dict[str, np.ndarray], dict[str, np.ndarray]]:
    import cv2

    poses = poses or diverse_target_poses()
    ground_truth_cameras = camera_poses(camera_ids)
    coefficients = (0.085, -0.035, 0.0015, -0.0018, 0.009) if distortion else ()
    projection_models = {
        camera_id: CameraIntrinsics(
            width=960,
            height=720,
            fx=820.0 + 11.0 * index,
            fy=815.0 + 9.0 * index,
            cx=479.5,
            cy=359.5,
            distortion_model="brown-conrady" if distortion else "none",
            distortion_coeffs=coefficients,
            pixel_geometry="raw" if distortion else "rectified",
            frame=f"{camera_id}/color_optical",
        )
        for index, camera_id in enumerate(camera_ids)
    }
    points = target_points()
    ids = tuple(range(len(points)))
    rng = np.random.default_rng(20260828)
    observations = []
    for camera_id in sorted(camera_ids):
        visible = visibility.get(camera_id, set()) if visibility is not None else set(poses)
        model = projection_models[camera_id]
        camera_matrix = np.asarray(
            [[model.fx, 0.0, model.cx], [0.0, model.fy, model.cy], [0.0, 0.0, 1.0]]
        )
        distortion_coeffs = (
            np.asarray(model.distortion_coeffs) if model.distortion_coeffs else np.zeros(5)
        )
        for pose_id in sorted(visible):
            if pose_id not in poses:
                continue
            T_camera_from_target = np.linalg.inv(ground_truth_cameras[camera_id]) @ poses[
                pose_id
            ]
            rvec, _jacobian = cv2.Rodrigues(T_camera_from_target[:3, :3])
            image, _jacobian = cv2.projectPoints(
                points,
                rvec,
                T_camera_from_target[:3, 3],
                camera_matrix,
                distortion_coeffs,
            )
            pixels = image.reshape(-1, 2)
            if noise_px:
                pixels = pixels + rng.normal(0.0, noise_px, pixels.shape)
            observation_id = f"{camera_id}:{pose_id}"
            if observation_id in (corrupt_observation_ids or set()):
                pixels = pixels.copy()
                pixels[:2] += np.asarray([45.0, -35.0])
            observations.append(
                RigTargetObservation(
                    observation_id=observation_id,
                    camera_id=camera_id,
                    pose_id=pose_id,
                    point_ids=ids,
                    object_points_m=points,
                    image_points_px=pixels,
                    timestamp_ns=1_000_000_000 + len(observations),
                    quality={"passed": True, "synthetic": True},
                    split="holdout" if pose_id in (holdout_pose_ids or set()) else "solve",
                )
            )
    initial = {
        camera_id: _perturb_pose(pose, index)
        if initial_perturbation
        else pose.copy()
        for index, (camera_id, pose) in enumerate(sorted(ground_truth_cameras.items()))
    }
    data = RigCalibrationObservations(
        target_identity={"kind": "synthetic-grid", "point_count": len(points)},
        camera_bundle_hashes={camera_id: f"synthetic-{camera_id}" for camera_id in camera_ids},
        camera_identities={
            camera_id: {"camera_name": camera_id, "kind": "synthetic"}
            for camera_id in camera_ids
        },
        projection_models=projection_models,
        observations=tuple(observations),
        initial_camera_poses=initial,
    )
    return data, ground_truth_cameras, poses


def shuffled(data: RigCalibrationObservations, seed: int) -> RigCalibrationObservations:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data.observations))
    return replace(data, observations=tuple(data.observations[index] for index in indices))


def _look_at(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    z_axis = target - position
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(np.asarray([0.0, 1.0, 0.0]), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    result = np.eye(4)
    result[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    result[:3, 3] = position
    return result


def _pose(translation: tuple[float, float, float], yaw_deg: float, pitch_deg: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("yx", [yaw_deg, pitch_deg], degrees=True).as_matrix()
    result[:3, 3] = translation
    return result


def _perturb_pose(matrix: np.ndarray, index: int) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    translation_mm = (7.0 + 2.0 * index, -5.0 + index, 6.0 - index)
    rotation_deg = (1.5 + 0.8 * index, -1.5, 1.0)
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_euler("xyz", rotation_deg, degrees=True).as_matrix()
    delta[:3, 3] = np.asarray(translation_mm) / 1000.0
    return delta @ matrix
