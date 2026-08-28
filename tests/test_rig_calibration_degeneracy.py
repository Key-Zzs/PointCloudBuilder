from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pointcloud_builder.rig_calibration.graph import CalibrationPreflightError
from pointcloud_builder.rig_calibration.solver import solve_rig_calibration
from tests.rig_calibration_synthetic import make_scene


def _pose(translation: tuple[float, float, float]) -> np.ndarray:
    value = np.eye(4)
    value[:3, 3] = translation
    return value


def test_tiny_central_image_region_fails_closed_with_coverage_reason() -> None:
    poses = {
        f"pose_{index}": _pose((0.002 * index, 0.001 * (index % 2), 0.002 * (index % 3)))
        for index in range(12)
    }
    data, _truth, _poses = make_scene(poses=poses, noise_px=0.0)
    with pytest.raises(CalibrationPreflightError) as captured:
        solve_rig_calibration(data)
    assert captured.value.code == "INSUFFICIENT_POSE_DIVERSITY"
    assert "insufficient_image_coverage" in captured.value.detail


def test_spatially_diverse_but_frontoparallel_poses_report_weak_observability() -> None:
    translations = [
        (0.0, 0.0, 0.0),
        (-0.14, -0.08, 0.05),
        (0.14, -0.08, -0.05),
        (-0.14, 0.08, 0.12),
        (0.14, 0.08, -0.12),
        (0.0, -0.14, 0.10),
        (0.0, 0.14, -0.10),
        (-0.10, 0.0, -0.14),
        (0.10, 0.0, 0.14),
        (-0.15, 0.04, 0.0),
        (0.15, -0.04, 0.0),
        (0.04, 0.15, 0.03),
    ]
    poses = {f"pose_{index}": _pose(value) for index, value in enumerate(translations)}
    data, _truth, _poses = make_scene(poses=poses, noise_px=0.0)
    with pytest.raises(CalibrationPreflightError) as captured:
        solve_rig_calibration(data)
    assert captured.value.code == "INSUFFICIENT_POSE_DIVERSITY"
    assert "nearly_frontoparallel_board_normals" in captured.value.detail


def test_inconsistent_point_id_geometry_fails_before_solving() -> None:
    data, _truth, _poses = make_scene(noise_px=0.0)
    observations = list(data.observations)
    index = next(
        index
        for index, value in enumerate(observations)
        if value.camera_id == "camera_b"
    )
    shifted = observations[index].object_points_m.copy()
    shifted[:, 0] += 0.01
    observations[index] = replace(observations[index], object_points_m=shifted)
    with pytest.raises(ValueError, match="inconsistent target geometry"):
        replace(data, observations=tuple(observations))


def test_missing_explicit_observation_quality_fails_closed() -> None:
    data, _truth, _poses = make_scene(noise_px=0.0)
    observations = list(data.observations)
    observations[0] = replace(observations[0], quality={})
    altered = replace(data, observations=tuple(observations))
    with pytest.raises(CalibrationPreflightError) as captured:
        solve_rig_calibration(altered)
    assert captured.value.code == "OBSERVATION_QUALITY_FAILED"
