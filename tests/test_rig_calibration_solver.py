from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pointcloud_builder.camera_model import CameraIntrinsics
from pointcloud_builder.rig_calibration.config import RigCalibrationConfig
from pointcloud_builder.rig_calibration.graph import CalibrationPreflightError
from pointcloud_builder.rig_calibration.solver import (
    _block_robust_residuals,
    solve_rig_calibration,
)
from tests.rig_calibration_synthetic import (
    diverse_target_poses,
    make_scene,
    shuffled,
)


def _assert_camera_accuracy(solution, expected, *, translation_mm=1.5, rotation_deg=0.15):
    for camera_id, ground_truth in expected.items():
        error = _independent_transform_error(
            solution.T_workspace_from_camera[camera_id], ground_truth
        )
        assert error["translation_mm"] < translation_mm, (camera_id, error)
        assert error["rotation_deg"] < rotation_deg, (camera_id, error)


def test_two_camera_noisy_joint_solver_recovers_perturbed_extrinsics() -> None:
    data, cameras, _poses = make_scene(noise_px=0.25)
    for camera_id, ground_truth in cameras.items():
        perturbation = _independent_transform_error(
            data.initial_camera_poses[camera_id], ground_truth
        )
        assert 5.0 <= perturbation["translation_mm"] <= 15.0
        assert 1.0 <= perturbation["rotation_deg"] <= 5.0
    solution = solve_rig_calibration(data)
    assert solution.passed
    assert solution.optimizer["parameterization"] == "rotation_vector_plus_translation"
    assert solution.optimizer["gauge_anchor"] == "T_workspace_from_target[pose_0]=I"
    assert solution.reprojection["final"]["p95_px"] < 0.75
    _assert_camera_accuracy(solution, cameras)


@pytest.mark.parametrize("loss", ["huber", "cauchy"])
def test_robust_objective_weights_equal_norm_corner_blocks_equally(loss: str) -> None:
    residuals = np.asarray([[3.0, 4.0], [5.0, 0.0]], dtype=np.float64)
    mapped = _block_robust_residuals(residuals, loss, 1.0).reshape(-1, 2)
    np.testing.assert_allclose(
        np.linalg.norm(mapped, axis=1),
        np.repeat(np.linalg.norm(mapped[0]), 2),
        atol=1e-12,
        rtol=0.0,
    )


def test_three_camera_full_and_partial_visibility_connected_graphs_pass() -> None:
    camera_ids = ("camera_a", "camera_b", "camera_c")
    full, full_truth, _poses = make_scene(camera_ids=camera_ids, noise_px=0.1)
    full_solution = solve_rig_calibration(full)
    assert full_solution.passed
    _assert_camera_accuracy(full_solution, full_truth, translation_mm=1.0, rotation_deg=0.1)

    partial_visibility = {
        "camera_a": {f"pose_{index}" for index in (0, 1, 2, 3, 4)},
        "camera_b": {f"pose_{index}" for index in (0, 2, 4, 5, 6)},
        "camera_c": {f"pose_{index}" for index in (1, 3, 5, 6, 7)},
    }
    partial, partial_truth, _poses = make_scene(
        camera_ids=camera_ids,
        poses=diverse_target_poses(8),
        visibility=partial_visibility,
        noise_px=0.1,
    )
    partial_solution = solve_rig_calibration(partial)
    assert partial_solution.passed
    assert partial_solution.graph["connected"] is True
    _assert_camera_accuracy(partial_solution, partial_truth, translation_mm=1.5, rotation_deg=0.15)


def test_disconnected_camera_graph_fails_closed() -> None:
    camera_ids = ("camera_a", "camera_b", "camera_c")
    visibility = {
        "camera_a": {f"pose_{index}" for index in range(6)},
        "camera_b": {f"pose_{index}" for index in range(6)},
        "camera_c": {"pose_8", "pose_9", "pose_10"},
    }
    data, _cameras, _poses = make_scene(camera_ids=camera_ids, visibility=visibility)
    with pytest.raises(CalibrationPreflightError) as captured:
        solve_rig_calibration(data)
    assert captured.value.code == "DISCONNECTED_CALIBRATION_GRAPH"


def test_four_camera_generic_smoke_is_not_hardcoded_to_three() -> None:
    camera_ids = ("camera_a", "camera_b", "camera_c", "camera_d")
    data, truth, _poses = make_scene(camera_ids=camera_ids, noise_px=0.1)
    solution = solve_rig_calibration(data)
    assert solution.passed
    assert set(solution.T_workspace_from_camera) == set(camera_ids)
    _assert_camera_accuracy(solution, truth, translation_mm=1.5, rotation_deg=0.15)


def test_camera_and_pose_input_order_invariance() -> None:
    data, _truth, _poses = make_scene(
        camera_ids=("camera_a", "camera_b", "camera_c"), noise_px=0.1
    )
    reference = solve_rig_calibration(data)
    for order in (
        ("camera_c", "camera_a", "camera_b"),
        ("camera_b", "camera_c", "camera_a"),
    ):
        permuted = replace(
            shuffled(data, 17),
            projection_models={key: data.projection_models[key] for key in order},
            camera_bundle_hashes={key: data.camera_bundle_hashes[key] for key in order},
            camera_identities={key: data.camera_identities[key] for key in order},
        )
        result = solve_rig_calibration(permuted)
        for camera_id in reference.T_workspace_from_camera:
            np.testing.assert_allclose(
                result.T_workspace_from_camera[camera_id],
                reference.T_workspace_from_camera[camera_id],
                atol=1e-9,
                rtol=0.0,
            )


def test_twenty_identical_poses_fail_pose_diversity_preflight() -> None:
    poses = {f"pose_{index}": np.eye(4) for index in range(20)}
    data, _truth, _poses = make_scene(poses=poses, noise_px=0.0)
    with pytest.raises(CalibrationPreflightError) as captured:
        solve_rig_calibration(data)
    assert captured.value.code == "INSUFFICIENT_POSE_DIVERSITY"


def test_robust_cauchy_loss_contains_two_corrupted_corners() -> None:
    data, truth, _poses = make_scene(
        noise_px=0.1,
        corrupt_observation_ids={"camera_b:pose_7"},
    )
    solution = solve_rig_calibration(
        data, RigCalibrationConfig(robust_loss="cauchy", loss_scale_px=1.0)
    )
    assert solution.passed
    _assert_camera_accuracy(solution, truth, translation_mm=2.0, rotation_deg=0.2)
    assert solution.per_pose_metrics["pose_7"]["max_px"] > 20.0


def test_nonzero_distortion_is_required_by_bundle_adjustment() -> None:
    data, truth, _poses = make_scene(noise_px=0.1, distortion=True)
    distortion_gate = RigCalibrationConfig(final_reprojection_p95_px=0.30)
    correct = solve_rig_calibration(data, distortion_gate)
    assert correct.passed
    _assert_camera_accuracy(correct, truth, translation_mm=1.5, rotation_deg=0.15)

    pinhole_models = {
        key: CameraIntrinsics(
            width=value.width,
            height=value.height,
            fx=value.fx,
            fy=value.fy,
            cx=value.cx,
            cy=value.cy,
            frame=value.frame,
        )
        for key, value in data.projection_models.items()
    }
    wrong = solve_rig_calibration(
        replace(data, projection_models=pinhole_models), distortion_gate
    )
    assert not wrong.passed
    assert wrong.reprojection["final"]["p95_px"] > 0.30


def test_odd_even_pose_split_has_gauge_invariant_extrinsic_stability() -> None:
    data, _truth, _poses = make_scene(
        camera_ids=("camera_a", "camera_b", "camera_c"), noise_px=0.1
    )
    solutions = {}
    for parity, anchor in ((0, "pose_0"), (1, "pose_1")):
        observations = tuple(
            item
            for item in data.observations
            if int(item.pose_id.split("_")[-1]) % 2 == parity
        )
        split = replace(data, observations=observations, initial_camera_poses={})
        solutions[parity] = solve_rig_calibration(
            split, RigCalibrationConfig(anchor_pose_id=anchor)
        )
        assert solutions[parity].passed
    for camera_id in ("camera_b", "camera_c"):
        even_relative = (
            np.linalg.inv(solutions[0].T_workspace_from_camera["camera_a"])
            @ solutions[0].T_workspace_from_camera[camera_id]
        )
        odd_relative = (
            np.linalg.inv(solutions[1].T_workspace_from_camera["camera_a"])
            @ solutions[1].T_workspace_from_camera[camera_id]
        )
        error = _independent_transform_error(odd_relative, even_relative)
        assert error["translation_mm"] <= 2.0
        assert error["rotation_deg"] <= 0.2


def _independent_transform_error(estimated, expected):
    delta = np.asarray(estimated, dtype=np.float64) @ np.linalg.inv(
        np.asarray(expected, dtype=np.float64)
    )
    cosine = float(np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    return {
        "translation_mm": float(1000.0 * np.linalg.norm(delta[:3, 3])),
        "rotation_deg": float(np.degrees(np.arccos(cosine))),
    }
