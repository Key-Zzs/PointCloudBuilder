from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from pointcloud_builder.integrations.camera_rig.types import FrameExplicitTransform
from pointcloud_builder.rig_calibration.artifact import (
    load_observations,
    load_solution,
    write_observations,
    write_solution,
)
from pointcloud_builder.rig_calibration.config import RigCalibrationConfig
from pointcloud_builder.rig_calibration.diagnostics import (
    candidate_diagnostic_contract,
    candidate_T_workspace_from_geometry_source,
)
from pointcloud_builder.rig_calibration.export import export_fixed_mount_candidates
from pointcloud_builder.rig_calibration.solver import solve_rig_calibration
from pointcloud_builder.rig_calibration.validation import (
    validate_rig_calibration_solution,
)
from tests.rig_calibration_synthetic import make_scene


def test_versioned_observation_solution_and_candidate_export_round_trip(tmp_path) -> None:
    data, _truth, _poses = make_scene(noise_px=0.1)
    observation_path = tmp_path / "observations.json"
    write_observations(data, observation_path)
    loaded_data = load_observations(observation_path)
    assert loaded_data.schema_version == "pointcloud-builder.rig-calibration-observations.v1"
    assert loaded_data.camera_ids == data.camera_ids
    assert loaded_data.projection_models["camera_a"].frame.endswith("color_optical")

    solution = solve_rig_calibration(loaded_data)
    solution_path = tmp_path / "solution.json"
    write_solution(solution, solution_path)
    loaded_solution = load_solution(solution_path)
    assert loaded_solution.passed
    assert loaded_solution.camera_frames["camera_a"] == "camera_a/color_optical"
    np.testing.assert_allclose(
        loaded_solution.T_workspace_from_camera["camera_b"],
        solution.T_workspace_from_camera["camera_b"],
    )

    validation = validate_rig_calibration_solution(loaded_solution, loaded_data)
    candidates = export_fixed_mount_candidates(
        loaded_solution,
        tmp_path / "candidates",
        validation_report=validation,
    )
    assert len(candidates) == 2
    candidate = json.loads(candidates[0].read_text())
    assert candidate["transform_name"] == "T_workspace_from_camera"
    assert candidate["source_frame"].endswith("color_optical")
    assert candidate["camera_bundle_sha256"] == "synthetic-camera_a"
    assert candidate["validation"]["production_applied"] is False


def test_multicamera_holdout_pose_validation_passes() -> None:
    holdout = {"pose_9", "pose_11"}
    data, _truth, _poses = make_scene(noise_px=0.1, holdout_pose_ids=holdout)
    solution = solve_rig_calibration(data)
    report = validate_rig_calibration_solution(solution, data)
    assert report["passed"]
    assert report["holdout"]["status"] == "PASS"
    assert report["holdout"]["pose_count"] == 2
    assert report["holdout"]["global_reprojection"]["p95_px"] < 1.0


def test_holdout_nested_status_fails_with_bad_multicamera_reprojection() -> None:
    data, _truth, _poses = make_scene(
        noise_px=0.1, holdout_pose_ids={"pose_9", "pose_11"}
    )
    solution = solve_rig_calibration(data)
    observations = tuple(
        replace(item, image_points_px=item.image_points_px + 20.0)
        if item.observation_id == "camera_b:pose_9"
        else item
        for item in data.observations
    )
    report = validate_rig_calibration_solution(
        solution, replace(data, observations=observations)
    )
    assert report["passed"] is False
    assert report["holdout"]["status"] == "FAIL"
    assert report["holdout"]["failure_reason"] == (
        "HOLDOUT_REPROJECTION_P95_EXCEEDS_GATE"
    )


def test_single_camera_holdout_pose_is_not_self_validating() -> None:
    data, _truth, _poses = make_scene(noise_px=0.1, holdout_pose_ids={"pose_9"})
    solution = solve_rig_calibration(data)
    observations = tuple(
        item
        for item in data.observations
        if not (item.pose_id == "pose_9" and item.camera_id == "camera_b")
    )
    report = validate_rig_calibration_solution(
        solution, replace(data, observations=observations)
    )
    assert report["passed"] is False
    assert report["holdout"]["failure_reason"] == (
        "HOLDOUT_POSE_REQUIRES_MULTICAMERA_VISIBILITY"
    )


def test_holdout_observation_respects_minimum_corner_gate() -> None:
    data, _truth, _poses = make_scene(noise_px=0.1, holdout_pose_ids={"pose_9"})
    solution = solve_rig_calibration(data)
    observations = tuple(
        replace(
            item,
            point_ids=item.point_ids[:4],
            object_points_m=item.object_points_m[:4],
            image_points_px=item.image_points_px[:4],
        )
        if item.observation_id == "camera_a:pose_9"
        else item
        for item in data.observations
    )
    report = validate_rig_calibration_solution(
        solution, replace(data, observations=observations)
    )
    assert report["passed"] is False
    assert report["holdout"]["failure_reason"] == "HOLDOUT_TOO_FEW_CORNERS"


def test_candidate_diagnostic_adapter_is_frame_explicit_and_never_applied() -> None:
    data, _truth, _poses = make_scene(noise_px=0.1)
    solution = solve_rig_calibration(data)
    T_color_from_ir_left = np.eye(4, dtype=np.float64)
    T_color_from_ir_left[0, 3] = 0.025
    internal_transform = FrameExplicitTransform(
        source_frame="camera_a/ir_left_optical",
        target_frame="camera_a/color_optical",
        matrix=T_color_from_ir_left,
    )

    result = candidate_T_workspace_from_geometry_source(
        solution,
        "camera_a",
        geometry_source_frame="camera_a/ir_left_optical",
        internal_transform=internal_transform,
    )
    expected = solution.T_workspace_from_camera["camera_a"] @ T_color_from_ir_left
    np.testing.assert_allclose(result, expected)

    contract = candidate_diagnostic_contract(solution)
    assert contract["candidate_only"] is True
    assert contract["production_applied"] is False
    assert len(contract["solution_fingerprint"]) == 64
    assert contract["target_identity"] == solution.target_identity
    assert contract["per_camera"]["camera_a"]["camera_bundle_sha256"] == (
        "synthetic-camera_a"
    )
    assert contract["per_camera"]["camera_a"]["camera_identity"] == (
        solution.camera_identities["camera_a"]
    )
    assert contract["per_camera"]["camera_a"]["source_frame"] == (
        "camera_a/color_optical"
    )

    with np.testing.assert_raises_regex(ValueError, "projection frame mismatch"):
        candidate_T_workspace_from_geometry_source(
            solution,
            "camera_a",
            geometry_source_frame="camera_a/ir_left_optical",
            internal_transform=FrameExplicitTransform(
                source_frame="camera_a/ir_left_optical",
                target_frame="camera_a/ir_left_optical",
                matrix=np.eye(4),
            ),
        )

    with np.testing.assert_raises_regex(ValueError, "geometry source frame mismatch"):
        candidate_T_workspace_from_geometry_source(
            solution,
            "camera_a",
            geometry_source_frame="camera_a/ir_left_optical",
            internal_transform=FrameExplicitTransform(
                source_frame="camera_a/depth_optical",
                target_frame="camera_a/color_optical",
                matrix=np.eye(4),
            ),
        )


def test_camera_ids_are_safe_single_path_components() -> None:
    data, _truth, _poses = make_scene(noise_px=0.1)
    for unsafe in ("../escape", "camera/escape", "camera\\escape", ".", ".."):
        models = dict(data.projection_models)
        model = models.pop("camera_a")
        models[unsafe] = model
        with np.testing.assert_raises_regex(ValueError, "safe single path component"):
            replace(data, projection_models=models)


def test_export_requires_validation_bound_to_exact_solution(tmp_path) -> None:
    data, _truth, _poses = make_scene(noise_px=0.1)
    solution = solve_rig_calibration(data)
    with np.testing.assert_raises_regex(ValueError, "explicitly passed"):
        export_fixed_mount_candidates(
            solution,
            tmp_path / "missing-validation",
            validation_report={"passed": False},
        )
    with np.testing.assert_raises_regex(ValueError, "does not bind"):
        export_fixed_mount_candidates(
            solution,
            tmp_path / "wrong-validation",
            validation_report={"passed": True, "solution_fingerprint": "wrong"},
        )


def test_validation_cannot_relax_solution_acceptance_config() -> None:
    data, _truth, _poses = make_scene(
        noise_px=0.1, holdout_pose_ids={"pose_9", "pose_11"}
    )
    solution = solve_rig_calibration(data)
    with np.testing.assert_raises_regex(ValueError, "exactly match"):
        validate_rig_calibration_solution(
            solution,
            data,
            RigCalibrationConfig(final_reprojection_p95_px=100.0),
        )
