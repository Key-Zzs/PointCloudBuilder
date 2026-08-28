#!/usr/bin/env python3
"""Run the frozen 2/3/4-camera synthetic rig-calibration acceptance matrix."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from pointcloud_builder.camera_model import CameraIntrinsics
from pointcloud_builder.local_paths import require_repo_local_path
from pointcloud_builder.rig_calibration.config import RigCalibrationConfig
from pointcloud_builder.rig_calibration.graph import CalibrationPreflightError
from pointcloud_builder.rig_calibration.solver import solve_rig_calibration
from tests.rig_calibration_synthetic import diverse_target_poses, make_scene, shuffled


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = require_repo_local_path(args.output, label="synthetic acceptance report")
    report: dict[str, object] = {
        "schema_version": "pointcloud-builder.rig-calibration-synthetic-acceptance.v1",
        "oracle": "OpenCV projectPoints; solver uses PCB projection path",
        "camera_pose_accuracy_gates": {"translation_mm": 2.0, "rotation_deg": 0.2},
        "target_pose_errors": "reported_not_acceptance_gated",
    }
    noise_results = {}
    for noise in (0.1, 0.25, 0.5):
        data, truth, _poses = make_scene(noise_px=noise)
        reprojection_gate = max(1.0, 3.0 * noise)
        solution = solve_rig_calibration(
            data,
            RigCalibrationConfig(final_reprojection_p95_px=reprojection_gate),
        )
        noise_results[str(noise)] = {
            **_solution_summary(solution, truth, _poses),
            "noise_sigma_px": noise,
            "frozen_reprojection_p95_gate_px": reprojection_gate,
        }
    report["two_camera_noise"] = noise_results

    camera_ids = ("camera_a", "camera_b", "camera_c")
    full, full_truth, _poses = make_scene(camera_ids=camera_ids, noise_px=0.25)
    full_solution = solve_rig_calibration(full)
    report["three_camera_full"] = _solution_summary(full_solution, full_truth, _poses)
    visibility = {
        "camera_a": {f"pose_{index}" for index in (0, 1, 2, 3, 4)},
        "camera_b": {f"pose_{index}" for index in (0, 2, 4, 5, 6)},
        "camera_c": {f"pose_{index}" for index in (1, 3, 5, 6, 7)},
    }
    partial, partial_truth, _poses = make_scene(
        camera_ids=camera_ids,
        poses=diverse_target_poses(8),
        visibility=visibility,
        noise_px=0.25,
    )
    partial_solution = solve_rig_calibration(partial)
    report["three_camera_partial"] = _solution_summary(
        partial_solution, partial_truth, _poses
    )

    four_ids = ("camera_a", "camera_b", "camera_c", "camera_d")
    four, four_truth, _poses = make_scene(camera_ids=four_ids, noise_px=0.25)
    four_solution = solve_rig_calibration(four)
    report["four_camera_smoke"] = _solution_summary(four_solution, four_truth, _poses)

    camera_order_differences = {}
    for order in (
        ("camera_c", "camera_a", "camera_b"),
        ("camera_b", "camera_c", "camera_a"),
    ):
        camera_permuted = replace(
            full,
            projection_models={key: full.projection_models[key] for key in order},
            camera_bundle_hashes={key: full.camera_bundle_hashes[key] for key in order},
            camera_identities={key: full.camera_identities[key] for key in order},
        )
        camera_permuted_solution = solve_rig_calibration(camera_permuted)
        camera_order_differences["-".join(order)] = max(
            np.max(
                np.abs(
                    full_solution.T_workspace_from_camera[camera_id]
                    - camera_permuted_solution.T_workspace_from_camera[camera_id]
                )
            )
            for camera_id in camera_ids
        )
    pose_permuted_solution = solve_rig_calibration(shuffled(full, 19))
    max_camera_order_difference = max(camera_order_differences.values())
    max_pose_order_difference = max(
        np.max(
            np.abs(
                full_solution.T_workspace_from_camera[camera_id]
                - pose_permuted_solution.T_workspace_from_camera[camera_id]
            )
        )
        for camera_id in camera_ids
    )
    report["camera_order_invariance"] = {
        "passed": bool(max_camera_order_difference <= 1e-9),
        "max_matrix_abs_difference": float(max_camera_order_difference),
        "per_order_max_matrix_abs_difference": {
            key: float(value) for key, value in camera_order_differences.items()
        },
    }
    report["pose_order_invariance"] = {
        "passed": bool(max_pose_order_difference <= 1e-9),
        "max_matrix_abs_difference": float(max_pose_order_difference),
    }

    report["odd_even_split_stability"] = _odd_even_split_stability(full)

    perturbations = [
        _independent_transform_error(
            four.initial_camera_poses[camera_id], four_truth[camera_id]
        )
        for camera_id in four_ids
    ]
    initial_translation = [value["translation_mm"] for value in perturbations]
    initial_rotation = [value["rotation_deg"] for value in perturbations]
    report["initial_extrinsic_perturbation"] = {
        "passed": bool(
            min(initial_translation) >= 5.0
            and max(initial_translation) <= 15.0
            and min(initial_rotation) >= 1.0
            and max(initial_rotation) <= 5.0
        ),
        "per_camera": dict(zip(four_ids, perturbations, strict=True)),
        "required_translation_mm": [5.0, 15.0],
        "required_rotation_deg": [1.0, 5.0],
    }

    disconnected_visibility = {
        "camera_a": {f"pose_{index}" for index in range(6)},
        "camera_b": {f"pose_{index}" for index in range(6)},
        "camera_c": {"pose_8", "pose_9", "pose_10"},
    }
    disconnected, _truth, _poses = make_scene(
        camera_ids=camera_ids, visibility=disconnected_visibility
    )
    report["disconnected_graph"] = _expected_preflight(
        disconnected, "DISCONNECTED_CALIBRATION_GRAPH"
    )
    repeated, _truth, _poses = make_scene(
        poses={f"pose_{index}": np.eye(4) for index in range(20)}, noise_px=0.0
    )
    report["same_pose_repetition"] = _expected_preflight(
        repeated, "INSUFFICIENT_POSE_DIVERSITY"
    )
    tiny_poses = {
        f"pose_{index}": _translation_pose(
            (0.002 * index, 0.001 * (index % 2), 0.002 * (index % 3))
        )
        for index in range(12)
    }
    tiny, _truth, _poses = make_scene(poses=tiny_poses, noise_px=0.0)
    report["tiny_image_coverage"] = _expected_preflight(
        tiny, "INSUFFICIENT_POSE_DIVERSITY", expected_detail="insufficient_image_coverage"
    )
    frontoparallel_translations = [
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
    frontoparallel, _truth, _poses = make_scene(
        poses={
            f"pose_{index}": _translation_pose(value)
            for index, value in enumerate(frontoparallel_translations)
        },
        noise_px=0.0,
    )
    report["frontoparallel"] = _expected_preflight(
        frontoparallel,
        "INSUFFICIENT_POSE_DIVERSITY",
        expected_detail="nearly_frontoparallel_board_normals",
    )

    outlier, outlier_truth, _poses = make_scene(
        noise_px=0.1, corrupt_observation_ids={"camera_b:pose_7"}
    )
    outlier_solution = solve_rig_calibration(
        outlier, RigCalibrationConfig(robust_loss="cauchy", loss_scale_px=1.0)
    )
    outlier_summary = _solution_summary(outlier_solution, outlier_truth, _poses)
    outlier_summary["corruption_detected"] = bool(
        outlier_solution.per_pose_metrics["pose_7"]["max_px"] > 20.0
    )
    outlier_summary["passed"] = bool(
        outlier_summary["passed"] and outlier_summary["corruption_detected"]
    )
    report["outlier"] = outlier_summary

    distorted, distorted_truth, _poses = make_scene(noise_px=0.1, distortion=True)
    distortion_config = RigCalibrationConfig(final_reprojection_p95_px=0.30)
    correct_distortion = solve_rig_calibration(distorted, distortion_config)
    pinhole_models = {
        key: CameraIntrinsics(
            value.width,
            value.height,
            value.fx,
            value.fy,
            value.cx,
            value.cy,
            frame=value.frame,
        )
        for key, value in distorted.projection_models.items()
    }
    wrong_distortion = solve_rig_calibration(
        replace(distorted, projection_models=pinhole_models), distortion_config
    )
    report["distortion"] = {
        "correct_model": _solution_summary(correct_distortion, distorted_truth, _poses),
        "pinhole_regression_passed": wrong_distortion.passed,
        "pinhole_regression_p95_px": wrong_distortion.reprojection["final"]["p95_px"],
        "passed": bool(correct_distortion.passed and not wrong_distortion.passed),
    }
    required_sections = [
        all(value["passed"] for value in noise_results.values()),
        report["three_camera_full"]["passed"],
        report["three_camera_partial"]["passed"],
        report["four_camera_smoke"]["passed"],
        report["camera_order_invariance"]["passed"],
        report["pose_order_invariance"]["passed"],
        report["odd_even_split_stability"]["passed"],
        report["initial_extrinsic_perturbation"]["passed"],
        report["disconnected_graph"]["passed"],
        report["same_pose_repetition"]["passed"],
        report["tiny_image_coverage"]["passed"],
        report["frontoparallel"]["passed"],
        report["outlier"]["passed"],
        report["distortion"]["passed"],
    ]
    report["N_CAMERA_IMPLEMENTATION"] = "PASS" if all(required_sections) else "FAIL"
    report["REAL_CAMERA_C_AVAILABLE"] = "NO"
    report["REAL_3_CAMERA_VALIDATION"] = "DEFERRED"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"N_CAMERA_IMPLEMENTATION={report['N_CAMERA_IMPLEMENTATION']}")
    return 0 if report["N_CAMERA_IMPLEMENTATION"] == "PASS" else 1


def _solution_summary(solution, truth, target_truth):
    errors = {
        camera_id: _independent_transform_error(
            solution.T_workspace_from_camera[camera_id], expected
        )
        for camera_id, expected in truth.items()
    }
    maximum_translation = max(value["translation_mm"] for value in errors.values())
    maximum_rotation = max(value["rotation_deg"] for value in errors.values())
    target_errors = {
        pose_id: _independent_transform_error(
            solution.T_workspace_from_target[pose_id], target_truth[pose_id]
        )
        for pose_id in solution.T_workspace_from_target
    }
    return {
        "passed": bool(
            solution.passed and maximum_translation <= 2.0 and maximum_rotation <= 0.2
        ),
        "solution_passed": solution.passed,
        "camera_pose_errors": errors,
        "maximum_translation_mm": maximum_translation,
        "maximum_rotation_deg": maximum_rotation,
        "maximum_target_translation_mm": max(
            value["translation_mm"] for value in target_errors.values()
        ),
        "maximum_target_rotation_deg": max(
            value["rotation_deg"] for value in target_errors.values()
        ),
        "reprojection": solution.reprojection["final"],
        "initial_camera_corrections": solution.camera_corrections,
        "optimizer": {
            "success": solution.optimizer["success"],
            "nfev": solution.optimizer["nfev"],
            "robust_loss": solution.optimizer["robust_loss"],
        },
        "condition_number": solution.observability["condition_number"],
    }


def _odd_even_split_stability(data):
    solutions = {}
    for parity, anchor in ((0, "pose_0"), (1, "pose_1")):
        selected = tuple(
            item
            for item in data.observations
            if int(item.pose_id.split("_")[-1]) % 2 == parity
        )
        split_data = replace(data, observations=selected, initial_camera_poses={})
        solutions[parity] = solve_rig_calibration(
            split_data, RigCalibrationConfig(anchor_pose_id=anchor)
        )
    anchor_camera = data.camera_ids[0]
    errors = {}
    for camera_id in data.camera_ids[1:]:
        even_relative = (
            np.linalg.inv(solutions[0].T_workspace_from_camera[anchor_camera])
            @ solutions[0].T_workspace_from_camera[camera_id]
        )
        odd_relative = (
            np.linalg.inv(solutions[1].T_workspace_from_camera[anchor_camera])
            @ solutions[1].T_workspace_from_camera[camera_id]
        )
        errors[camera_id] = _independent_transform_error(
            odd_relative, even_relative
        )
    return {
        "passed": bool(
            all(solution.passed for solution in solutions.values())
            and all(
                value["translation_mm"] <= 2.0
                and value["rotation_deg"] <= 0.2
                for value in errors.values()
            )
        ),
        "comparison": "gauge_invariant_relative_camera_transforms",
        "even_solution_passed": solutions[0].passed,
        "odd_solution_passed": solutions[1].passed,
        "per_camera_relative_error": errors,
        "translation_gate_mm": 2.0,
        "rotation_gate_deg": 0.2,
    }


def _independent_transform_error(estimated, expected):
    delta = np.asarray(estimated, dtype=np.float64) @ np.linalg.inv(
        np.asarray(expected, dtype=np.float64)
    )
    cosine = float(np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    return {
        "translation_mm": float(1000.0 * np.linalg.norm(delta[:3, 3])),
        "rotation_deg": float(np.degrees(np.arccos(cosine))),
    }


def _expected_preflight(data, code, expected_detail=None):
    try:
        solve_rig_calibration(data)
    except CalibrationPreflightError as error:
        return {
            "passed": error.code == code
            and (expected_detail is None or expected_detail in error.detail),
            "observed_code": error.code,
            "detail": error.detail,
        }
    return {"passed": False, "observed_code": "UNEXPECTED_SOLVER_PASS"}


def _translation_pose(translation):
    value = np.eye(4)
    value[:3, 3] = translation
    return value


if __name__ == "__main__":
    raise SystemExit(main())
