from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
from camera_rig.calibration.intrinsic_health import IntrinsicHealthThresholds

from pointcloud_builder.rig_calibration.intrinsic_health import (
    evaluate_rig_intrinsic_health,
    intrinsic_health_fingerprint,
    load_rig_intrinsic_health,
    validate_rig_intrinsic_health,
    write_rig_intrinsic_health,
)
from tests.rig_calibration_synthetic import make_scene


def _bind_pose_plan(data):
    solve = sorted(
        {item.pose_id for item in data.observations if item.split == "solve"}
    )
    holdout = sorted(
        {item.pose_id for item in data.observations if item.split == "holdout"}
    )
    poses = sorted(set(solve) | set(holdout))
    return replace(
        data,
        bootstrap_qualifications={
            camera_id: {
                "schema_version": "camera-rig.calibration-authority.v1",
                "qualification_scope": "bootstrap_only",
                "production_authoritative": False,
                "qualification_state": "BOOTSTRAP_QUALIFIED",
                "qualification_fingerprint": "a" * 64,
                "target_metrology_sha256": "b" * 64,
                "metric_depth_receipt_sha256": "c" * 64,
            }
            for camera_id in data.camera_ids
        },
        pose_plan_sha256="d" * 64,
        pose_plan_summary={
            "pose_ids": poses,
            "solve_pose_ids": solve,
            "holdout_pose_ids": holdout,
            "capture_complete": True,
            "per_pose_camera_ids": {
                pose: sorted(
                    {
                        item.camera_id
                        for item in data.observations
                        if item.pose_id == pose
                    }
                )
                for pose in poses
            },
        },
    )


def _thresholds() -> IntrinsicHealthThresholds:
    return IntrinsicHealthThresholds(
        minimum_train_poses=8,
        minimum_holdout_poses=4,
        minimum_corners_per_pose=12,
        minimum_image_centroid_span_fraction=0.10,
    )


def test_rig_intrinsic_health_passes_and_round_trips_without_mutating_factory(
    tmp_path,
) -> None:
    data, _truth, _poses = make_scene(
        noise_px=0.05,
        holdout_pose_ids={"pose_8", "pose_9", "pose_10", "pose_11"},
    )
    data = _bind_pose_plan(data)
    before = {name: model for name, model in data.projection_models.items()}
    report = evaluate_rig_intrinsic_health(
        data, thresholds=_thresholds(), observations_sha256="e" * 64
    )
    output = tmp_path / "intrinsic-health.json"
    write_rig_intrinsic_health(report, output)

    assert load_rig_intrinsic_health(output) == report
    assert report["status"] == "PASS"
    assert data.projection_models == before


def test_missing_camera_pose_fails_closed() -> None:
    data, _truth, _poses = make_scene(
        noise_px=0.05,
        holdout_pose_ids={"pose_8", "pose_9", "pose_10", "pose_11"},
    )
    observations = tuple(
        item
        for item in data.observations
        if not (item.camera_id == "camera_b" and item.pose_id == "pose_11")
    )
    data = _bind_pose_plan(replace(data, observations=observations))
    report = evaluate_rig_intrinsic_health(
        data, thresholds=_thresholds(), observations_sha256="e" * 64
    )

    assert report["status"] == "FAIL"
    assert report["per_camera"]["camera_b"]["status"] == "INSUFFICIENT_EVIDENCE"


def test_connected_partial_visibility_uses_each_cameras_own_split() -> None:
    data, _truth, _poses = make_scene(
        noise_px=0.05,
        holdout_pose_ids={f"pose_{index}" for index in range(6, 12)},
    )
    observations = tuple(
        item
        for item in data.observations
        if not (item.camera_id == "camera_b" and item.pose_id == "pose_11")
    )
    data = _bind_pose_plan(replace(data, observations=observations))
    report = evaluate_rig_intrinsic_health(
        data,
        thresholds=IntrinsicHealthThresholds(
            minimum_train_poses=6,
            minimum_holdout_poses=4,
            minimum_corners_per_pose=12,
            minimum_image_centroid_span_fraction=0.05,
            minimum_distance_span_fraction=0.01,
            minimum_tilt_span_deg=1.0,
            maximum_centroid_design_condition_number=500.0,
        ),
        observations_sha256="e" * 64,
    )
    assert report["per_camera"]["camera_b"]["holdout_pose_ids"] == [
        "pose_10",
        "pose_6",
        "pose_7",
        "pose_8",
        "pose_9",
    ]
    assert report["per_camera"]["camera_b"]["status"] == "PASS"


def test_rehashed_intrinsic_report_rejects_phantom_pose_visibility() -> None:
    data, _truth, _poses = make_scene(
        noise_px=0.05,
        holdout_pose_ids={"pose_8", "pose_9", "pose_10", "pose_11"},
    )
    report = evaluate_rig_intrinsic_health(
        _bind_pose_plan(data), thresholds=_thresholds(), observations_sha256="e" * 64
    )
    forged = deepcopy(report)
    forged["pose_plan_summary"]["per_pose_camera_ids"]["pose_1"].remove("camera_a")
    forged["intrinsic_health_fingerprint"] = intrinsic_health_fingerprint(forged)

    with pytest.raises(ValueError, match="split differs from pose visibility"):
        validate_rig_intrinsic_health(forged)
