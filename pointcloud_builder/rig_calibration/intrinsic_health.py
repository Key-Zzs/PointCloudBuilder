"""Rig-wide orchestration of immutable factory-intrinsic health diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from camera_rig.calibration.intrinsic_health import (
    IntrinsicHealthObservation,
    IntrinsicHealthThresholds,
    evaluate_intrinsic_health,
    validate_intrinsic_health_report,
)
from camera_rig.core.intrinsics import CameraIntrinsics as CameraRigIntrinsics

from pointcloud_builder.rig_calibration.types import RigCalibrationObservations

INTRINSIC_HEALTH_SCHEMA_VERSION = "pointcloud-builder.intrinsic-health.v1"


def evaluate_rig_intrinsic_health(
    data: RigCalibrationObservations,
    *,
    observations_sha256: str,
    thresholds: IntrinsicHealthThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate every factory K/D using solve poses for refit and frozen holdout poses."""

    train_pose_ids = tuple(
        sorted({item.pose_id for item in data.observations if item.split == "solve"})
    )
    holdout_pose_ids = tuple(
        sorted({item.pose_id for item in data.observations if item.split == "holdout"})
    )
    target_digest = canonical_json_sha256(data.target_identity)
    per_camera: dict[str, dict[str, Any]] = {}
    for camera_id in data.camera_ids:
        available = {
            item.pose_id for item in data.observations if item.camera_id == camera_id
        }
        camera_train_pose_ids = tuple(
            pose for pose in train_pose_ids if pose in available
        )
        camera_holdout_pose_ids = tuple(
            pose for pose in holdout_pose_ids if pose in available
        )
        camera_observations = tuple(
            IntrinsicHealthObservation(
                pose_id=item.pose_id,
                object_points_m=item.object_points_m,
                image_points_px=item.image_points_px,
            )
            for item in data.observations
            if item.camera_id == camera_id
        )
        per_camera[camera_id] = evaluate_intrinsic_health(
            camera_observations,
            _to_camera_rig_intrinsics(data.projection_models[camera_id]),
            train_pose_ids=camera_train_pose_ids,
            holdout_pose_ids=camera_holdout_pose_ids,
            thresholds=thresholds,
            camera_identity_sha256=canonical_json_sha256(
                data.camera_identities[camera_id]
            ),
            target_identity_sha256=target_digest,
            provenance={
                "camera_id": camera_id,
                "camera_bundle_sha256": data.camera_bundle_hashes[camera_id],
                "bootstrap_qualification": data.bootstrap_qualifications.get(camera_id),
                "split_authority": "frozen_observation_artifact",
                "observations_sha256": observations_sha256,
                "pose_plan_sha256": data.pose_plan_sha256,
            },
        )
    status = (
        "PASS"
        if per_camera
        and all(report.get("status") == "PASS" for report in per_camera.values())
        else "FAIL"
    )
    report: dict[str, Any] = {
        "schema_version": INTRINSIC_HEALTH_SCHEMA_VERSION,
        "status": status,
        "passed": status == "PASS",
        "factory_intrinsics_immutable": True,
        "camera_set": list(data.camera_ids),
        "train_pose_ids": list(train_pose_ids),
        "holdout_pose_ids": list(holdout_pose_ids),
        "target_identity": data.target_identity,
        "target_identity_sha256": target_digest,
        "camera_bundle_hashes": data.camera_bundle_hashes,
        "camera_identities": data.camera_identities,
        "bootstrap_qualifications": data.bootstrap_qualifications,
        "observations_sha256": observations_sha256,
        "pose_plan_sha256": data.pose_plan_sha256,
        "pose_plan_summary": data.pose_plan_summary,
        "per_camera": per_camera,
    }
    report["intrinsic_health_fingerprint"] = intrinsic_health_fingerprint(report)
    return report


def intrinsic_health_fingerprint(report: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in report.items()
        if key != "intrinsic_health_fingerprint"
    }
    return canonical_json_sha256(payload)


def validate_rig_intrinsic_health(report: dict[str, Any]) -> dict[str, Any]:
    """Strictly validate a rig-wide report before production promotion."""

    required = {
        "schema_version",
        "status",
        "passed",
        "factory_intrinsics_immutable",
        "camera_set",
        "train_pose_ids",
        "holdout_pose_ids",
        "target_identity",
        "target_identity_sha256",
        "camera_bundle_hashes",
        "camera_identities",
        "bootstrap_qualifications",
        "observations_sha256",
        "pose_plan_sha256",
        "pose_plan_summary",
        "per_camera",
        "intrinsic_health_fingerprint",
    }
    if set(report) != required:
        raise ValueError("intrinsic-health report has missing or unknown fields")
    if report.get("schema_version") != INTRINSIC_HEALTH_SCHEMA_VERSION:
        raise ValueError("unsupported intrinsic-health report schema")
    cameras = report.get("camera_set")
    per_camera = report.get("per_camera")
    if (
        not isinstance(cameras, list)
        or not cameras
        or cameras != sorted(cameras)
        or len(set(cameras)) != len(cameras)
        or not isinstance(per_camera, dict)
        or set(per_camera) != set(cameras)
    ):
        raise ValueError("intrinsic-health camera set is invalid")
    train = report.get("train_pose_ids")
    holdout = report.get("holdout_pose_ids")
    if (
        not isinstance(train, list)
        or not train
        or not isinstance(holdout, list)
        or not holdout
    ):
        raise ValueError("intrinsic-health train and holdout splits must be non-empty")
    if set(train) & set(holdout):
        raise ValueError("intrinsic-health train and holdout splits overlap")
    for name in ("observations_sha256", "pose_plan_sha256"):
        value = report.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"intrinsic-health {name} must be a lowercase SHA-256")
    pose_plan = report.get("pose_plan_summary")
    if not isinstance(pose_plan, dict):
        raise TypeError("intrinsic-health pose-plan summary must be an object")
    per_pose_visibility = pose_plan.get("per_pose_camera_ids")
    if (
        not isinstance(pose_plan.get("solve_pose_ids"), list)
        or set(pose_plan["solve_pose_ids"]) != set(train)
        or len(pose_plan["solve_pose_ids"]) != len(train)
        or not isinstance(pose_plan.get("holdout_pose_ids"), list)
        or set(pose_plan["holdout_pose_ids"]) != set(holdout)
        or len(pose_plan["holdout_pose_ids"]) != len(holdout)
        or not isinstance(per_pose_visibility, dict)
        or set(per_pose_visibility) != set(train) | set(holdout)
        or any(
            not isinstance(value, list)
            or len(set(value)) != len(value)
            or not set(value) <= set(cameras)
            for value in per_pose_visibility.values()
        )
    ):
        raise ValueError("intrinsic-health pose-plan splits or visibility differ")
    hashes = report.get("camera_bundle_hashes")
    identities = report.get("camera_identities")
    bootstrap = report.get("bootstrap_qualifications")
    if not all(
        isinstance(value, dict) and set(value) == set(cameras)
        for value in (
            hashes,
            identities,
            bootstrap,
        )
    ):
        raise ValueError("intrinsic-health camera provenance sets differ")
    for camera_id in cameras:
        camera_report = per_camera[camera_id]
        if not isinstance(camera_report, dict):
            raise TypeError(f"{camera_id}: intrinsic-health result must be an object")
        try:
            validate_intrinsic_health_report(camera_report, require_pass=True)
        except Exception as error:
            raise ValueError(
                f"{camera_id}: intrinsic-health semantics invalid: {error}"
            ) from error
        if camera_report.get("camera_identity_sha256") != canonical_json_sha256(
            identities[camera_id]
        ):
            raise ValueError(f"{camera_id}: intrinsic-health camera identity differs")
        if camera_report.get("target_identity_sha256") != report.get(
            "target_identity_sha256"
        ):
            raise ValueError(f"{camera_id}: intrinsic-health target identity differs")
        expected_train = [
            pose_id for pose_id in train if camera_id in per_pose_visibility[pose_id]
        ]
        expected_holdout = [
            pose_id for pose_id in holdout if camera_id in per_pose_visibility[pose_id]
        ]
        if (
            camera_report.get("train_pose_ids") != expected_train
            or camera_report.get("holdout_pose_ids") != expected_holdout
        ):
            raise ValueError(
                f"{camera_id}: intrinsic-health split differs from pose visibility"
            )
        provenance = camera_report.get("provenance")
        if not isinstance(provenance, dict) or (
            provenance.get("camera_id") != camera_id
            or provenance.get("camera_bundle_sha256") != hashes[camera_id]
            or provenance.get("bootstrap_qualification") != bootstrap[camera_id]
            or provenance.get("observations_sha256")
            != report.get("observations_sha256")
            or provenance.get("pose_plan_sha256") != report.get("pose_plan_sha256")
        ):
            raise ValueError(f"{camera_id}: intrinsic-health provenance differs")
    passed = all(
        isinstance(value, dict) and value.get("status") == "PASS"
        for value in per_camera.values()
    )
    if report.get("passed") is not passed or report.get("status") != (
        "PASS" if passed else "FAIL"
    ):
        raise ValueError(
            "intrinsic-health aggregate status differs from camera results"
        )
    if report.get("factory_intrinsics_immutable") is not True:
        raise ValueError("intrinsic-health report must preserve factory intrinsics")
    if canonical_json_sha256(report.get("target_identity")) != report.get(
        "target_identity_sha256"
    ):
        raise ValueError("intrinsic-health target identity fingerprint mismatch")
    if intrinsic_health_fingerprint(report) != report.get(
        "intrinsic_health_fingerprint"
    ):
        raise ValueError("intrinsic-health report fingerprint mismatch")
    return report


def write_rig_intrinsic_health(report: dict[str, Any], path: str | Path) -> None:
    validated = validate_rig_intrinsic_health(report)
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(validated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_rig_intrinsic_health(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("intrinsic-health report root must be an object")
    return validate_rig_intrinsic_health(raw)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _to_camera_rig_intrinsics(value: Any) -> CameraRigIntrinsics:
    return CameraRigIntrinsics(
        frame=value.frame,
        width=value.width,
        height=value.height,
        fx=value.fx,
        fy=value.fy,
        cx=value.cx,
        cy=value.cy,
        distortion_model=value.distortion_model,
        distortion_coeffs=tuple(value.distortion_coeffs),
    )
