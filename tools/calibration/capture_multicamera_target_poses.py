#!/usr/bin/env python3
"""Interactively capture stationary target poses into a local-only PCB artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from camera_rig.api import load_camera_config, load_provisioned_camera_bundle
from camera_rig.targets import load_target
from camera_rig.targets.charuco.detector import CharucoDetector
from camera_rig.targets.charuco.quality import CharucoQualityThresholds

from pointcloud_builder.integrations.camera_rig import calibration_from_camera_bundle
from pointcloud_builder.local_paths import require_repo_local_path
from pointcloud_builder.rig import load_rig_config
from pointcloud_builder.rig.live import LiveRigAcquisition
from pointcloud_builder.rig_calibration.artifact import write_observations
from pointcloud_builder.rig_calibration.observations import (
    from_camera_rig_target_observation,
)
from pointcloud_builder.rig_calibration.types import RigCalibrationObservations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-config", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pose-count", type=int, default=30)
    parser.add_argument(
        "--holdout-pose-count",
        type=int,
        default=6,
        help="Predeclare the final N captured poses as holdout observations.",
    )
    parser.add_argument(
        "--min-corners-per-observation",
        type=int,
        default=6,
        help="Reject detections below this corner count (default: solver gate of 6).",
    )
    parser.add_argument("--settle-matched-sets", type=int, default=3)
    parser.add_argument("--non-interactive", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pose_count < 2 or args.settle_matched_sets < 0:
        raise ValueError(
            "--pose-count must be at least two and settle count non-negative"
        )
    if args.holdout_pose_count < 0 or args.holdout_pose_count >= args.pose_count:
        raise ValueError(
            "--holdout-pose-count must be non-negative and below pose count"
        )
    if args.min_corners_per_observation < 4:
        raise ValueError("--min-corners-per-observation must be at least four")
    output = require_repo_local_path(args.output, label="real calibration observations")
    log_path = output.with_name(f"{output.stem}.capture-log.json")
    plan_path = output.with_name(f"{output.stem}.pose-plan.json")
    if output.exists() or log_path.exists() or plan_path.exists():
        raise FileExistsError(
            "real calibration capture output already exists; choose a fresh artifact root"
        )
    rig_config_path = require_repo_local_path(args.rig_config, label="real rig config")
    target_path = require_repo_local_path(args.target, label="real target artifact")
    rig = load_rig_config(rig_config_path)
    if len(rig.enabled_cameras) < 2 or any(
        camera.source.type != "camera_rig_live" for camera in rig.enabled_cameras
    ):
        raise ValueError("capture requires at least two enabled CameraRig live cameras")
    target = load_target(target_path)
    if not (
        target.target_name == "charuco_a4_v1"
        and target.squares_x == 7
        and target.squares_y == 5
        and abs(target.square_length_m - 0.03) <= 1e-12
        and abs(target.board_width_m - 0.21) <= 1e-12
        and abs(target.board_height_m - 0.15) <= 1e-12
    ):
        raise ValueError(
            "production capture requires the authoritative charuco_a4_v1 target"
        )
    detector = _production_detector(target)
    camera_configs = {}
    bundles = {}
    projection_models = {}
    initial_camera_poses = {}
    bundle_hashes = {}
    camera_identities = {}
    serials = set()
    bootstrap_qualifications = {}
    if args.pose_count != 30 or args.holdout_pose_count != 6:
        raise ValueError("production capture uses the frozen 30-pose/6-holdout plan")
    if {camera.name for camera in rig.enabled_cameras} != {
        "camera_a",
        "camera_b",
        "camera_c",
    }:
        raise ValueError(
            "production capture requires exactly camera_a/camera_b/camera_c"
        )
    for camera in rig.enabled_cameras:
        source = camera.source
        if source.type != "camera_rig_live":
            raise TypeError("production capture requires CameraRig live sources")
        runtime_path = require_repo_local_path(
            source.camera_config, label="real CameraRig runtime config"
        )
        provision_path = require_repo_local_path(
            source.provision_artifact, label="real CameraBundle"
        )
        camera_configs[camera.name] = load_camera_config(runtime_path)
        bundle = load_provisioned_camera_bundle(provision_path)
        camera_config = camera_configs[camera.name]
        if camera_config.camera.name != camera.name:
            raise ValueError(f"camera {camera.name!r} runtime config identity mismatch")
        if bundle.device.camera_name != camera.name:
            raise ValueError(
                f"camera {camera.name!r} provision bundle identity mismatch"
            )
        if camera_config.camera.serial != bundle.device.serial:
            raise ValueError(
                f"camera {camera.name!r} runtime/provision serial mismatch"
            )
        if camera_config.camera.serial in serials:
            raise ValueError("live rig cameras must have distinct serial identities")
        serials.add(camera_config.camera.serial)
        if not camera_config.capture.copy_frames:
            raise ValueError(
                "live calibration capture requires CameraRig copy_frames=true"
            )
        fixed = bundle.fixed_mount_calibration
        if fixed is None or fixed.parent_frame != rig.output_frame:
            raise ValueError(
                f"camera {camera.name!r} provision parent frame differs from rig output"
            )
        authority = bundle.provenance.get("calibration_authority")
        if not isinstance(authority, dict):
            raise TypeError(
                f"camera {camera.name!r} lacks bootstrap qualification authority"
            )
        bootstrap_qualifications[camera.name] = dict(authority)
        bundles[camera.name] = bundle
        calibration = calibration_from_camera_bundle(bundle, camera_name=camera.name)
        color_frame = calibration.intrinsic_frames["color"]
        projection_models[camera.name] = calibration.intrinsics["color"]
        initial_camera_poses[camera.name] = calibration.transform(
            color_frame, calibration.workspace_frame
        ).matrix
        bundle_hashes[camera.name] = _bundle_sha256(provision_path)
        camera_identities[camera.name] = bundle.device.to_dict()
    pose_plan = _frozen_pose_plan(args.pose_count, args.holdout_pose_count)
    plan_payload = {
        "schema_version": "pointcloud-builder.rig-calibration-pose-plan.v1",
        "frozen_before_hardware_open": True,
        "camera_set": sorted(camera_configs),
        "target_identity": {
            "target_name": target.target_name,
            "target_frame": target.target_frame,
            "target_spec_sha256": target.artifact_sha256,
        },
        "pose_count": args.pose_count,
        "holdout_pose_count": args.holdout_pose_count,
        "poses": pose_plan,
        "rules": {
            "camera_mounts_stationary": True,
            "detection_policy": detector.thresholds.policy,
            "pose_0_canonical_first": True,
            "train_holdout_split_predeclared": True,
            "holdout_not_used_by_bundle_adjustment": True,
        },
    }
    plan_path.write_text(
        json.dumps(plan_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    observations = []
    capture_log = []
    acquisition = LiveRigAcquisition(
        camera_configs,
        timing=rig.timing,
        live_config=rig.live,
        required_streams_by_camera={name: ("color",) for name in camera_configs},
    )
    with acquisition:
        for pose_index in range(args.pose_count):
            plan_item = pose_plan[pose_index]
            pose_id = plan_item["pose_id"]
            split = plan_item["split"]
            if not args.non_interactive:
                instruction = plan_item["operator_instruction"]
                input(f"{pose_id}: {instruction}; press Enter to capture ")
            for _ in range(args.settle_matched_sets):
                if acquisition.next_frame_set(timeout_s=5.0) is None:
                    raise TimeoutError(
                        "no matched frame set while waiting for the board to settle"
                    )
            frame_set = acquisition.next_frame_set(timeout_s=5.0)
            if frame_set is None:
                raise TimeoutError(f"no matched frame set for {pose_id}")
            pose_log: dict[str, Any] = {
                "pose_id": pose_id,
                "split": split,
                "cameras": {},
            }
            for camera_id, envelope in sorted(frame_set.envelopes.items()):
                detected = detector.detect(envelope.frame.color.data)
                accepted = bool(
                    detected.quality.passed
                    and len(detected.point_ids) >= args.min_corners_per_observation
                )
                pose_log["cameras"][camera_id] = {
                    "accepted": accepted,
                    "corner_count": len(detected.point_ids),
                    "quality": detected.quality.to_dict(),
                    "timestamp_ns": envelope.host_receive_timestamp_ns,
                }
                if accepted:
                    observations.append(
                        from_camera_rig_target_observation(
                            detected,
                            observation_id=f"{camera_id}:{pose_id}",
                            camera_id=camera_id,
                            pose_id=pose_id,
                            timestamp_ns=envelope.host_receive_timestamp_ns,
                            split=split,
                        )
                    )
            print(_pose_capture_result(pose_log), flush=True)
            capture_log.append(pose_log)
    artifact = RigCalibrationObservations(
        target_identity={
            "target_name": target.target_name,
            "target_frame": target.target_frame,
            "target_spec_sha256": target.artifact_sha256,
        },
        camera_bundle_hashes=bundle_hashes,
        camera_identities=camera_identities,
        projection_models=projection_models,
        initial_camera_poses=initial_camera_poses,
        bootstrap_qualifications=bootstrap_qualifications,
        pose_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        pose_plan_summary={
            "pose_ids": [f"pose_{index}" for index in range(args.pose_count)],
            "solve_pose_ids": [
                f"pose_{index}"
                for index in range(args.pose_count - args.holdout_pose_count)
            ],
            "holdout_pose_ids": [
                f"pose_{index}"
                for index in range(
                    args.pose_count - args.holdout_pose_count, args.pose_count
                )
            ],
            "capture_complete": len(capture_log) == args.pose_count,
            "per_pose_camera_ids": {
                item["pose_id"]: sorted(
                    camera_id
                    for camera_id, value in item["cameras"].items()
                    if value["accepted"] is True
                )
                for item in capture_log
            },
        },
        observations=tuple(observations),
        workspace_frame=rig.output_frame,
    )
    write_observations(artifact, output)
    log_path.write_text(
        json.dumps(
            {
                "schema_version": "pointcloud-builder.rig-calibration-capture-log.v1",
                "created_at_unix_ns": time.time_ns(),
                "pose_count": args.pose_count,
                "holdout_pose_count": args.holdout_pose_count,
                "min_corners_per_observation": args.min_corners_per_observation,
                "detection_policy": detector.thresholds.policy,
                "detection_thresholds": detector.thresholds.__dict__,
                "poses": capture_log,
                "pose_plan_path": str(plan_path),
                "pose_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                "acquisition": acquisition.report(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = (
        f"CAPTURED_POSES={args.pose_count}; HOLDOUT_POSES={args.holdout_pose_count}; "
        f"ACCEPTED_OBSERVATIONS={len(observations)}; "
        "local-only artifact written"
    )
    print(summary)
    if not observations:
        print(
            "CAPTURE_STATUS=FAIL; REASON=NO_ACCEPTED_OBSERVATIONS; "
            "preserved capture log for diagnosis"
        )
        return 2
    return 0


def _production_detector(target: object) -> CharucoDetector:
    """Use the preregistered pose-observability detection policy for multipose."""
    return CharucoDetector(
        target,
        thresholds=CharucoQualityThresholds.uncertainty_validated(),
    )


def _pose_capture_result(pose_log: dict[str, Any]) -> str:
    camera_results = []
    for camera_id, value in sorted(pose_log["cameras"].items()):
        accepted = "ACCEPTED" if value["accepted"] else "REJECTED"
        reasons = value["quality"]["failure_reasons"]
        reason_suffix = f",reasons={'|'.join(reasons)}" if reasons else ""
        camera_results.append(
            f"{camera_id}={accepted}(corners={value['corner_count']}{reason_suffix})"
        )
    return f"POSE_RESULT={pose_log['pose_id']}; " + "; ".join(camera_results)


def _bundle_sha256(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        source = source / "camera_bundle.json"
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _frozen_pose_plan(pose_count: int, holdout_count: int) -> list[dict[str, str]]:
    instructions = (
        "Place the board at the canonical workspace pose; keep every camera fixed",
        "Move the board left in the shared field of view",
        "Move the board right in the shared field of view",
        "Move the board high in the shared field of view",
        "Move the board low in the shared field of view",
        "Move the board nearer while all cameras retain full target visibility",
        "Move the board farther while all cameras retain sufficient corners",
        "Apply a moderate positive yaw",
        "Apply a moderate negative yaw",
        "Apply a moderate positive pitch",
        "Apply a moderate negative pitch",
        "Apply combined positive yaw and pitch",
        "Apply combined negative yaw and positive pitch",
        "Apply combined positive yaw and negative pitch",
        "Apply combined negative yaw and pitch",
        "Use a left-near pose with moderate obliquity",
        "Use a right-near pose with moderate obliquity",
        "Use a high-far pose with moderate obliquity",
        "Use a low-far pose with moderate obliquity",
        "Use a left-high pose at mid distance",
        "Use a right-high pose at mid distance",
        "Use a left-low pose at mid distance",
        "Use a right-low pose at mid distance",
        "Use a distinct mid-field compound orientation",
        "Holdout: use a novel center-near compound orientation",
        "Holdout: use a novel center-far compound orientation",
        "Holdout: use a novel left-high compound orientation",
        "Holdout: use a novel right-low compound orientation",
        "Holdout: use a novel positive-yaw negative-pitch orientation",
        "Holdout: use a novel negative-yaw positive-pitch orientation",
    )
    if pose_count != len(instructions) or holdout_count != 6:
        raise ValueError(
            "frozen production pose plan requires exactly 30 poses and 6 holdouts"
        )
    return [
        {
            "pose_id": f"pose_{index}",
            "split": "holdout" if index >= pose_count - holdout_count else "solve",
            "operator_instruction": instruction,
        }
        for index, instruction in enumerate(instructions)
    ]


if __name__ == "__main__":
    raise SystemExit(main())
