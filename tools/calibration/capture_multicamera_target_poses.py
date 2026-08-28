#!/usr/bin/env python3
"""Interactively capture stationary target poses into a local-only PCB artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from camera_rig.api import load_camera_config, load_provisioned_camera_bundle
from camera_rig.targets import load_target, registry

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
    parser.add_argument("--pose-count", type=int, default=20)
    parser.add_argument("--settle-matched-sets", type=int, default=3)
    parser.add_argument("--non-interactive", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pose_count < 2 or args.settle_matched_sets < 0:
        raise ValueError("--pose-count must be at least two and settle count non-negative")
    output = require_repo_local_path(args.output, label="real calibration observations")
    rig_config_path = require_repo_local_path(args.rig_config, label="real rig config")
    target_path = require_repo_local_path(args.target, label="real target artifact")
    rig = load_rig_config(rig_config_path)
    if len(rig.enabled_cameras) < 2 or any(
        camera.source.type != "camera_rig_live" for camera in rig.enabled_cameras
    ):
        raise ValueError("capture requires at least two enabled CameraRig live cameras")
    target = load_target(target_path)
    detector = registry.create(plugin_name=target.plugin, target_spec=target)
    camera_configs = {}
    bundles = {}
    projection_models = {}
    initial_camera_poses = {}
    bundle_hashes = {}
    camera_identities = {}
    serials = set()
    for camera in rig.enabled_cameras:
        source = camera.source
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
            raise ValueError(f"camera {camera.name!r} provision bundle identity mismatch")
        if camera_config.camera.serial != bundle.device.serial:
            raise ValueError(f"camera {camera.name!r} runtime/provision serial mismatch")
        if camera_config.camera.serial in serials:
            raise ValueError("live rig cameras must have distinct serial identities")
        serials.add(camera_config.camera.serial)
        if not camera_config.capture.copy_frames:
            raise ValueError("live calibration capture requires CameraRig copy_frames=true")
        fixed = bundle.fixed_mount_calibration
        if fixed is None or fixed.parent_frame != rig.output_frame:
            raise ValueError(
                f"camera {camera.name!r} provision parent frame differs from rig output"
            )
        bundles[camera.name] = bundle
        calibration = calibration_from_camera_bundle(bundle, camera_name=camera.name)
        color_frame = calibration.intrinsic_frames["color"]
        projection_models[camera.name] = calibration.intrinsics["color"]
        initial_camera_poses[camera.name] = calibration.transform(
            color_frame, calibration.workspace_frame
        ).matrix
        bundle_hashes[camera.name] = _bundle_sha256(provision_path)
        camera_identities[camera.name] = bundle.device.to_dict()
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
            pose_id = f"pose_{pose_index}"
            if not args.non_interactive:
                instruction = (
                    "Place the board at the canonical workspace pose"
                    if pose_index == 0
                    else "Move the board to a new diverse pose, then hold it stationary"
                )
                input(f"{pose_id}: {instruction}; press Enter to capture ")
            for _ in range(args.settle_matched_sets):
                if acquisition.next_frame_set(timeout_s=5.0) is None:
                    raise TimeoutError("no matched frame set while waiting for the board to settle")
            frame_set = acquisition.next_frame_set(timeout_s=5.0)
            if frame_set is None:
                raise TimeoutError(f"no matched frame set for {pose_id}")
            pose_log = {"pose_id": pose_id, "cameras": {}}
            for camera_id, envelope in sorted(frame_set.envelopes.items()):
                detected = detector.detect(envelope.frame.color.data)
                accepted = bool(detected.quality.passed and len(detected.point_ids) >= 4)
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
                        )
                    )
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
        observations=tuple(observations),
        workspace_frame=rig.output_frame,
    )
    write_observations(artifact, output)
    log_path = output.with_name(f"{output.stem}.capture-log.json")
    log_path.write_text(
        json.dumps(
            {
                "schema_version": "pointcloud-builder.rig-calibration-capture-log.v1",
                "created_at_unix_ns": time.time_ns(),
                "poses": capture_log,
                "acquisition": acquisition.report(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"CAPTURED_POSES={args.pose_count}; ACCEPTED_OBSERVATIONS={len(observations)}; "
        "local-only artifact written"
    )
    return 0


def _bundle_sha256(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        source = source / "camera_bundle.json"
    return hashlib.sha256(source.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
