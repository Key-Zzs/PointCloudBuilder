#!/usr/bin/env python3
"""Audit one CameraRig bundle against librealsense without printing calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

from camera_rig.api import (
    CameraSession,
    load_camera_bundle,
    load_camera_config,
)

from pointcloud_builder.local_paths import require_repo_local_path
from pointcloud_builder.projection_parity import (
    audit_camera_bundle_projection,
    write_projection_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--camera-label", required=True, choices=("camera_a", "camera_b"))
    parser.add_argument(
        "--runtime-config",
        type=Path,
        help="Optional private CameraRig config used to prove the physical camera is online",
    )
    parser.add_argument(
        "--allow-busy-connected",
        action="store_true",
        help=(
            "When a user-owned pipeline holds the device, accept an identity-matched "
            "connected-device probe while recording dedicated capture as deferred"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_path = require_repo_local_path(args.bundle, label="real CameraBundle")
    output_path = require_repo_local_path(args.output, label="real projection report")
    bundle = load_camera_bundle(bundle_path)
    report = audit_camera_bundle_projection(
        bundle,
        camera_label=args.camera_label,
        bundle_path=bundle_path,
    )
    if args.runtime_config is not None:
        runtime_path = require_repo_local_path(
            args.runtime_config, label="real CameraRig runtime config"
        )
        config = load_camera_config(runtime_path)
        try:
            with CameraSession.from_config(config) as session:
                frame = session.capture()
        except Exception as error:
            if not args.allow_busy_connected or "busy" not in str(error).casefold():
                raise
            if not _configured_device_is_connected(config, bundle):
                raise RuntimeError(
                    "busy-device fallback could not bind the configured camera identity"
                ) from error
            report["real_hardware_binding"] = {
                "status": "PASS_CONNECTED_DEVICE_BUSY",
                "camera_label": args.camera_label,
                "configured_identity_connected": True,
                "dedicated_capture": "DEFERRED_EXISTING_ACTIVE_PIPELINE",
                "serial_redacted": True,
            }
        else:
            required_streams = ("color", "depth", "ir_left", "ir_right")
            missing = sorted(set(required_streams) - set(frame.streams))
            if missing:
                raise RuntimeError(f"physical capture is missing required streams: {missing}")
            report["real_hardware_binding"] = {
                "status": "PASS",
                "camera_label": args.camera_label,
                "captured_required_streams": list(required_streams),
                "serial_redacted": True,
            }
    else:
        report["real_hardware_binding"] = {
            "status": "NOT_RUN",
            "reason": "--runtime-config was not provided",
        }
    write_projection_report(report, output_path)
    model_status = report["acceptance"]["MODEL_PARITY"]
    hardware_status = report["real_hardware_binding"]["status"]
    print(
        f"{args.camera_label}: MODEL_PARITY={model_status}; "
        f"REAL_HARDWARE_BINDING={hardware_status}; report written under local output"
    )
    return (
        0
        if model_status == "PASS"
        and hardware_status in {"PASS", "PASS_CONNECTED_DEVICE_BUSY", "NOT_RUN"}
        else 1
    )


def _configured_device_is_connected(config: object, bundle: object) -> bool:
    import pyrealsense2 as rs

    configured_serial = str(config.camera.serial)  # type: ignore[attr-defined]
    bundle_serial = str(bundle.device.serial)  # type: ignore[attr-defined]
    if configured_serial != bundle_serial:
        return False
    connected = {
        str(device.get_info(rs.camera_info.serial_number))
        for device in rs.context().query_devices()
    }
    return configured_serial in connected


if __name__ == "__main__":
    raise SystemExit(main())
