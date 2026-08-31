#!/usr/bin/env python3
"""Discover RealSense identity/link topology without changing device state."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

PORT_PATTERN = re.compile(r"/usb(?P<bus>\d+)/(?P<node>\d+(?:-[\d.]+)+)/")


def usb_coordinates(physical_port: str) -> tuple[int, str]:
    match = PORT_PATTERN.search(physical_port)
    if match is None:
        raise ValueError(f"cannot parse RealSense physical port: {physical_port!r}")
    return int(match.group("bus")), match.group("node")


def assign_camera_names(
    devices: list[dict[str, Any]], existing_identity: dict[str, Any] | None
) -> dict[str, dict[str, Any]]:
    if existing_identity is not None:
        serial_to_name = {
            str(value["serial"]): name
            for name, value in existing_identity.items()
            if name.startswith("camera_") and isinstance(value, dict) and "serial" in value
        }
        if set(serial_to_name) != {str(item["serial"]) for item in devices}:
            raise ValueError("connected serial identities differ from the established identity map")
        return {serial_to_name[str(item["serial"])]: item for item in devices}
    ordered = sorted(devices, key=lambda item: str(item["physical_port"]))
    return {f"camera_{chr(97 + index)}": item for index, item in enumerate(ordered)}


def discover() -> list[dict[str, Any]]:
    import pyrealsense2 as rs

    result = []
    for device in rs.context().query_devices():
        value = lambda field: device.get_info(field)  # noqa: E731
        physical_port = value(rs.camera_info.physical_port)
        bus, node = usb_coordinates(physical_port)
        result.append(
            {
                "model": value(rs.camera_info.name),
                "serial": value(rs.camera_info.serial_number),
                "physical_port": physical_port,
                "usb_descriptor": value(rs.camera_info.usb_type_descriptor),
                "firmware_version": value(rs.camera_info.firmware_version),
                "actual_link_speed_mbps": _read_speed(Path(f"/sys/bus/usb/devices/{node}/speed")),
                "root_hub_bus": bus,
                "root_hub_nominal_speed_mbps": _read_speed(
                    Path(f"/sys/bus/usb/devices/usb{bus}/speed")
                ),
            }
        )
    return result


def _read_speed(path: Path) -> int:
    try:
        return round(float(path.read_text(encoding="utf-8").strip()))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot read USB speed from {path}") from error


def write_reports(
    devices: list[dict[str, Any]],
    *,
    report_path: Path,
    identity_path: Path,
    expected_count: int = 2,
) -> dict[str, Any]:
    if expected_count < 1:
        raise ValueError("expected_count must be positive")
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    existing = (
        json.loads(identity_path.read_text(encoding="utf-8"))
        if identity_path.exists()
        else None
    )
    named = assign_camera_names(devices, existing)
    shared = len({int(item["root_hub_bus"]) for item in devices}) == 1
    identity = {
        "schema_version": "pointcloud-builder.camera-identity-map.v1",
        "mapping_timestamp": (
            existing.get("mapping_timestamp", now) if existing is not None else now
        ),
        "last_observed_timestamp": now,
        **{
            name: {
                "serial": item["serial"],
                "physical_port": item["physical_port"],
            }
            for name, item in sorted(named.items())
        },
    }
    passed = bool(
        len(named) == expected_count
        and all("D435I" in str(item["model"]).upper() for item in named.values())
        and all(int(item["actual_link_speed_mbps"]) >= 5000 for item in named.values())
    )
    report = {
        "schema_version": "pointcloud-builder.usb-topology.v1",
        "captured_at": now,
        "device_count": len(named),
        "expected_count": expected_count,
        "devices": dict(sorted(named.items())),
        "shared_root_hub": shared,
        "topology_status": "SHARED_ROOT_HUB_OBSERVED" if shared else "SEPARATE_ROOT_HUBS",
        "status": "PASS" if passed else "FAIL",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--identity-map", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=2)
    args = parser.parse_args()
    report = write_reports(
        discover(),
        report_path=args.report,
        identity_path=args.identity_map,
        expected_count=args.expected_count,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "device_count": report["device_count"],
                "camera_names": sorted(report["devices"]),
                "models": [value["model"] for value in report["devices"].values()],
                "links_mbps": [
                    value["actual_link_speed_mbps"]
                    for value in report["devices"].values()
                ],
                "shared_root_hub": report["shared_root_hub"],
                "topology_status": report["topology_status"],
            },
            indent=2,
        )
    )
    if report["status"] != "PASS":
        raise SystemExit("USB topology preflight failed")


if __name__ == "__main__":
    main()
