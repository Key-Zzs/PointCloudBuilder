from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).parents[1] / "tools/mapping/check_usb_topology.py"
SPEC = importlib.util.spec_from_file_location("check_usb_topology", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _device(serial: str, port: str) -> dict:
    return {"serial": serial, "physical_port": port}


def test_usb_coordinates_parse_realsense_video_path() -> None:
    assert MODULE.usb_coordinates(
        "/sys/devices/pci/usb6/6-2/6-2:1.0/video4linux/video0"
    ) == (6, "6-2")
    with pytest.raises(ValueError, match="cannot parse"):
        MODULE.usb_coordinates("unknown")


def test_initial_mapping_sorts_physical_ports() -> None:
    named = MODULE.assign_camera_names(
        [_device("B", "/usb6/6-2/"), _device("A", "/usb6/6-1/")], None
    )
    assert named["camera_a"]["serial"] == "A"
    assert named["camera_b"]["serial"] == "B"


def test_existing_mapping_keeps_names_after_ports_change() -> None:
    existing = {
        "schema_version": "pointcloud-builder.camera-identity-map.v1",
        "camera_a": {"serial": "A", "physical_port": "/usb6/6-1/"},
        "camera_b": {"serial": "B", "physical_port": "/usb6/6-2/"},
    }
    named = MODULE.assign_camera_names(
        [_device("A", "/usb4/4-2/"), _device("B", "/usb4/4-1/")], existing
    )
    assert named["camera_a"]["serial"] == "A"
    assert named["camera_a"]["physical_port"] == "/usb4/4-2/"
    assert named["camera_b"]["serial"] == "B"


def test_changed_serial_set_fails_closed() -> None:
    existing = {"camera_a": {"serial": "A"}, "camera_b": {"serial": "B"}}
    with pytest.raises(ValueError, match="differ"):
        MODULE.assign_camera_names(
            [_device("A", "/usb6/6-1/"), _device("C", "/usb6/6-2/")], existing
        )
