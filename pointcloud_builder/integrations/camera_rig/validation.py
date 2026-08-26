"""Fail-closed validation shared by CameraRig integration components."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def validate_passed_fixed_bundle(bundle: Any, camera_name: str | None = None) -> None:
    """Require a passed bundle with passed fixed-mount calibration."""

    if getattr(bundle, "status", None) != "passed":
        raise ValueError("CameraBundle status must be 'passed'")
    quality = getattr(bundle, "quality", None)
    if quality is None or not bool(getattr(quality, "passed", False)):
        raise ValueError("CameraBundle quality decision must be passed")
    fixed = getattr(bundle, "fixed_mount_calibration", None)
    if fixed is None:
        raise ValueError("CameraBundle is missing fixed_mount_calibration")
    if not bool(getattr(getattr(fixed, "quality", None), "passed", False)):
        raise ValueError("fixed_mount_calibration quality decision must be passed")
    if camera_name is not None:
        actual = getattr(getattr(bundle, "device", None), "camera_name", None)
        if actual != camera_name:
            raise ValueError(
                f"CameraBundle camera_name mismatch: expected {camera_name!r}, got {actual!r}"
            )


def require_streams(frame: Any, required_streams: Iterable[str]) -> None:
    """Require named CameraFrame streams without accepting aliases silently."""

    streams = getattr(frame, "streams", None)
    if not isinstance(streams, dict):
        raise TypeError("expected a CameraRig CameraFrame with a streams mapping")
    missing = sorted(name for name in set(required_streams) if streams.get(name) is None)
    if missing:
        raise ValueError(f"CameraFrame is missing required streams: {missing}")


def validate_frame_identity(
    frame: Any,
    *,
    expected_camera_name: str,
    expected_serial: str,
) -> None:
    """Bind a CameraFrame to the same physical identity as its calibration bundle."""

    if getattr(frame, "camera_name", None) != expected_camera_name:
        raise ValueError("CameraFrame camera_name does not match the CameraBundle")
    if getattr(frame, "serial", None) != expected_serial:
        raise ValueError("CameraFrame serial does not match the CameraBundle")
