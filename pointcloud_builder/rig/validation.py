"""Cross-artifact validation for rig runtimes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pointcloud_builder.rig.config import RigConfig


def validate_rig_runtimes(config: RigConfig, runtimes: Mapping[str, Any]) -> None:
    expected = {camera.name for camera in config.enabled_cameras}
    actual = set(runtimes)
    if actual != expected:
        raise ValueError(f"rig runtime cameras mismatch: expected {sorted(expected)}, got {sorted(actual)}")
    for camera in config.enabled_cameras:
        runtime = runtimes[camera.name]
        if runtime.source.camera_name != camera.name:
            raise ValueError(f"camera {camera.name!r} source identity mismatch")
        if runtime.pipeline.context.calibration.camera_name != camera.name:
            raise ValueError(f"camera {camera.name!r} bundle identity mismatch")
        if runtime.pipeline.context.workspace_frame != config.output_frame:
            raise ValueError(
                f"camera {camera.name!r} output frame differs from bundle parent frame"
            )
        if runtime.pipeline.context.depth_mode != camera.depth.mode:
            raise ValueError(f"camera {camera.name!r} depth mode mismatch")
