"""Build strict FFS calibration directly from a passed CameraRig bundle."""

from __future__ import annotations

import numpy as np

from pointcloud_builder.ffs.calibration import FFSCalibration, make_calibration
from pointcloud_builder.integrations.camera_rig.calibration_adapter import (
    camera_intrinsics_to_pcb,
    resolve_bundle_transform,
)
from pointcloud_builder.integrations.camera_rig.dependencies import CameraBundle
from pointcloud_builder.integrations.camera_rig.validation import (
    validate_passed_fixed_bundle,
)


def calibration_from_camera_bundle(
    bundle: CameraBundle,
    camera_name: str | None = None,
) -> FFSCalibration:
    """Construct left-IR FFS geometry without copied or fabricated values."""

    validate_passed_fixed_bundle(bundle, camera_name)
    missing = sorted({"ir_left", "ir_right", "color"} - set(bundle.intrinsics))
    if missing:
        raise ValueError(f"CameraBundle is missing FFS intrinsics: {missing}")
    left = bundle.intrinsics["ir_left"]
    right = bundle.intrinsics["ir_right"]
    color = bundle.intrinsics["color"]
    T_right_from_left = resolve_bundle_transform(bundle, left.frame, right.frame)
    T_color_from_left = resolve_bundle_transform(bundle, left.frame, color.frame)
    baseline_m = float(np.linalg.norm(T_right_from_left.matrix[:3, 3]))
    return make_calibration(
        camera_intrinsics_to_pcb(left, pixel_geometry="rectified"),
        camera_intrinsics_to_pcb(right, pixel_geometry="rectified"),
        tuple(float(value) for value in left.distortion_coeffs),
        tuple(float(value) for value in right.distortion_coeffs),
        _extrinsics(T_right_from_left.matrix),
        _extrinsics(T_color_from_left.matrix),
        baseline_m,
        rectification_mode="require_rectified",
    )


def _extrinsics(matrix: np.ndarray):
    from pointcloud_builder.camera_model import CameraExtrinsics

    rotation = tuple(tuple(float(value) for value in row) for row in matrix[:3, :3])
    translation = tuple(float(value) for value in matrix[:3, 3])
    return CameraExtrinsics(rotation=rotation, translation=translation)  # type: ignore[arg-type]
