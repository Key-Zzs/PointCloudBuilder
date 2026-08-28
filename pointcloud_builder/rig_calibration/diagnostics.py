"""Candidate-only adapters for existing cross-camera alignment diagnostics."""

from __future__ import annotations

import numpy as np

from pointcloud_builder.integrations.camera_rig.types import FrameExplicitTransform
from pointcloud_builder.rig_calibration.artifact import solution_fingerprint
from pointcloud_builder.rig_calibration.se3 import compose, validate_transform
from pointcloud_builder.rig_calibration.types import RigCalibrationSolution


def candidate_T_workspace_from_geometry_source(
    solution: RigCalibrationSolution,
    camera_id: str,
    *,
    geometry_source_frame: str,
    internal_transform: FrameExplicitTransform,
) -> np.ndarray:
    """Resolve a diagnostic geometry transform without applying it to production.

    A color-target calibration commonly optimizes ``T_workspace_from_color``
    while FFS geometry lives in ``ir_left``.  The caller must supply the
    authoritative, frame-explicit CameraRig ``T_color_from_ir_left``; both
    source and target frame names are checked before composition.
    """

    if camera_id not in solution.T_workspace_from_camera:
        raise KeyError(f"candidate solution has no camera {camera_id!r}")
    expected_projection_frame = solution.camera_frames[camera_id]
    if internal_transform.target_frame != expected_projection_frame:
        raise ValueError(
            f"projection frame mismatch: expected {expected_projection_frame!r}, "
            f"got {internal_transform.target_frame!r}"
        )
    if not geometry_source_frame.strip():
        raise ValueError("geometry_source_frame must be non-empty")
    if internal_transform.source_frame != geometry_source_frame:
        raise ValueError(
            f"geometry source frame mismatch: expected {geometry_source_frame!r}, "
            f"got {internal_transform.source_frame!r}"
        )
    internal = validate_transform(
        internal_transform.matrix,
        name=(
            f"T_{internal_transform.target_frame}_from_"
            f"{internal_transform.source_frame}"
        ),
    )
    return compose(solution.T_workspace_from_camera[camera_id], internal)


def candidate_diagnostic_contract(solution: RigCalibrationSolution) -> dict[str, object]:
    """Return explicit overrides for diagnostic-only before/after evaluation."""

    return {
        "schema_version": "pointcloud-builder.rig-calibration-diagnostic-overrides.v1",
        "candidate_only": True,
        "production_applied": False,
        "solution_fingerprint": solution_fingerprint(solution),
        "workspace_frame": solution.workspace_frame,
        "target_identity": solution.target_identity,
        "per_camera": {
            camera_id: {
                "camera_bundle_sha256": solution.camera_bundle_hashes[camera_id],
                "camera_identity": solution.camera_identities[camera_id],
                "source_frame": solution.camera_frames[camera_id],
                "target_frame": solution.workspace_frame,
                "transform_name": "T_workspace_from_camera",
                "T_target_from_source": matrix.tolist(),
            }
            for camera_id, matrix in sorted(solution.T_workspace_from_camera.items())
        },
    }
