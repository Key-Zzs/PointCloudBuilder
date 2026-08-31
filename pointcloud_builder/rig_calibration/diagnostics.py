"""Candidate-only adapters for existing cross-camera alignment diagnostics."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

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


def require_validated_candidate(
    solution: RigCalibrationSolution, validation_report: dict[str, Any]
) -> dict[str, Any]:
    """Fail closed unless a report passes and binds to the exact candidate."""

    fingerprint = solution_fingerprint(solution)
    if not solution.passed:
        raise ValueError("candidate diagnostic requires a passed calibration solution")
    if (
        validation_report.get("passed") is not True
        or validation_report.get("status") != "PASS"
        or validation_report.get("solution_fingerprint") != fingerprint
    ):
        raise ValueError(
            "candidate validation must pass and bind to the exact solution fingerprint"
        )
    holdout = validation_report.get("holdout")
    if not isinstance(holdout, dict) or holdout.get("status") != "PASS":
        raise ValueError("candidate diagnostic requires passed multicamera holdout")
    return holdout


def apply_candidate_to_live_pipeline(
    pipeline: Any,
    rig_config: Any,
    solution: RigCalibrationSolution,
    validation_report: dict[str, Any],
) -> dict[str, Any]:
    """Apply validated transforms in memory for candidate-only live viewing.

    The provisioned CameraBundles and rig configuration are never modified. The
    caller must build a fresh live pipeline and invoke this before acquisition.
    """

    holdout = require_validated_candidate(solution, validation_report)
    cameras = {camera.name: camera for camera in rig_config.enabled_cameras}
    runtimes = pipeline.processor.runtimes
    expected = set(solution.T_workspace_from_camera)
    if set(cameras) != expected or set(runtimes) != expected:
        raise ValueError("candidate, live rig, and runtime camera sets differ")
    if solution.workspace_frame != rig_config.output_frame:
        raise ValueError("candidate workspace frame differs from live rig")

    geometry_contract: dict[str, Any] = {}
    for camera_name in sorted(expected):
        camera = cameras[camera_name]
        runtime = runtimes[camera_name]
        context = runtime.pipeline.context
        if context.workspace_frame != solution.workspace_frame:
            raise ValueError(f"{camera_name}: live runtime workspace frame mismatch")
        bundle = context.calibration.bundle
        if bundle.device.to_dict() != solution.camera_identities[camera_name]:
            raise ValueError(f"{camera_name}: candidate camera identity mismatch")
        if (
            _camera_bundle_artifact_sha256(camera.source.provision_artifact)
            != solution.camera_bundle_hashes[camera_name]
        ):
            raise ValueError(f"{camera_name}: candidate CameraBundle hash mismatch")
        internal = context.calibration.transform(
            context.source_frame, solution.camera_frames[camera_name]
        )
        matrix = candidate_T_workspace_from_geometry_source(
            solution,
            camera_name,
            geometry_source_frame=context.source_frame,
            internal_transform=internal,
        )
        runtime.pipeline.context = replace(
            context,
            T_workspace_from_source=FrameExplicitTransform(
                source_frame=context.source_frame,
                target_frame=solution.workspace_frame,
                matrix=matrix,
            ),
        )
        runtime.provenance["calibration_mode"] = "validated_candidate_only"
        runtime.provenance["production_applied"] = False
        geometry_contract[camera_name] = {
            "source_frame": context.source_frame,
            "target_frame": solution.workspace_frame,
            "T_workspace_from_geometry_source": matrix.tolist(),
        }

    contract = dict(candidate_diagnostic_contract(solution))
    contract.update(
        {
            "holdout": holdout,
            "geometry_source_overrides": geometry_contract,
            "live_view_only": True,
        }
    )
    return contract


def _camera_bundle_artifact_sha256(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        source = source / "camera_bundle.json"
    return hashlib.sha256(source.read_bytes()).hexdigest()
