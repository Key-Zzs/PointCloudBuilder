"""Explicit candidate export; no in-place CameraBundle mutation is possible here."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pointcloud_builder.rig_calibration.artifact import solution_fingerprint
from pointcloud_builder.rig_calibration.types import RigCalibrationSolution


def export_fixed_mount_candidates(
    solution: RigCalibrationSolution,
    output_root: str | Path,
    *,
    validation_report: dict[str, Any],
) -> tuple[Path, ...]:
    """Write validated per-camera candidate transforms to a new output root."""

    if not solution.passed:
        raise ValueError("refusing to export a rig calibration solution that did not pass")
    if validation_report.get("passed") is not True:
        raise ValueError("refusing to export without an explicitly passed validation report")
    if validation_report.get("solution_fingerprint") != solution_fingerprint(solution):
        raise ValueError("validation report does not bind to this candidate solution")
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for camera_id, matrix in sorted(solution.T_workspace_from_camera.items()):
        value: dict[str, Any] = {
            "schema_version": "pointcloud-builder.fixed-mount-candidate.v1",
            "status": "candidate",
            "camera_id": camera_id,
            "target_identity": solution.target_identity,
            "camera_bundle_sha256": solution.camera_bundle_hashes[camera_id],
            "camera_identity": solution.camera_identities[camera_id],
            "source_frame": solution.camera_frames[camera_id],
            "target_frame": solution.workspace_frame,
            "transform_name": "T_workspace_from_camera",
            "T_target_from_source": matrix.tolist(),
            "validation": {
                "rig_solution_passed": True,
                "validation_passed": True,
                "solution_fingerprint": validation_report["solution_fingerprint"],
                "production_applied": False,
            },
        }
        path = (root / f"{camera_id}.fixed-mount-candidate.json").resolve()
        if not path.is_relative_to(root):
            raise ValueError("candidate output escaped its requested output root")
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(path)
    return tuple(written)
