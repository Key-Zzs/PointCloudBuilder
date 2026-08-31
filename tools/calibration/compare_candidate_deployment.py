#!/usr/bin/env python3
"""Prove that one promoted deployment preserves its exact candidate extrinsics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from pointcloud_builder.rig_calibration.artifact import (
    load_solution,
    solution_fingerprint,
)
from pointcloud_builder.rig_calibration.deployment import (
    load_rig_calibration_deployment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-solution", required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    local = (Path.cwd() / ".local").resolve()
    if not output.is_relative_to(local):
        raise ValueError("comparison output must stay under .local/")
    if output.exists():
        raise FileExistsError(f"comparison output already exists: {output}")

    solution = load_solution(args.candidate_solution)
    deployment = load_rig_calibration_deployment(args.deployment)
    fingerprint = solution_fingerprint(solution)
    if fingerprint != deployment.solution_fingerprint:
        raise ValueError("candidate and deployment solution fingerprints differ")
    if set(solution.T_workspace_from_camera) != set(deployment.per_camera):
        raise ValueError("candidate and deployment camera sets differ")

    per_camera = {}
    for name in sorted(solution.T_workspace_from_camera):
        candidate = solution.T_workspace_from_camera[name]
        deployed = deployment.per_camera[name]["T_workspace_from_camera"]
        delta = np.linalg.inv(candidate) @ deployed
        cosine = float(np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
        per_camera[name] = {
            "maximum_matrix_absolute_difference": float(
                np.max(np.abs(candidate - deployed))
            ),
            "translation_difference_mm": float(
                np.linalg.norm(delta[:3, 3]) * 1000.0
            ),
            "rotation_geodesic_difference_deg": float(math.degrees(math.acos(cosine))),
        }
    gates = {
        "solution_fingerprint_exact": True,
        "camera_set_exact": True,
        "all_matrix_differences_le_1e-12": all(
            value["maximum_matrix_absolute_difference"] <= 1e-12
            for value in per_camera.values()
        ),
    }
    report = {
        "schema_version": "pointcloud-builder.candidate-deployment-equivalence.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "passed": all(gates.values()),
        "solution_fingerprint": fingerprint,
        "rig_calibration_fingerprint": deployment.artifact_fingerprint,
        "camera_set": sorted(per_camera),
        "per_camera": per_camera,
        "composition_note": (
            "candidate and deployment share each projection-frame transform; "
            "both compose the same CameraRig internal source transform"
        ),
        "gates": gates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("candidate/deployment equivalence failed")


if __name__ == "__main__":
    main()
