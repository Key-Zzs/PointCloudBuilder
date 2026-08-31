#!/usr/bin/env python3
"""Evaluate all N-camera pairs from an immutable rig-depth recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml
from camera_rig.api import load_provisioned_camera_bundle

from pointcloud_builder.deprojection import deproject_depth
from pointcloud_builder.diagnostics.cross_camera_alignment import apply_transform
from pointcloud_builder.integrations.camera_rig.calibration_adapter import (
    resolve_bundle_transform,
)
from pointcloud_builder.mapping.depth_packet import provision_identity_sha256
from pointcloud_builder.mapping.recording import (
    iter_rig_depth_recording,
    validate_rig_depth_recording,
)
from pointcloud_builder.rig import load_rig_config
from pointcloud_builder.rig_calibration.artifact import (
    load_solution,
    solution_fingerprint,
)
from pointcloud_builder.rig_calibration.deployment import (
    camera_bundle_artifact_sha256,
    load_rig_calibration_deployment,
)
from pointcloud_builder.rig_calibration.diagnostics import require_validated_candidate
from pointcloud_builder.rig_calibration.physical_acceptance import (
    NCameraAcceptanceThresholds,
    aggregate_ncamera_evaluations,
    bind_physical_acceptance,
    diagnostic_point_to_plane_residual,
    evaluate_ncamera_alignment,
    generate_pairs,
    pair_key,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rig-config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rig-calibration")
    mode.add_argument("--candidate-solution")
    parser.add_argument("--candidate-validation")
    parser.add_argument("--recording", required=True)
    parser.add_argument("--mapping-config", required=True)
    parser.add_argument("--physical-acceptance")
    parser.add_argument("--thresholds")
    parser.add_argument("--matched-sets", type=int, default=5)
    parser.add_argument(
        "--declared-no-overlap",
        action="append",
        default=[],
        metavar="CAMERA_A:CAMERA_B",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.matched_sets < 1:
        raise ValueError("--matched-sets must be positive")
    candidate_mode = args.candidate_solution is not None
    if candidate_mode != (args.candidate_validation is not None):
        raise ValueError(
            "candidate mode requires both --candidate-solution and "
            "--candidate-validation"
        )
    if candidate_mode and args.thresholds is None:
        raise ValueError("candidate mode requires preregistered --thresholds")
    if candidate_mode and args.physical_acceptance is not None:
        raise ValueError("candidate mode cannot reuse a prior physical acceptance")
    if not candidate_mode and args.thresholds is not None:
        raise ValueError("deployed regression uses --physical-acceptance thresholds")

    output = _private_output(args.output)
    if output.exists():
        raise FileExistsError(f"N-camera acceptance output already exists: {output}")
    rig = load_rig_config(args.rig_config)
    if candidate_mode:
        if rig.rig_calibration is not None:
            raise ValueError("candidate acceptance requires rig_calibration to be omitted")
        solution = load_solution(args.candidate_solution)
        validation_path = Path(args.candidate_validation).expanduser().resolve()
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        require_validated_candidate(solution, validation)
        candidate_fingerprint = solution_fingerprint(solution)
        calibration = SimpleNamespace(
            workspace_frame=solution.workspace_frame,
            solution_fingerprint=candidate_fingerprint,
            camera_ids=tuple(sorted(solution.T_workspace_from_camera)),
            per_camera={
                name: {
                    "camera_identity": solution.camera_identities[name],
                    "camera_bundle_sha256": solution.camera_bundle_hashes[name],
                    "projection_frame": solution.camera_frames[name],
                    "T_workspace_from_camera": solution.T_workspace_from_camera[name],
                }
                for name in sorted(solution.T_workspace_from_camera)
            },
        )
        deployment = None
    else:
        if rig.rig_calibration is None:
            raise ValueError("deployed acceptance requires configured rig_calibration")
        deployment = load_rig_calibration_deployment(args.rig_calibration)
        configured = load_rig_calibration_deployment(rig.rig_calibration.artifact)
        if deployment.artifact_fingerprint != configured.artifact_fingerprint:
            raise ValueError("CLI and rig-config deployments differ")
        calibration = deployment
    recording = Path(args.recording).expanduser().resolve()
    recording_manifest = validate_rig_depth_recording(recording)
    camera_set = sorted(camera.name for camera in rig.enabled_cameras)
    if recording_manifest["camera_names"] != camera_set:
        raise ValueError("recording and rig-config camera sets differ")
    if calibration.camera_ids != tuple(camera_set):
        raise ValueError("calibration and recording camera sets differ")

    cameras = {camera.name: camera for camera in rig.enabled_cameras}
    bundles = {}
    for name in camera_set:
        source = cameras[name].source
        bundle = load_provisioned_camera_bundle(source.provision_artifact)
        expected = calibration.per_camera[name]
        if bundle.device.to_dict() != expected["camera_identity"]:
            raise ValueError(f"{name}: deployed camera identity mismatch")
        if (
            camera_bundle_artifact_sha256(source.provision_artifact, bundle=bundle)
            != expected["camera_bundle_sha256"]
        ):
            raise ValueError(f"{name}: deployed CameraBundle hash mismatch")
        bundles[name] = bundle

    mapping = _mapping_config(args.mapping_config, calibration.workspace_frame)
    frame_sets = list(iter_rig_depth_recording(recording))
    selected = _select_evenly(frame_sets, args.matched_sets)
    evaluations: list[dict[str, Any]] = []
    selected_clouds: list[dict[str, np.ndarray]] = []
    for frame_set in selected:
        clouds: dict[str, np.ndarray] = {}
        for observation in frame_set.observations:
            name = observation.camera_name
            source = cameras[name].source
            bundle = bundles[name]
            if observation.bundle_identity != str(bundle.bundle_id):
                raise ValueError(f"{name}: recording CameraBundle identity mismatch")
            if observation.provision_sha256 != provision_identity_sha256(
                source.provision_artifact
            ):
                raise ValueError(f"{name}: recording provision receipt mismatch")
            value = calibration.per_camera[name]
            internal = resolve_bundle_transform(
                bundle, observation.source_frame, value["projection_frame"]
            )
            T_workspace_from_source = (
                value["T_workspace_from_camera"] @ internal.matrix
            )
            depth = torch.from_numpy(np.array(observation.depth, copy=True))
            points_camera, _mask = deproject_depth(
                depth,
                observation.intrinsics,
                observation.depth_scale_m_per_unit,
            )
            points = apply_transform(
                points_camera.numpy(), T_workspace_from_source
            )
            points = _crop(points, mapping["workspace_crop"])
            clouds[name] = points
        board_masks = {
            name: _board_mask(points, mapping["expected_plane"])
            for name, points in clouds.items()
        }
        evaluations.append(
            evaluate_ncamera_alignment(
                clouds,
                thresholds=_thresholds(
                    args.thresholds or args.physical_acceptance, deployment
                ),
                board_masks=board_masks,
                declared_no_overlap_pairs={
                    _parse_pair(value) for value in args.declared_no_overlap
                },
            )
        )
        selected_clouds.append(clouds)
    thresholds = _thresholds(args.thresholds or args.physical_acceptance, deployment)
    declared_no_overlap = {
        tuple(sorted(_parse_pair(value))) for value in args.declared_no_overlap
    }
    diagnostic_residuals = {
        pair_key(left, right): diagnostic_point_to_plane_residual(
            [clouds[left] for clouds in selected_clouds],
            [clouds[right] for clouds in selected_clouds],
        )
        for left, right in generate_pairs(camera_set)
        if (left, right) not in declared_no_overlap
    }
    report = aggregate_ncamera_evaluations(
        evaluations,
        thresholds=thresholds,
        diagnostic_residuals=diagnostic_residuals,
    )
    runtime_receipt = {
        "workspace_frame": calibration.workspace_frame,
        "calibration_mode": (
            "validated_candidate" if candidate_mode else "validated_multipose_deployment"
        ),
        "production_applied": not candidate_mode,
        "rig_calibration_schema": (
            None
            if candidate_mode
            else "pointcloud-builder.rig-calibration-deployment.v1"
        ),
        "rig_calibration_fingerprint": (
            None if candidate_mode else deployment.artifact_fingerprint
        ),
        "solution_fingerprint": calibration.solution_fingerprint,
        "source_recording_manifest_sha256": _sha256(recording / "manifest.json"),
        "selected_matched_set_indices": [
            item.matched_set_index for item in selected
        ],
        "all_rig": {
            **report["all_rig"],
            "camera_drop_statistics": _recording_drop_statistics(
                recording_manifest, camera_set
            ),
            "matcher_statistics": {
                "available_complete_sets": len(frame_sets),
                "evaluated_complete_sets": len(selected),
            },
        },
    }
    report.update(runtime_receipt)
    if candidate_mode:
        report = bind_physical_acceptance(
            report,
            args.candidate_solution,
            source_receipts={
                "validation": {
                    "path": str(validation_path),
                    "sha256": _sha256(validation_path),
                },
                "source_recording_manifest": {
                    "path": str(recording / "manifest.json"),
                    "sha256": _sha256(recording / "manifest.json"),
                },
            },
        )
        report.update(
            {
                key: value
                for key, value in runtime_receipt.items()
                if key != "all_rig"
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "camera_set": report["camera_set"],
                "pair_count": len(report["per_pair"]),
                "connected": report["gates"]["connected_overlap_graph"],
                "rig_calibration_fingerprint": (
                    None if candidate_mode else deployment.artifact_fingerprint
                ),
                "solution_fingerprint": calibration.solution_fingerprint,
                "production_applied": not candidate_mode,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit("N-camera alignment acceptance failed")


def _mapping_config(path: str, workspace_frame: str) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("mapping config must be a mapping")
    crop = raw.get("workspace_crop")
    plane = raw.get("expected_plane")
    if not isinstance(crop, dict) or not isinstance(plane, dict):
        raise ValueError("mapping config requires workspace_crop and expected_plane")
    if plane.get("frame") != workspace_frame:
        raise ValueError("mapping expected-plane frame differs from deployment")
    return {"workspace_crop": crop, "expected_plane": plane}


def _thresholds(
    explicit: str | None, deployment: Any | None
) -> NCameraAcceptanceThresholds:
    path = None if explicit is None else Path(explicit).expanduser().resolve()
    if path is None and deployment is None:
        raise ValueError("candidate acceptance requires explicit thresholds")
    if path is None:
        raw_deployment = json.loads(deployment.artifact_path.read_text(encoding="utf-8"))
        receipt = raw_deployment["source_receipts"]["physical_acceptance"]
        path = Path(receipt["path"])
        if _sha256(path) != receipt["sha256"]:
            raise ValueError("deployed physical-acceptance receipt changed")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if deployment is not None and raw.get("solution_fingerprint") != (
        deployment.solution_fingerprint
    ):
        raise ValueError("physical thresholds bind a different solution")
    values = raw.get("thresholds", raw)
    if not isinstance(values, dict):
        raise ValueError("physical acceptance has no threshold contract")
    return NCameraAcceptanceThresholds(**values)


def _select_evenly(values: list[Any], count: int) -> list[Any]:
    if count > len(values):
        raise ValueError("requested matched sets exceed recording length")
    indices = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[int(index)] for index in indices]


def _crop(points: np.ndarray, raw: dict[str, Any]) -> np.ndarray:
    if raw.get("enabled") is not True:
        return points
    mask = np.ones(len(points), dtype=bool)
    for axis, name in enumerate(("x", "y", "z")):
        bounds = raw[name]
        mask &= (points[:, axis] >= float(bounds[0])) & (
            points[:, axis] <= float(bounds[1])
        )
    return points[mask]


def _board_mask(points: np.ndarray, raw: dict[str, Any]) -> np.ndarray:
    z = raw["z_search_range_m"]
    return (
        (points[:, 0] >= float(raw["x"][0]))
        & (points[:, 0] <= float(raw["x"][1]))
        & (points[:, 1] >= float(raw["y"][0]))
        & (points[:, 1] <= float(raw["y"][1]))
        & (points[:, 2] >= float(z[0]))
        & (points[:, 2] <= float(z[1]))
    )


def _parse_pair(value: str) -> tuple[str, str]:
    parts = value.split(":")
    if len(parts) != 2 or not all(parts):
        raise ValueError("--declared-no-overlap must be CAMERA_A:CAMERA_B")
    return parts[0], parts[1]


def _recording_drop_statistics(
    manifest: dict[str, Any], camera_set: list[str]
) -> dict[str, int]:
    expected = int(manifest["matched_set_count"])
    observed = {name: 0 for name in camera_set}
    for frame in manifest["matched_sets"]:
        for camera in frame["cameras"]:
            observed[camera["camera_name"]] += 1
    return {name: expected - observed[name] for name in camera_set}


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _private_output(value: str) -> Path:
    output = Path(value).expanduser().resolve()
    if not output.is_relative_to((Path.cwd() / ".local").resolve()):
        raise ValueError("real N-camera acceptance reports must be under .local/")
    return output


if __name__ == "__main__":
    main()
