"""Fail-closed production deployment for validated multi-pose rig calibration."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from pointcloud_builder.integrations.camera_rig.types import FrameExplicitTransform
from pointcloud_builder.mapping.depth_packet import canonical_bundle_sha256
from pointcloud_builder.rig_calibration.artifact import (
    load_solution,
    solution_fingerprint,
)
from pointcloud_builder.rig_calibration.intrinsic_health import (
    load_rig_intrinsic_health,
)
from pointcloud_builder.rig_calibration.se3 import compose, validate_transform
from pointcloud_builder.rig_calibration.types import validate_bootstrap_qualifications

DEPLOYMENT_SCHEMA_VERSION = "pointcloud-builder.rig-calibration-deployment.v1"
PRODUCTION_CAMERA_IDS = {"camera_a", "camera_b", "camera_c"}
PHYSICAL_ACCEPTANCE_SCHEMA_VERSION = (
    "pointcloud-builder.rig-calibration-physical-acceptance.v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class ResolvedRigCalibrationDeployment:
    """Validated deployment plus its privacy-safe runtime provenance."""

    artifact_path: Path
    artifact_fingerprint: str
    workspace_frame: str
    solution_fingerprint: str
    target_identity: dict[str, Any]
    per_camera: dict[str, dict[str, Any]]

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.per_camera))

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "calibration_mode": "validated_multipose_deployment",
            "production_applied": True,
            "rig_calibration_schema": DEPLOYMENT_SCHEMA_VERSION,
            "rig_calibration_fingerprint": self.artifact_fingerprint,
            "solution_fingerprint": self.solution_fingerprint,
            "workspace_frame": self.workspace_frame,
            "camera_bundle_hashes": {
                camera_id: self.per_camera[camera_id]["camera_bundle_sha256"]
                for camera_id in self.camera_ids
            },
        }


def promote_rig_calibration(
    solution_path: str | Path,
    validation_path: str | Path,
    physical_acceptance_path: str | Path,
    output_path: str | Path,
    *,
    intrinsic_health_path: str | Path,
    created_by: str = "tools/calibration/promote_rig_calibration.py",
) -> dict[str, Any]:
    """Promote one exactly-bound candidate and physical acceptance."""

    solution_source = Path(solution_path).expanduser().resolve()
    validation_source = Path(validation_path).expanduser().resolve()
    physical_source = Path(physical_acceptance_path).expanduser().resolve()
    intrinsic_source = Path(intrinsic_health_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    solution = load_solution(solution_source)
    validation = _read_json(validation_source)
    physical = _read_json(physical_source)
    intrinsic = load_rig_intrinsic_health(intrinsic_source)
    fingerprint = solution_fingerprint(solution)
    _require_passed_solution_validation(solution, validation, fingerprint)
    _require_physical_acceptance(
        solution,
        physical,
        fingerprint,
        solution_sha256=sha256_file(solution_source),
        validation_sha256=sha256_file(validation_source),
    )
    _require_intrinsic_health(solution, intrinsic)
    _require_fixed_three_camera_pose_plan(solution, intrinsic)

    cameras = {}
    for camera_id in sorted(solution.T_workspace_from_camera):
        identity = solution.camera_identities[camera_id]
        cameras[camera_id] = {
            "camera_id": camera_id,
            "camera_identity": identity,
            "camera_identity_sha256": canonical_json_sha256(identity),
            "camera_bundle_sha256": solution.camera_bundle_hashes[camera_id],
            "projection_frame": solution.camera_frames[camera_id],
            "T_workspace_from_camera": solution.T_workspace_from_camera[
                camera_id
            ].tolist(),
        }
    payload: dict[str, Any] = {
        "schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "status": "deployed",
        "workspace_frame": solution.workspace_frame,
        "target_identity": solution.target_identity,
        "target_identity_sha256": canonical_json_sha256(solution.target_identity),
        "solution_fingerprint": fingerprint,
        "solution_sha256": sha256_file(solution_source),
        "source_observations_sha256": solution.observations_sha256,
        "pose_plan_sha256": solution.pose_plan_sha256,
        "pose_plan_summary": solution.pose_plan_summary,
        "validation_sha256": sha256_file(validation_source),
        "physical_acceptance_sha256": sha256_file(physical_source),
        "intrinsic_health_sha256": sha256_file(intrinsic_source),
        "bootstrap_qualifications": solution.bootstrap_qualifications,
        "production_qualification_state": "PRODUCTION_QUALIFIED",
        "source_receipts": {
            "solution": _source_receipt(solution_source),
            "validation": _source_receipt(validation_source),
            "physical_acceptance": _source_receipt(physical_source),
            "intrinsic_health": _source_receipt(intrinsic_source),
        },
        "cameras": cameras,
        "creation_metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "created_by": created_by,
            "candidate_state": "physically_accepted_candidate",
            "production_applied": True,
        },
    }
    payload["rig_calibration_fingerprint"] = deployment_fingerprint(payload)
    if output.exists():
        raise FileExistsError(f"deployment output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, payload)
    load_rig_calibration_deployment(output)
    return payload


def load_rig_calibration_deployment(
    path: str | Path,
) -> ResolvedRigCalibrationDeployment:
    """Load and structurally validate one immutable deployment artifact."""

    artifact_path = Path(path).expanduser().resolve()
    raw = _read_json(artifact_path)
    if raw.get("schema_version") != DEPLOYMENT_SCHEMA_VERSION:
        raise ValueError("unsupported rig calibration deployment schema")
    if raw.get("status") != "deployed":
        raise ValueError("rig calibration deployment status must be 'deployed'")
    workspace = raw.get("workspace_frame")
    solution_id = raw.get("solution_fingerprint")
    stored_fingerprint = raw.get("rig_calibration_fingerprint")
    if not isinstance(workspace, str) or not workspace.strip():
        raise ValueError("deployment workspace_frame must be non-empty")
    _require_sha256(solution_id, "solution_fingerprint")
    _require_sha256(stored_fingerprint, "rig_calibration_fingerprint")
    assert isinstance(solution_id, str)
    assert isinstance(stored_fingerprint, str)
    if deployment_fingerprint(raw) != stored_fingerprint:
        raise ValueError("rig calibration deployment fingerprint mismatch")
    if canonical_json_sha256(raw.get("target_identity")) != raw.get(
        "target_identity_sha256"
    ):
        raise ValueError("deployment target identity fingerprint mismatch")
    if raw.get("production_qualification_state") != "PRODUCTION_QUALIFIED":
        raise ValueError("deployment lacks PRODUCTION_QUALIFIED authority")
    for name in (
        "solution_sha256",
        "validation_sha256",
        "physical_acceptance_sha256",
        "intrinsic_health_sha256",
        "source_observations_sha256",
        "pose_plan_sha256",
    ):
        _require_sha256(raw.get(name), name)
    if not isinstance(raw.get("pose_plan_summary"), dict):
        raise TypeError("deployment pose-plan summary must be an object")
    receipts = raw.get("source_receipts")
    if not isinstance(receipts, dict) or set(receipts) != {
        "solution",
        "validation",
        "physical_acceptance",
        "intrinsic_health",
    }:
        raise ValueError("deployment source receipt set mismatch")
    receipt_bindings = {
        "solution": "solution_sha256",
        "validation": "validation_sha256",
        "physical_acceptance": "physical_acceptance_sha256",
        "intrinsic_health": "intrinsic_health_sha256",
    }
    for receipt_name, digest_name in receipt_bindings.items():
        receipt = receipts.get(receipt_name)
        if (
            not isinstance(receipt, dict)
            or set(receipt) != {"path", "sha256"}
            or not isinstance(receipt.get("path"), str)
            or not receipt["path"].strip()
            or receipt.get("sha256") != raw.get(digest_name)
        ):
            raise ValueError(f"deployment {receipt_name} source receipt differs")
    cameras = raw.get("cameras")
    if not isinstance(cameras, dict) or set(cameras) != PRODUCTION_CAMERA_IDS:
        raise ValueError(
            "production deployment requires exactly camera_a/camera_b/camera_c"
        )
    checked: dict[str, dict[str, Any]] = {}
    bootstrap = raw.get("bootstrap_qualifications")
    if not isinstance(bootstrap, dict) or set(bootstrap) != set(cameras):
        raise ValueError("deployment bootstrap qualification set mismatch")
    validate_bootstrap_qualifications(bootstrap, camera_ids=set(cameras))
    for camera_id, value in sorted(cameras.items()):
        if not isinstance(value, dict) or value.get("camera_id") != camera_id:
            raise ValueError("deployment camera identity/key mismatch")
        if (
            not isinstance(value.get("projection_frame"), str)
            or not value["projection_frame"].strip()
        ):
            raise ValueError(f"{camera_id}: deployment projection frame is invalid")
        _require_sha256(value.get("camera_bundle_sha256"), "camera_bundle_sha256")
        if canonical_json_sha256(value.get("camera_identity")) != value.get(
            "camera_identity_sha256"
        ):
            raise ValueError(f"{camera_id}: camera identity receipt mismatch")
        matrix = validate_transform(
            np.asarray(value.get("T_workspace_from_camera"), dtype=np.float64),
            name=f"deployment cameras[{camera_id!r}].T_workspace_from_camera",
        )
        checked[camera_id] = {**value, "T_workspace_from_camera": matrix}
    return ResolvedRigCalibrationDeployment(
        artifact_path=artifact_path,
        artifact_fingerprint=stored_fingerprint,
        workspace_frame=workspace,
        solution_fingerprint=solution_id,
        target_identity=dict(raw["target_identity"]),
        per_camera=checked,
    )


def resolve_configured_rig_calibration(
    config: Any,
    *,
    bundles: Mapping[str, Any],
    provision_artifacts: Mapping[str, str | Path],
) -> ResolvedRigCalibrationDeployment | None:
    """Validate the configured deployment against the exact runtime rig."""

    configured = getattr(config, "rig_calibration", None)
    if configured is None:
        return None
    resolved = load_rig_calibration_deployment(configured.artifact)
    enabled = {camera.name for camera in config.enabled_cameras}
    if set(resolved.per_camera) != enabled:
        raise ValueError("deployment and runtime camera sets differ")
    if resolved.workspace_frame != config.output_frame:
        raise ValueError("deployment and runtime workspace frames differ")
    if set(bundles) != enabled or set(provision_artifacts) != enabled:
        raise ValueError("runtime bundle set is incomplete for deployment validation")
    for camera_id in sorted(enabled):
        expected = resolved.per_camera[camera_id]
        bundle = bundles[camera_id]
        if bundle.device.to_dict() != expected["camera_identity"]:
            raise ValueError(f"{camera_id}: deployed camera identity mismatch")
        actual_hash = camera_bundle_artifact_sha256(
            provision_artifacts[camera_id], bundle=bundle
        )
        if actual_hash != expected["camera_bundle_sha256"]:
            raise ValueError(f"{camera_id}: deployed CameraBundle hash mismatch")
    return resolved


def configured_rig_calibration_provenance(config: Any) -> dict[str, Any]:
    """Resolve config-level identity for map compatibility before hardware opens."""

    configured = getattr(config, "rig_calibration", None)
    if configured is None:
        return {
            "calibration_mode": "fixed_provision",
            "rig_calibration_schema": None,
            "rig_calibration_fingerprint": None,
            "solution_fingerprint": None,
            "camera_bundle_hashes": {},
            "production_applied": False,
        }
    return load_rig_calibration_deployment(configured.artifact).provenance


def runtime_rig_calibration_provenance(
    runtimes: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize the exact centrally resolved geometry used by runtime builders."""

    if not runtimes:
        raise ValueError("rig calibration provenance requires at least one runtime")
    values = {name: dict(runtime.provenance) for name, runtime in runtimes.items()}
    fields = (
        "calibration_mode",
        "rig_calibration_schema",
        "rig_calibration_fingerprint",
        "solution_fingerprint",
        "production_applied",
    )
    common = {}
    for field in fields:
        found = {value.get(field) for value in values.values()}
        if len(found) != 1:
            raise ValueError(f"runtime cameras disagree on {field}")
        common[field] = next(iter(found))
    if common["production_applied"] is not True:
        raise ValueError("production runtime provenance must be applied")
    return {
        **common,
        "camera_bundle_hashes": {
            name: values[name]["camera_bundle_sha256"] for name in sorted(values)
        },
    }


def apply_deployment_to_context(
    context: Any,
    camera_id: str,
    deployment: ResolvedRigCalibrationDeployment | None,
) -> tuple[Any, dict[str, Any]]:
    """Resolve one geometry source transform using the central precedence rule."""

    bundle_hash = canonical_bundle_sha256(context.calibration.bundle)
    if deployment is None:
        return context, {
            "calibration_mode": "fixed_provision",
            "production_applied": False,
            "rig_calibration_schema": None,
            "rig_calibration_fingerprint": None,
            "solution_fingerprint": None,
            "camera_bundle_sha256": bundle_hash,
        }
    value = deployment.per_camera[camera_id]
    internal = context.calibration.transform(
        context.source_frame, value["projection_frame"]
    )
    matrix = compose(value["T_workspace_from_camera"], internal.matrix)
    updated = replace(
        context,
        workspace_frame=deployment.workspace_frame,
        T_workspace_from_source=FrameExplicitTransform(
            source_frame=context.source_frame,
            target_frame=deployment.workspace_frame,
            matrix=matrix,
        ),
    )
    provenance = {
        **deployment.provenance,
        "camera_bundle_sha256": value["camera_bundle_sha256"],
        "projection_frame": value["projection_frame"],
        "geometry_source_frame": context.source_frame,
    }
    return updated, provenance


def deployment_fingerprint(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("rig_calibration_fingerprint", None)
    return canonical_json_sha256(payload)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def camera_bundle_artifact_sha256(path: str | Path, *, bundle: Any) -> str:
    text = str(path)
    if text.startswith("synthetic://"):
        return canonical_bundle_sha256(bundle)
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        candidate = candidate / "camera_bundle.json"
    return sha256_file(candidate)


def _require_passed_solution_validation(
    solution: Any, validation: dict[str, Any], fingerprint: str
) -> None:
    if not solution.passed:
        raise ValueError("refusing to deploy a failed calibration solution")
    if (
        validation.get("schema_version")
        != "pointcloud-builder.rig-calibration-validation.v1"
        or validation.get("passed") is not True
        or validation.get("status") != "PASS"
        or validation.get("solution_fingerprint") != fingerprint
    ):
        raise ValueError("validation must PASS and bind the exact solution")
    holdout = validation.get("holdout")
    if not isinstance(holdout, dict) or holdout.get("status") != "PASS":
        raise ValueError("production deployment requires a passed holdout")
    checks = {
        "workspace_frame": validation.get("workspace_frame")
        == solution.workspace_frame,
        "target_identity": validation.get("target_identity")
        == solution.target_identity,
        "camera_bundle_hashes": validation.get("camera_bundle_hashes")
        == solution.camera_bundle_hashes,
        "camera_identities": validation.get("camera_identities")
        == solution.camera_identities,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "solution/validation provenance mismatch: " + ", ".join(failed)
        )


def _require_physical_acceptance(
    solution: Any,
    physical: dict[str, Any],
    fingerprint: str,
    *,
    solution_sha256: str,
    validation_sha256: str,
) -> None:
    from pointcloud_builder.rig_calibration.physical_acceptance import (
        REAL_DUAL_MULTIPOSE_V1_THRESHOLDS,
    )

    required_fields = {
        "schema_version",
        "status",
        "passed",
        "solution_fingerprint",
        "workspace_frame",
        "target_identity",
        "camera_set",
        "camera_bundle_hashes",
        "camera_identities",
        "per_pair",
        "all_rig",
        "thresholds",
        "gates",
        "diagnostic_residual_writeback",
        "source_receipts",
        "calibration_mode",
        "production_applied",
        "rig_calibration_schema",
        "rig_calibration_fingerprint",
        "source_recording_manifest_sha256",
        "selected_matched_set_indices",
    }
    if (
        set(physical) != required_fields
        or physical.get("schema_version") != PHYSICAL_ACCEPTANCE_SCHEMA_VERSION
        or physical.get("passed") is not True
        or physical.get("status") != "PASS"
        or physical.get("solution_fingerprint") != fingerprint
    ):
        raise ValueError("physical 3D acceptance must PASS and bind the exact solution")
    expected_cameras = sorted(solution.T_workspace_from_camera)
    checks = {
        "camera_set": physical.get("camera_set") == expected_cameras,
        "workspace_frame": physical.get("workspace_frame") == solution.workspace_frame,
        "target_identity": physical.get("target_identity") == solution.target_identity,
        "camera_bundle_hashes": physical.get("camera_bundle_hashes")
        == solution.camera_bundle_hashes,
        "camera_identities": physical.get("camera_identities")
        == solution.camera_identities,
        "candidate_mode": physical.get("calibration_mode") == "validated_candidate"
        and physical.get("production_applied") is False
        and physical.get("rig_calibration_schema") is None
        and physical.get("rig_calibration_fingerprint") is None,
    }
    expected_thresholds = asdict(REAL_DUAL_MULTIPOSE_V1_THRESHOLDS)
    checks["frozen_thresholds"] = physical.get("thresholds") == expected_thresholds
    recording_sha256 = physical.get("source_recording_manifest_sha256")
    receipts = physical.get("source_receipts")
    checks["source_receipts"] = (
        isinstance(receipts, dict)
        and set(receipts) == {"solution", "validation", "source_recording_manifest"}
        and _valid_source_receipt(receipts.get("solution"), solution_sha256)
        and _valid_source_receipt(receipts.get("validation"), validation_sha256)
        and _valid_physical_recording_receipt(
            receipts.get("source_recording_manifest"),
            recording_sha256,
            camera_ids=expected_cameras,
            workspace_frame=solution.workspace_frame,
            camera_bundle_hashes=solution.camera_bundle_hashes,
            selected_indices=physical.get("selected_matched_set_indices"),
        )
    )
    pairwise = physical.get("per_pair")
    expected_pairs = {"__".join(pair) for pair in combinations(expected_cameras, 2)}
    checks["per_pair"] = isinstance(pairwise, dict) and set(pairwise) == expected_pairs
    if checks["per_pair"]:
        assert isinstance(pairwise, dict)
        pair_checks = {
            pair: _physical_pair_semantics(
                pair,
                value,
                thresholds=expected_thresholds,
                evaluated_matched_set_count=300,
            )
            for pair, value in pairwise.items()
        }
        checks["per_pair_semantics"] = all(pair_checks.values())
        accepted_edges = [
            tuple(pair.split("__", maxsplit=1))
            for pair, value in pairwise.items()
            if isinstance(value, dict) and value.get("status") == "PASS"
        ]
        connected = _camera_graph_connected(expected_cameras, accepted_edges)
        physical_gates = physical.get("gates")
        checks["accepted_overlap_connectivity"] = connected
        checks["reported_overlap_connectivity"] = (
            isinstance(physical_gates, dict)
            and physical_gates.get("connected_overlap_graph") is connected
        )
        checks["all_rig_semantics"] = _physical_all_rig_semantics(
            physical.get("all_rig"),
            camera_ids=expected_cameras,
            accepted_edges=accepted_edges,
            selected_indices=physical.get("selected_matched_set_indices"),
        )
    checks["diagnostic_non_writeback"] = (
        physical.get("diagnostic_residual_writeback") is False
    )
    checks["all_gates"] = physical.get("gates") == {
        "all_pair_declarations_valid": True,
        "connected_overlap_graph": True,
    }
    checks["recording_sha256"] = (
        isinstance(recording_sha256, str)
        and _SHA256.fullmatch(recording_sha256) is not None
    )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("solution/physical acceptance mismatch: " + ", ".join(failed))


def _physical_pair_semantics(
    pair: str,
    value: Any,
    *,
    thresholds: dict[str, Any],
    evaluated_matched_set_count: int,
) -> bool:
    if not isinstance(value, dict):
        return False
    cameras = pair.split("__", maxsplit=1)
    if value.get("cameras") != cameras:
        return False
    if value.get("status") == "NOT_APPLICABLE_NO_OVERLAP":
        return value == {
            "status": "NOT_APPLICABLE_NO_OVERLAP",
            "cameras": cameras,
            "evaluated_matched_set_count": evaluated_matched_set_count,
            "physical_justification_required": True,
            "declared_no_overlap": True,
            "gates": {"no_measurable_overlap": True},
        }
    required = {
        "status",
        "cameras",
        "evaluated_matched_set_count",
        "overlap_point_count",
        "symmetric_nn",
        "board_interior",
        "board_plane",
        "diagnostic_residual_se3",
        "gates",
    }
    if (
        set(value) != required
        or value.get("evaluated_matched_set_count") != evaluated_matched_set_count
    ):
        return False
    overlap = value.get("overlap_point_count")
    symmetric = value.get("symmetric_nn")
    board = value.get("board_interior")
    plane = value.get("board_plane")
    residual = value.get("diagnostic_residual_se3")
    if (
        not isinstance(overlap, dict)
        or set(overlap) != {"minimum_per_direction"}
        or not isinstance(symmetric, dict)
        or set(symmetric)
        != {"median_of_matched_set_medians_mm", "p95_of_matched_set_p95_mm"}
        or not isinstance(board, dict)
        or set(board)
        != {
            "status",
            "median_of_matched_set_medians_mm",
            "p95_of_matched_set_p95_mm",
        }
        or not isinstance(plane, dict)
        or set(plane)
        != {
            "absolute_signed_offset_p95_mm",
            "normal_split_p95_deg",
            "double_layer_thickness_p95_mm",
        }
        or not isinstance(residual, dict)
        or set(residual)
        != {
            "translation_xyz_mm",
            "translation_norm_mm",
            "rotation_geodesic_deg",
            "aggregation",
            "diagnostic_only",
            "written_back",
        }
    ):
        return False
    overlap_count = overlap.get("minimum_per_direction")
    if isinstance(overlap_count, bool) or not isinstance(overlap_count, int):
        return False
    metric_names = {
        "symmetric_median": symmetric.get("median_of_matched_set_medians_mm"),
        "symmetric_p95": symmetric.get("p95_of_matched_set_p95_mm"),
        "board_median": board.get("median_of_matched_set_medians_mm"),
        "board_p95": board.get("p95_of_matched_set_p95_mm"),
        "plane_offset": plane.get("absolute_signed_offset_p95_mm"),
        "normal_split": plane.get("normal_split_p95_deg"),
        "double_layer_thickness": plane.get("double_layer_thickness_p95_mm"),
        "diagnostic_translation": residual.get("translation_norm_mm"),
        "diagnostic_rotation": residual.get("rotation_geodesic_deg"),
    }
    if any(not _finite_nonnegative(item) for item in metric_names.values()):
        return False
    translation = residual.get("translation_xyz_mm")
    if (
        not isinstance(translation, list)
        or len(translation) != 3
        or any(not _finite_number(item) for item in translation)
        or not math.isclose(
            float(metric_names["diagnostic_translation"]),
            float(np.linalg.norm(np.asarray(translation, dtype=np.float64))),
            abs_tol=1e-9,
        )
        or residual.get("aggregation") != "aggregate_matched_set_point_to_plane_icp"
        or residual.get("diagnostic_only") is not True
        or residual.get("written_back") is not False
    ):
        return False
    expected_gates = {
        "minimum_overlap_points": overlap_count >= thresholds["minimum_overlap_points"],
        "symmetric_median": metric_names["symmetric_median"]
        <= thresholds["maximum_symmetric_median_mm"],
        "symmetric_p95": metric_names["symmetric_p95"]
        <= thresholds["maximum_symmetric_p95_mm"],
        "board_available": board.get("status") == "AVAILABLE",
        "board_median": metric_names["board_median"]
        <= thresholds["maximum_board_median_mm"],
        "board_p95": metric_names["board_p95"] <= thresholds["maximum_board_p95_mm"],
        "plane_offset": metric_names["plane_offset"]
        <= thresholds["maximum_plane_offset_mm"],
        "normal_split": metric_names["normal_split"]
        <= thresholds["maximum_normal_split_deg"],
        "double_layer_thickness": metric_names["double_layer_thickness"]
        <= thresholds["maximum_double_layer_thickness_mm"],
        "diagnostic_translation": metric_names["diagnostic_translation"]
        <= thresholds["maximum_diagnostic_translation_mm"],
        "diagnostic_rotation": metric_names["diagnostic_rotation"]
        <= thresholds["maximum_diagnostic_rotation_deg"],
    }
    return value.get("gates") == expected_gates and value.get("status") == (
        "PASS" if all(expected_gates.values()) else "FAIL"
    )


def _physical_all_rig_semantics(
    value: Any,
    *,
    camera_ids: list[str],
    accepted_edges: list[tuple[str, str]],
    selected_indices: Any,
) -> bool:
    required = {
        "camera_graph_nodes",
        "accepted_pairwise_overlap_edges",
        "pairwise_overlap_connectivity",
        "per_camera_point_contribution",
        "all_camera_input_point_count",
        "camera_drop_statistics",
        "matcher_statistics",
    }
    if not isinstance(value, dict) or set(value) != required:
        return False
    reported_edges = value.get("accepted_pairwise_overlap_edges")
    expected_edges = {tuple(edge) for edge in accepted_edges}
    if (
        value.get("camera_graph_nodes") != camera_ids
        or not isinstance(reported_edges, list)
        or any(not isinstance(edge, list) or len(edge) != 2 for edge in reported_edges)
        or {tuple(edge) for edge in reported_edges} != expected_edges
    ):
        return False
    connected = _camera_graph_connected(camera_ids, accepted_edges)
    if value.get("pairwise_overlap_connectivity") != {
        "connected": connected,
        "accepted_edge_count": len(accepted_edges),
        "required_node_count": len(camera_ids),
    }:
        return False
    contributions = value.get("per_camera_point_contribution")
    if not isinstance(contributions, dict) or set(contributions) != set(camera_ids):
        return False
    counts: dict[str, int] = {}
    for camera_id, contribution in contributions.items():
        if not isinstance(contribution, dict) or set(contribution) != {
            "point_count",
            "fraction",
        }:
            return False
        point_count = contribution.get("point_count")
        if (
            isinstance(point_count, bool)
            or not isinstance(point_count, int)
            or point_count <= 0
        ):
            return False
        counts[camera_id] = point_count
    total = sum(counts.values())
    if value.get("all_camera_input_point_count") != total:
        return False
    if any(
        not math.isclose(
            float(contributions[camera_id]["fraction"]),
            counts[camera_id] / total,
            abs_tol=1e-12,
        )
        if _finite_nonnegative(contributions[camera_id].get("fraction"))
        else True
        for camera_id in camera_ids
    ):
        return False
    matcher = value.get("matcher_statistics")
    if not isinstance(matcher, dict):
        return False
    available = matcher.get("available_complete_sets")
    return not (
        value.get("camera_drop_statistics")
        != {camera_id: 0 for camera_id in camera_ids}
        or set(matcher) != {"available_complete_sets", "evaluated_complete_sets"}
        or matcher.get("evaluated_complete_sets") != 300
        or isinstance(available, bool)
        or not isinstance(available, int)
        or available < 300
        or not isinstance(selected_indices, list)
        or len(selected_indices) != 300
        or len(set(selected_indices)) != 300
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= available
            for index in selected_indices
        )
    )


def _valid_source_receipt(value: Any, expected_sha256: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"path", "sha256"}
        and isinstance(value.get("path"), str)
        and bool(value["path"])
        and isinstance(expected_sha256, str)
        and value.get("sha256") == expected_sha256
        and _SHA256.fullmatch(expected_sha256) is not None
    )


def _valid_physical_recording_receipt(
    value: Any,
    expected_sha256: Any,
    *,
    camera_ids: list[str],
    workspace_frame: str,
    camera_bundle_hashes: dict[str, str],
    selected_indices: Any,
) -> bool:
    from pointcloud_builder.mapping.recording import validate_rig_depth_recording

    if not _valid_source_receipt(value, expected_sha256):
        return False
    assert isinstance(value, dict)
    manifest_path = Path(value["path"]).expanduser().resolve()
    if (
        manifest_path.name != "manifest.json"
        or not manifest_path.is_file()
        or sha256_file(manifest_path) != expected_sha256
    ):
        return False
    try:
        manifest = validate_rig_depth_recording(manifest_path.parent)
    except (OSError, TypeError, ValueError):
        return False
    matched_sets = manifest.get("matched_sets")
    recording_indices = (
        [item.get("matched_set_index") for item in matched_sets]
        if isinstance(matched_sets, list)
        else None
    )
    calibration = manifest.get("rig_calibration")
    return (
        manifest.get("depth_source") == "ffs_stereo"
        and manifest.get("calibration_purpose") == "physical_acceptance_candidate"
        and manifest.get("workspace_frame") == workspace_frame
        and manifest.get("camera_names") == camera_ids
        and manifest.get("matched_set_count") == 300
        and recording_indices == selected_indices
        and isinstance(calibration, dict)
        and calibration.get("calibration_mode") == "fixed_provision"
        and calibration.get("production_applied") is False
        and calibration.get("camera_bundle_hashes") == camera_bundle_hashes
    )


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _finite_nonnegative(value: Any) -> bool:
    return _finite_number(value) and float(value) >= 0.0


def _require_intrinsic_health(solution: Any, report: dict[str, Any]) -> None:
    from camera_rig.calibration.intrinsic_health import IntrinsicHealthThresholds

    if not solution.bootstrap_qualifications:
        raise ValueError(
            "production deployment requires bootstrap qualification authority"
        )
    checks = {
        "status": report.get("status") == "PASS" and report.get("passed") is True,
        "camera_set": report.get("camera_set") == sorted(solution.camera_frames),
        "target_identity": report.get("target_identity") == solution.target_identity,
        "camera_bundle_hashes": (
            report.get("camera_bundle_hashes") == solution.camera_bundle_hashes
        ),
        "camera_identities": report.get("camera_identities")
        == solution.camera_identities,
        "bootstrap_qualifications": (
            report.get("bootstrap_qualifications") == solution.bootstrap_qualifications
        ),
        "factory_intrinsics_immutable": (
            report.get("factory_intrinsics_immutable") is True
        ),
        "observations_sha256": report.get("observations_sha256")
        == solution.observations_sha256,
        "pose_plan_sha256": report.get("pose_plan_sha256") == solution.pose_plan_sha256,
        "pose_plan_summary": report.get("pose_plan_summary")
        == solution.pose_plan_summary,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    per_camera = report.get("per_camera")
    if isinstance(per_camera, dict):
        frozen_thresholds = IntrinsicHealthThresholds().to_dict()
        if any(
            not isinstance(value, dict) or value.get("thresholds") != frozen_thresholds
            for value in per_camera.values()
        ):
            failed.append("frozen_intrinsic_thresholds")
    if failed:
        raise ValueError("solution/intrinsic health mismatch: " + ", ".join(failed))


def _require_fixed_three_camera_pose_plan(
    solution: Any, report: dict[str, Any]
) -> None:
    if set(solution.camera_frames) != PRODUCTION_CAMERA_IDS:
        raise ValueError(
            "production promotion requires exactly camera_a/camera_b/camera_c"
        )
    summary = solution.pose_plan_summary
    if not isinstance(summary, dict):
        raise TypeError("production promotion requires a frozen pose-plan summary")
    pose_ids = summary.get("pose_ids")
    solve_ids = summary.get("solve_pose_ids")
    holdout_ids = summary.get("holdout_pose_ids")
    per_pose = summary.get("per_pose_camera_ids")
    checks = {
        "pose_count_30": isinstance(pose_ids, list) and len(pose_ids) == 30,
        "solve_count_24": isinstance(solve_ids, list) and len(solve_ids) == 24,
        "holdout_count_6": isinstance(holdout_ids, list) and len(holdout_ids) == 6,
        "split_partition": isinstance(pose_ids, list)
        and isinstance(solve_ids, list)
        and isinstance(holdout_ids, list)
        and set(solve_ids).isdisjoint(holdout_ids)
        and set(solve_ids) | set(holdout_ids) == set(pose_ids),
        "capture_complete": summary.get("capture_complete") is True,
        "all_poses_attempted": isinstance(per_pose, dict)
        and isinstance(pose_ids, list)
        and set(per_pose) == set(pose_ids),
        "intrinsic_global_split": report.get("train_pose_ids")
        == sorted(solve_ids or [])
        and report.get("holdout_pose_ids") == sorted(holdout_ids or []),
        "source_observations_bound": isinstance(solution.observations_sha256, str),
        "pose_plan_bound": isinstance(solution.pose_plan_sha256, str),
    }
    if (
        isinstance(pose_ids, list)
        and isinstance(solve_ids, list)
        and isinstance(holdout_ids, list)
        and isinstance(per_pose, dict)
    ):
        valid_visibility = all(
            isinstance(per_pose.get(pose_id), list)
            and bool(per_pose[pose_id])
            and len(set(per_pose[pose_id])) == len(per_pose[pose_id])
            and set(per_pose[pose_id]) <= PRODUCTION_CAMERA_IDS
            for pose_id in pose_ids
        )
        checks["pose_0_first"] = bool(pose_ids) and pose_ids[0] == "pose_0"
        checks["unique_pose_ids"] = len(set(pose_ids)) == len(pose_ids)
        checks["nonempty_valid_visibility"] = valid_visibility
        if valid_visibility:
            solve_edges = {
                tuple(pair)
                for pose_id in solve_ids
                for pair in combinations(sorted(per_pose[pose_id]), 2)
            }
            checks["connected_solve_visibility"] = _camera_graph_connected(
                sorted(PRODUCTION_CAMERA_IDS), sorted(solve_edges)
            )
            checks["per_camera_split_support"] = all(
                sum(camera_id in per_pose[pose_id] for pose_id in solve_ids) >= 12
                and sum(camera_id in per_pose[pose_id] for pose_id in holdout_ids) >= 4
                for camera_id in PRODUCTION_CAMERA_IDS
            )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("production pose-plan contract failed: " + ", ".join(failed))


def _camera_graph_connected(
    camera_ids: list[str], edges: list[tuple[str, str]]
) -> bool:
    if not camera_ids:
        return False
    adjacency = {camera_id: set() for camera_id in camera_ids}
    for left, right in edges:
        if left not in adjacency or right not in adjacency or left == right:
            return False
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen = {camera_ids[0]}
    pending = [camera_ids[0]]
    while pending:
        current = pending.pop()
        for neighbor in sorted(adjacency[current] - seen):
            seen.add(neighbor)
            pending.append(neighbor)
    return seen == set(camera_ids)


def _source_receipt(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _require_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("artifact root must be a JSON object")
    return raw


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
