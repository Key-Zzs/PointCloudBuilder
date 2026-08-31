"""Fail-closed production deployment for validated multi-pose rig calibration."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
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
from pointcloud_builder.rig_calibration.se3 import compose, validate_transform

DEPLOYMENT_SCHEMA_VERSION = "pointcloud-builder.rig-calibration-deployment.v1"
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
    created_by: str = "tools/calibration/promote_rig_calibration.py",
) -> dict[str, Any]:
    """Promote one exactly-bound candidate and physical acceptance."""

    solution_source = Path(solution_path).expanduser().resolve()
    validation_source = Path(validation_path).expanduser().resolve()
    physical_source = Path(physical_acceptance_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    solution = load_solution(solution_source)
    validation = _read_json(validation_source)
    physical = _read_json(physical_source)
    fingerprint = solution_fingerprint(solution)
    _require_passed_solution_validation(solution, validation, fingerprint)
    _require_physical_acceptance(solution, physical, fingerprint)

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
        "validation_sha256": sha256_file(validation_source),
        "physical_acceptance_sha256": sha256_file(physical_source),
        "source_receipts": {
            "solution": _source_receipt(solution_source),
            "validation": _source_receipt(validation_source),
            "physical_acceptance": _source_receipt(physical_source),
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
    if deployment_fingerprint(raw) != stored_fingerprint:
        raise ValueError("rig calibration deployment fingerprint mismatch")
    if canonical_json_sha256(raw.get("target_identity")) != raw.get(
        "target_identity_sha256"
    ):
        raise ValueError("deployment target identity fingerprint mismatch")
    for name in ("validation_sha256", "physical_acceptance_sha256"):
        _require_sha256(raw.get(name), name)
    receipts = raw.get("source_receipts")
    if not isinstance(receipts, dict) or set(receipts) != {
        "solution",
        "validation",
        "physical_acceptance",
    }:
        raise ValueError("deployment source receipt set mismatch")
    cameras = raw.get("cameras")
    if not isinstance(cameras, dict) or len(cameras) < 2:
        raise ValueError("deployment requires at least two cameras")
    checked: dict[str, dict[str, Any]] = {}
    for camera_id, value in sorted(cameras.items()):
        if not isinstance(value, dict) or value.get("camera_id") != camera_id:
            raise ValueError("deployment camera identity/key mismatch")
        if not isinstance(value.get("projection_frame"), str) or not value[
            "projection_frame"
        ].strip():
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
            "production_applied": True,
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
            "production_applied": True,
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
        "workspace_frame": validation.get("workspace_frame") == solution.workspace_frame,
        "target_identity": validation.get("target_identity") == solution.target_identity,
        "camera_bundle_hashes": validation.get("camera_bundle_hashes")
        == solution.camera_bundle_hashes,
        "camera_identities": validation.get("camera_identities")
        == solution.camera_identities,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("solution/validation provenance mismatch: " + ", ".join(failed))


def _require_physical_acceptance(
    solution: Any, physical: dict[str, Any], fingerprint: str
) -> None:
    if (
        physical.get("schema_version") != PHYSICAL_ACCEPTANCE_SCHEMA_VERSION
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
    }
    pairwise = physical.get("per_pair")
    expected_pairs = {
        "__".join(pair) for pair in combinations(expected_cameras, 2)
    }
    checks["per_pair"] = (
        isinstance(pairwise, dict) and set(pairwise) == expected_pairs
    )
    if checks["per_pair"]:
        checks["per_pair_status"] = all(
            isinstance(value, dict)
            and value.get("status") in {"PASS", "NOT_APPLICABLE_NO_OVERLAP"}
            and isinstance(value.get("gates"), dict)
            and bool(value["gates"])
            and all(gate is True for gate in value["gates"].values())
            and (
                value.get("status") == "NOT_APPLICABLE_NO_OVERLAP"
                or (
                    isinstance(value.get("diagnostic_residual_se3"), dict)
                    and value["diagnostic_residual_se3"].get("diagnostic_only")
                    is True
                    and value["diagnostic_residual_se3"].get("written_back")
                    is False
                )
            )
            for value in pairwise.values()
        )
    checks["thresholds"] = isinstance(physical.get("thresholds"), dict) and bool(
        physical["thresholds"]
    )
    checks["diagnostic_non_writeback"] = (
        physical.get("diagnostic_residual_writeback") is False
    )
    checks["all_gates"] = isinstance(physical.get("gates"), dict) and all(
        value is True for value in physical["gates"].values()
    )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("solution/physical acceptance mismatch: " + ", ".join(failed))


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
