#!/usr/bin/env python3
"""Prepare private per-camera CameraRig configs from a confirmed identity map."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_TEMPLATE = (
    ROOT / "third_party/CameraRig/configs/examples/single_camera_contract.yaml"
)
PROVISION_TEMPLATE = (
    ROOT / "third_party/CameraRig/configs/examples/fixed_provision_contract.yaml"
)
IDENTITY_SCHEMA = "pointcloud-builder.camera-identity-map.v1"
REPORT_SCHEMA = "pointcloud-builder.camera-rig-calibration-preparation.v1"
CAMERA_LABEL = re.compile(r"camera_[a-z][a-z0-9_]*\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class PreparationError(ValueError):
    """Raised when private calibration inputs are incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise PreparationError(f"could not load YAML {path}: {error}") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise PreparationError(f"YAML root must be a string-keyed mapping: {path}")
    return value


def load_identity_map(path: Path, expected_count: int) -> dict[str, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreparationError(
            f"could not load camera identity map {path}: {error}"
        ) from error
    if not isinstance(value, dict) or value.get("schema_version") != IDENTITY_SCHEMA:
        raise PreparationError(f"identity map must use schema {IDENTITY_SCHEMA}")
    cameras: dict[str, str] = {}
    for label, entry in value.items():
        if not CAMERA_LABEL.fullmatch(label):
            continue
        if not isinstance(entry, dict) or not isinstance(entry.get("serial"), str):
            raise PreparationError(
                f"identity entry {label!r} must contain a serial string"
            )
        serial = entry["serial"].strip()
        if not serial:
            raise PreparationError(f"identity entry {label!r} has an empty serial")
        cameras[label] = serial
    if len(cameras) != expected_count:
        raise PreparationError(
            f"identity map contains {len(cameras)} cameras; expected exactly {expected_count}"
        )
    if len(set(cameras.values())) != len(cameras):
        raise PreparationError("identity map contains duplicate camera serials")
    return dict(sorted(cameras.items()))


def _load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreparationError(f"could not load {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise PreparationError(f"{label} must be a JSON object: {path}")
    return value


def validate_target_artifact(path: Path) -> dict[str, str]:
    target = path.resolve(strict=False)
    if target.name != "target_spec.json" or not target.is_file():
        raise PreparationError(f"target must be an existing target_spec.json: {path}")
    root = target.parent
    spec = _load_json_mapping(target, "target specification")
    schema = spec.get("schema_version")
    if schema not in {
        "camera-rig.target.charuco-resolved.v1",
        "camera-rig.target.charuco-resolved.v2",
    }:
        raise PreparationError(f"unsupported resolved target schema: {schema!r}")
    target_frame = spec.get("target_frame")
    if not isinstance(target_frame, str) or not target_frame:
        raise PreparationError("target specification has no target_frame")
    target_name = spec.get("target_name")
    if not isinstance(target_name, str) or not target_name:
        raise PreparationError("target specification has no target_name")

    checksum_path = root / "checksums.sha256"
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise PreparationError(f"could not load target checksums: {error}") from error
    checksums: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", maxsplit=1)
        if (
            len(parts) != 2
            or not SHA256.fullmatch(parts[0])
            or not parts[1]
            or Path(parts[1]).name != parts[1]
            or parts[1] in checksums
        ):
            raise PreparationError("target checksums.sha256 is invalid")
        checksums[parts[1]] = parts[0]
    if "target_spec.json" not in checksums:
        raise PreparationError("target checksums do not contain target_spec.json")

    if schema == "camera-rig.target.charuco-resolved.v1":
        expected_payloads = {
            f"{target_name}_board.png",
            f"{target_name}_print.pdf",
            f"{target_name}_preview.png",
            "generation_report.json",
            "target_spec.json",
        }
    elif spec.get("source_type") == "existing_physical":
        expected_payloads = {"registration_report.json", "target_spec.json"}
    else:
        artifact_files = spec.get("artifact_files")
        if not isinstance(artifact_files, dict) or not all(
            isinstance(name, str) and name for name in artifact_files.values()
        ):
            raise PreparationError("generated v2 target has invalid artifact_files")
        expected_payloads = set(artifact_files.values()) | {
            "generation_report.json",
            "target_spec.json",
        }
    if set(checksums) != expected_payloads:
        raise PreparationError("target checksums contain missing or unexpected paths")

    expected_files = set(checksums) | {"checksums.sha256"}
    actual_files: set[str] = set()
    for candidate in root.iterdir():
        if candidate.is_symlink():
            raise PreparationError(
                f"target artifact contains a symlink: {candidate.name}"
            )
        if candidate.is_file():
            actual_files.add(candidate.name)
    if actual_files != expected_files:
        raise PreparationError("target artifact contains missing or unexpected files")
    for name, expected in checksums.items():
        if sha256_file(root / name) != expected:
            raise PreparationError(f"target artifact checksum mismatch: {name}")

    report_names = {"generation_report.json", "registration_report.json"} & set(
        checksums
    )
    if len(report_names) != 1:
        raise PreparationError(
            "target artifact must contain exactly one provenance report"
        )
    report = _load_json_mapping(root / report_names.pop(), "target provenance report")
    if report.get("status") != "PASS":
        raise PreparationError("target provenance report status is not PASS")
    target_sha256 = checksums["target_spec.json"]
    if report.get("target_spec_sha256") != target_sha256:
        raise PreparationError(
            "target provenance report has a different target SHA-256"
        )
    return {"sha256": target_sha256, "target_frame": target_frame}


def _camera_section(config: dict[str, Any], path: Path) -> dict[str, Any]:
    value = config.get("camera")
    if not isinstance(value, dict):
        raise PreparationError(f"template or config has no camera mapping: {path}")
    return value


def build_runtime_config(
    template: dict[str, Any], label: str, serial: str
) -> dict[str, Any]:
    config = deepcopy(template)
    camera = _camera_section(config, RUNTIME_TEMPLATE)
    camera["name"] = label
    camera["serial"] = serial
    return config


def build_provision_config(
    template: dict[str, Any],
    *,
    label: str,
    serial: str,
    config_path: Path,
    target_path: Path,
    target_sha256: str,
    target_frame: str,
) -> dict[str, Any]:
    config = deepcopy(template)
    camera = _camera_section(config, PROVISION_TEMPLATE)
    camera["name"] = label
    camera["serial"] = serial
    target = config.get("target")
    workspace = config.get("workspace")
    if not isinstance(target, dict) or not isinstance(workspace, dict):
        raise PreparationError(
            "fixed provision template has no target/workspace mapping"
        )
    target["artifact"] = Path(
        os.path.relpath(
            target_path.resolve(strict=False),
            config_path.parent.resolve(strict=False),
        )
    ).as_posix()
    target["expected_sha256"] = target_sha256
    target["detection_policy"] = "uncertainty_validated"
    workspace["target_frame"] = target_frame
    workspace["T_workspace_from_target"] = {
        "matrix": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
    return config


def _atomic_write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.write_text(
            yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def plan_yaml(
    path: Path,
    desired: dict[str, Any],
    *,
    check: bool,
) -> str:
    if not path.exists():
        if check:
            raise PreparationError(f"required private config is missing: {path}")
        return "CREATED"
    existing = load_yaml_mapping(path)
    if existing == desired:
        return "UNCHANGED"
    if check:
        raise PreparationError(
            f"private config differs from the prepared contract: {path}; "
            "rerun without --check to replace it"
        )
    return "UPDATED"


def apply_yaml_plan(
    path: Path,
    desired: dict[str, Any],
    *,
    status: str,
) -> dict[str, str]:
    result = {"status": status, "path": str(path)}
    if status == "UNCHANGED":
        return result
    _atomic_write_yaml(path, desired)
    return result


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_camera_count < 1:
        raise PreparationError("--expected-camera-count must be positive")
    if args.runtime_only and args.target is not None:
        raise PreparationError("--runtime-only does not accept --target")
    if args.runtime_only and args.workspace_equals_target:
        raise PreparationError(
            "--runtime-only does not accept --workspace-equals-target"
        )
    if not args.runtime_only and args.target is None:
        raise PreparationError("full preparation requires --target")
    if not args.runtime_only and not args.workspace_equals_target:
        raise PreparationError(
            "fixed provisioning requires an explicit --workspace-equals-target acknowledgement"
        )
    cameras = load_identity_map(args.identity_map, args.expected_camera_count)
    target = None if args.runtime_only else validate_target_artifact(args.target)
    runtime_template = load_yaml_mapping(RUNTIME_TEMPLATE)
    provision_template = (
        None if args.runtime_only else load_yaml_mapping(PROVISION_TEMPLATE)
    )
    desired_configs: dict[str, dict[str, tuple[Path, dict[str, Any]]]] = {}
    plans: dict[str, dict[str, str]] = {}
    for label, serial in cameras.items():
        config_root = args.asset_root / label / "configs"
        runtime_path = config_root / "runtime.yaml"
        runtime = build_runtime_config(runtime_template, label, serial)
        desired_configs[label] = {"runtime": (runtime_path, runtime)}
        if not args.runtime_only:
            assert (
                provision_template is not None
                and target is not None
                and args.target is not None
            )
            provision_path = config_root / "fixed_provision.yaml"
            provision = build_provision_config(
                provision_template,
                label=label,
                serial=serial,
                config_path=provision_path,
                target_path=args.target,
                target_sha256=target["sha256"],
                target_frame=target["target_frame"],
            )
            desired_configs[label]["fixed_provision"] = (provision_path, provision)
        plans[label] = {
            name: plan_yaml(
                path,
                desired,
                check=args.check,
            )
            for name, (path, desired) in desired_configs[label].items()
        }
    prepared: dict[str, dict[str, dict[str, str]]] = {}
    for label, configs in desired_configs.items():
        prepared[label] = {
            name: apply_yaml_plan(
                path,
                desired,
                status=plans[label][name],
            )
            for name, (path, desired) in configs.items()
        }
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "passed": True,
        "mode": (
            "runtime_only_check"
            if args.runtime_only and args.check
            else "runtime_only_prepare"
            if args.runtime_only
            else "full_check"
            if args.check
            else "full_prepare"
        ),
        "camera_count": len(cameras),
        "camera_labels": list(cameras),
        "identity_map_sha256": sha256_file(args.identity_map),
        "target_spec_sha256": None if target is None else target["sha256"],
        "workspace_equals_target": not args.runtime_only,
        "prepared": prepared,
    }
    _atomic_write_json(args.report, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create or verify private CameraRig runtime/provision YAML from a confirmed "
            "camera identity map without printing serial numbers."
        )
    )
    parser.add_argument("--identity-map", type=Path, required=True)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--asset-root", type=Path, default=Path(".local/camera_rig"))
    parser.add_argument("--expected-camera-count", type=int, required=True)
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="prepare runtime YAML before an existing physical target has been registered",
    )
    parser.add_argument(
        "--workspace-equals-target",
        action="store_true",
        help="acknowledge that the fixed-provision workspace frame equals the target frame",
    )
    parser.add_argument(
        "--check", action="store_true", help="verify without changing configs"
    )
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except PreparationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    summary = {
        "passed": report["passed"],
        "mode": report["mode"],
        "camera_count": report["camera_count"],
        "camera_labels": report["camera_labels"],
    }
    if report["target_spec_sha256"] is not None:
        summary["target_spec_sha256"] = report["target_spec_sha256"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
