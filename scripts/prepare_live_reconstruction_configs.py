#!/usr/bin/env python3
"""Generate private mapping and live-rig configs from validated local artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from pointcloud_builder.config import load_config
from pointcloud_builder.mapping.config import load_tsdf_config
from pointcloud_builder.rig import load_rig_config

REPO_ROOT = Path(__file__).resolve().parents[1]
MAPPING_TEMPLATE = REPO_ROOT / "configs/mapping/native_workspace_example.yaml"
PROFILE_TEMPLATES = {
    "live_rig_ffs_rgb.yaml": REPO_ROOT
    / "configs/mapping/dense_rgb_reconstruction_example.yaml",
    "live_rig_ffs_rgb_raw.yaml": REPO_ROOT
    / "configs/mapping/raw_rgb_concatenation_example.yaml",
    "live_rig_ffs_rgb_compact.yaml": REPO_ROOT
    / "configs/mapping/compact_rgb_reconstruction_example.yaml",
}
TSDF_TEMPLATE = REPO_ROOT / "configs/mapping/tsdf_example.yaml"
OUTPUT_NAMES = (
    "mapping.yaml",
    "mapping_acceptance.yaml",
    *PROFILE_TEMPLATES,
    "tsdf_frozen_ffs.yaml",
)
IDENTITY_SCHEMA = "pointcloud-builder.camera-identity-map.v1"
CAMERA_NAME = re.compile(r"camera_[a-z][a-z0-9_]*\Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Git-ignored workspace mapping and live FFS rig configs without "
            "embedding serial numbers."
        )
    )
    parser.add_argument("--identity-map", required=True)
    parser.add_argument("--camera-rig-root", default=".local/camera_rig")
    parser.add_argument(
        "--ffs-config", default=".local/configs/ffs_tensorrt_plugin_rgb.yaml"
    )
    parser.add_argument("--output-dir", default=".local/configs")
    parser.add_argument("--expected-camera-count", type=int, default=3)
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Explicitly replace differing generated configs, including mapping.yaml; "
            "normally preserve an existing workspace-specific mapping.yaml."
        ),
    )
    args = parser.parse_args(argv)

    identity_path = _private_path(args.identity_map, label="identity map")
    camera_rig_root = _private_path(args.camera_rig_root, label="CameraRig root")
    ffs_config = _private_path(args.ffs_config, label="FFS config")
    output_dir = _private_path(args.output_dir, label="output directory")
    camera_names = load_camera_names(identity_path, args.expected_camera_count)
    workspace_frame = validate_private_inputs(
        camera_names,
        camera_rig_root=camera_rig_root,
        ffs_config=ffs_config,
    )

    existing_mapping = output_dir / "mapping.yaml"
    if existing_mapping.is_file() and not args.force:
        mapping = _load_yaml(existing_mapping)
        validate_mapping(mapping, workspace_frame)
    else:
        mapping = _load_yaml(MAPPING_TEMPLATE)
        mapping["expected_plane"]["frame"] = workspace_frame

    configs = render_configs(
        camera_names,
        camera_rig_root=camera_rig_root,
        ffs_config=ffs_config,
        output_dir=output_dir,
        workspace_frame=workspace_frame,
        mapping=mapping,
    )
    statuses = write_configs(configs, output_dir=output_dir, force=args.force)
    print(
        json.dumps(
            {
                "schema_version": "pointcloud-builder.live-reconstruction-configs.v1",
                "status": "PASS",
                "camera_names": camera_names,
                "workspace_frame": workspace_frame,
                "files": statuses,
                "rig_calibration": "omitted_until_validated_multipose_promotion",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def load_camera_names(path: Path, expected_count: int) -> list[str]:
    if expected_count < 1:
        raise ValueError("expected camera count must be positive")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load identity map {path}: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != IDENTITY_SCHEMA:
        raise ValueError(f"identity map must use schema {IDENTITY_SCHEMA}")
    names = sorted(name for name in value if CAMERA_NAME.fullmatch(name))
    if len(names) != expected_count:
        raise ValueError(
            f"identity map has {len(names)} camera names; expected {expected_count}"
        )
    return names


def validate_private_inputs(
    camera_names: list[str], *, camera_rig_root: Path, ffs_config: Path
) -> str:
    from camera_rig.api import load_camera_config, load_provisioned_camera_bundle

    pipeline = load_config(ffs_config)
    ffs = pipeline.depth_source.ffs
    if (
        ffs is None
        or ffs.backend != "tensorrt_plugin"
        or not pipeline.pointcloud.use_rgb
        or pipeline.pointcloud.output_format != "xyzrgb"
    ):
        raise ValueError(
            "live FFS config must use tensorrt_plugin with pointcloud XYZRGB enabled"
        )

    workspace_frames: set[str] = set()
    for camera_name in camera_names:
        runtime_path = camera_rig_root / camera_name / "configs/runtime.yaml"
        provision_path = camera_rig_root / camera_name / "provision"
        runtime = load_camera_config(runtime_path)
        bundle = load_provisioned_camera_bundle(provision_path)
        if runtime.camera.name != camera_name or bundle.device.camera_name != camera_name:
            raise ValueError(f"{camera_name}: runtime/provision logical identity mismatch")
        if runtime.camera.serial != bundle.device.serial:
            raise ValueError(f"{camera_name}: runtime/provision serial identity mismatch")
        if bundle.status != "passed" or bundle.fixed_mount_calibration is None:
            raise ValueError(f"{camera_name}: provision bundle has not passed")
        workspace_frames.add(bundle.fixed_mount_calibration.parent_frame)
    if len(workspace_frames) != 1:
        raise ValueError("camera provision bundles do not share one workspace frame")
    return next(iter(workspace_frames))


def render_configs(
    camera_names: list[str],
    *,
    camera_rig_root: Path,
    ffs_config: Path,
    output_dir: Path,
    workspace_frame: str,
    mapping: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    validate_mapping(mapping, workspace_frame)
    rendered = {
        "mapping.yaml": deepcopy(mapping),
        "mapping_acceptance.yaml": deepcopy(mapping),
        "tsdf_frozen_ffs.yaml": _load_yaml(TSDF_TEMPLATE),
    }
    for output_name, template_path in PROFILE_TEMPLATES.items():
        config = _load_yaml(template_path)
        prototype = config["cameras"][0]
        cameras = []
        for camera_name in camera_names:
            camera = deepcopy(prototype)
            camera["name"] = camera_name
            camera["source"] = {
                "type": "camera_rig_live",
                "camera_config": _portable_path(
                    camera_rig_root / camera_name / "configs/runtime.yaml",
                    output_dir,
                ),
                "provision_artifact": _portable_path(
                    camera_rig_root / camera_name / "provision", output_dir
                ),
            }
            camera["depth"] = {"mode": "ffs_stereo"}
            camera["pointcloud"] = {"use_rgb": True}
            camera["pipeline_config"] = _portable_path(ffs_config, output_dir)
            camera["local_crop"] = {"enabled": False}
            cameras.append(camera)
        config["output_frame"] = workspace_frame
        config["cameras"] = cameras
        config["timing"]["reference_camera"] = camera_names[0]
        config.pop("rig_calibration", None)
        rendered[output_name] = config
    return rendered


def write_configs(
    configs: dict[str, dict[str, Any]], *, output_dir: Path, force: bool
) -> dict[str, str]:
    if set(configs) != set(OUTPUT_NAMES):
        raise ValueError("generated live reconstruction config set is incomplete")
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, str] = {}
    staged: list[tuple[Path, Path]] = []
    try:
        for name in OUTPUT_NAMES:
            destination = output_dir / name
            if destination.is_file() and not force:
                existing = _load_yaml(destination)
                if name == "mapping.yaml":
                    statuses[name] = "preserved_existing_workspace_mapping"
                    continue
                if existing == configs[name]:
                    statuses[name] = "unchanged"
                    continue
                raise FileExistsError(
                    f"private config already exists with different content: {destination}; "
                    "inspect it before using --force"
                )
            temporary = output_dir / f".{name}.tmp"
            temporary.write_text(
                yaml.safe_dump(configs[name], sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            temporary.chmod(0o600)
            if name.startswith("live_rig_"):
                load_rig_config(temporary)
            elif name.startswith("tsdf_"):
                load_tsdf_config(temporary)
            else:
                validate_mapping(_load_yaml(temporary), configs[name]["expected_plane"]["frame"])
            staged.append((temporary, destination))
            statuses[name] = "written"
        for temporary, destination in staged:
            os.replace(temporary, destination)
            destination.chmod(0o600)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    return statuses


def validate_mapping(value: dict[str, Any], workspace_frame: str) -> None:
    plane = value.get("expected_plane")
    if not isinstance(plane, dict):
        raise TypeError("mapping config must contain expected_plane")
    if plane.get("frame") != workspace_frame:
        raise ValueError("mapping expected_plane.frame must match provision workspace")
    for key in ("x", "y", "z_search_range_m"):
        bounds = plane.get(key)
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or float(bounds[0]) > float(bounds[1])
        ):
            raise ValueError(f"mapping expected_plane.{key} must be an ordered pair")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"could not load YAML {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return value


def _portable_path(path: Path, declaring_dir: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start=declaring_dir.resolve())).as_posix()


def _private_path(value: str, *, label: str) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    private_root = (REPO_ROOT / ".local").resolve()
    if not resolved.is_relative_to(private_root):
        raise ValueError(f"{label} must be stored under .local/")
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
