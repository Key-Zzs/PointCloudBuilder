#!/usr/bin/env python3
"""Generate private FFS smoke and live-RGB pipeline configurations."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any

import yaml

from pointcloud_builder.config import load_config

from prepare_ffs_assets import check_assets


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "configs/mapping/ffs_workspace_example.yaml"
OUTPUT_NAMES = (
    "ffs_pytorch.yaml",
    "ffs_tensorrt_single.yaml",
    "ffs_tensorrt_two_stage.yaml",
    "ffs_tensorrt_plugin.yaml",
    "ffs_tensorrt_plugin_rgb.yaml",
)
ASSET_KEYS = {
    "checkpoint_path",
    "model_config_path",
    "engine_path",
    "feature_engine_path",
    "post_engine_path",
    "plugin_library_path",
    "manifest_path",
    "config_path",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Git-ignored FFS pipeline YAML files from checked assets."
    )
    parser.add_argument("--asset-root", default=".local/ffs")
    parser.add_argument("--output-dir", default=".local/configs")
    parser.add_argument("--camera-name", default="camera_a")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace existing generated private pipeline YAML files.",
    )
    args = parser.parse_args(argv)

    asset_root = _private_path(args.asset_root, label="asset root")
    output_dir = _private_path(args.output_dir, label="output directory")
    report = check_assets(asset_root)
    if not report["passed"]:
        failed = sorted(
            name for name, result in report["checks"].items() if not result["passed"]
        )
        raise RuntimeError(
            "FFS assets must pass prepare_ffs_assets.py --check before pipeline "
            f"configuration generation; failed={failed}"
        )

    configs = render_configs(
        _load_template(TEMPLATE),
        asset_root=asset_root,
        output_dir=output_dir,
        camera_name=args.camera_name,
    )
    written = write_configs(configs, output_dir=output_dir, force=args.force)
    print(
        json.dumps(
            {
                "schema_version": "pointcloud-builder.ffs-pipeline-configs.v1",
                "status": "PASS",
                "files": [str(path.relative_to(REPO_ROOT)) for path in written],
                "standalone_smoke_use_rgb": False,
                "live_rgb_config": "ffs_tensorrt_plugin_rgb.yaml",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def render_configs(
    template: dict[str, Any],
    *,
    asset_root: Path,
    output_dir: Path,
    camera_name: str,
) -> dict[str, dict[str, Any]]:
    if not camera_name.strip():
        raise ValueError("camera name must not be empty")
    base = deepcopy(template)
    base["camera"]["name"] = camera_name
    base["pointcloud"]["use_rgb"] = False
    base["pointcloud"]["output_format"] = "xyz"
    ffs = base["depth_source"]["ffs"]
    for key in ASSET_KEYS:
        ffs.pop(key, None)
    ffs.update(
        {
            "artifact_id": "fp16_o3",
            "width": 640,
            "height": 480,
            "max_disp": 192,
            "valid_iters": 8,
            "precision": "fp16",
            "builder_optimization_level": 3,
            "workspace_gib": 8,
        }
    )

    artifact_dir = asset_root / "artifacts"
    route_paths = {
        "pytorch": {
            "checkpoint_path": artifact_dir / "model_best_bp2_serialize.pth",
            "model_config_path": artifact_dir / "cfg.yaml",
        },
        "tensorrt_single": {
            "engine_path": artifact_dir / "tensorrt_single_fp16_o3.engine",
            "manifest_path": artifact_dir
            / "tensorrt_single_fp16_o3.manifest.json",
            "config_path": artifact_dir / "tensorrt_single_fp16_o3.yaml",
        },
        "tensorrt_two_stage": {
            "feature_engine_path": artifact_dir
            / "tensorrt_two_stage_feature_fp16_o3.engine",
            "post_engine_path": artifact_dir
            / "tensorrt_two_stage_post_fp16_o3.engine",
            "manifest_path": artifact_dir
            / "tensorrt_two_stage_fp16_o3.manifest.json",
            "config_path": artifact_dir / "tensorrt_two_stage_fp16_o3.yaml",
        },
        "tensorrt_plugin": {
            "engine_path": artifact_dir / "tensorrt_plugin_fp16_o3.engine",
            "plugin_library_path": asset_root / "build/libffs_gwc_plugin.so",
            "manifest_path": artifact_dir
            / "tensorrt_plugin_fp16_o3.manifest.json",
            "config_path": artifact_dir / "tensorrt_plugin_fp16_o3.yaml",
        },
    }

    rendered: dict[str, dict[str, Any]] = {}
    for backend, paths in route_paths.items():
        config = deepcopy(base)
        route = config["depth_source"]["ffs"]
        route["backend"] = backend
        route.update(
            {
                key: _portable_path(path, output_dir)
                for key, path in paths.items()
            }
        )
        rendered[f"ffs_{backend}.yaml"] = config

    live_rgb = deepcopy(rendered["ffs_tensorrt_plugin.yaml"])
    live_rgb["pointcloud"]["use_rgb"] = True
    live_rgb["pointcloud"]["output_format"] = "xyzrgb"
    rendered["ffs_tensorrt_plugin_rgb.yaml"] = live_rgb
    return rendered


def write_configs(
    configs: dict[str, dict[str, Any]], *, output_dir: Path, force: bool
) -> list[Path]:
    unexpected = sorted(set(configs) - set(OUTPUT_NAMES))
    missing = sorted(set(OUTPUT_NAMES) - set(configs))
    if unexpected or missing:
        raise ValueError(
            f"generated FFS config set mismatch; missing={missing}, unexpected={unexpected}"
        )
    existing = [output_dir / name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if existing and not force:
        raise FileExistsError(
            "FFS pipeline configs already exist; refusing to overwrite them. Pass "
            "--force only when explicit regeneration is intended: "
            + ", ".join(str(path) for path in existing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary: list[tuple[Path, Path]] = []
    try:
        for name in OUTPUT_NAMES:
            destination = output_dir / name
            staged = output_dir / f".{name}.tmp"
            staged.write_text(
                yaml.safe_dump(configs[name], sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            staged.chmod(0o600)
            load_config(staged)
            temporary.append((staged, destination))
        for staged, destination in temporary:
            os.replace(staged, destination)
            destination.chmod(0o600)
    finally:
        for staged, _ in temporary:
            staged.unlink(missing_ok=True)
    return [output_dir / name for name in OUTPUT_NAMES]


def _load_template(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"FFS template must be a YAML mapping: {path}")
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
