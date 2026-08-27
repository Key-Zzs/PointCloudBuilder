#!/usr/bin/env python3
"""Check or build the documented private FFS asset bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from pointcloud_builder.ffs.manifest import load_manifest, sha256_file


CHECKPOINT_SHA256 = "98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692"
MODEL_CONFIG_SHA256 = "d45afe99b176454d5aff416edf16c8da6a99579f8f374b927f37907442a7d6bc"
REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--build-tensorrt", action="store_true")
    parser.add_argument("--asset-root", default=".local/ffs")
    parser.add_argument("--checkpoint")
    parser.add_argument("--model-config")
    parser.add_argument("--tensorrt-root")
    parser.add_argument("--cuda-arch", default="120")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    root = _private_asset_root(args.asset_root)
    if args.build_tensorrt:
        if not args.checkpoint or not args.model_config or not args.tensorrt_root:
            raise ValueError(
                "--build-tensorrt requires --checkpoint, --model-config, and --tensorrt-root"
            )
        _build(root, args)
    report = check_assets(root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def check_assets(root: Path) -> dict[str, Any]:
    artifacts = root / "artifacts"
    plugin = root / "build/libffs_gwc_plugin.so"
    checks: dict[str, dict[str, Any]] = {}

    def checksum(name: str, path: Path, expected: str) -> None:
        actual = sha256_file(path) if path.is_file() else None
        checks[name] = {
            "expected_name": path.name,
            "present": path.is_file(),
            "sha256_matches": actual == expected,
            "passed": actual == expected,
        }

    checksum(
        "checkpoint", artifacts / "model_best_bp2_serialize.pth", CHECKPOINT_SHA256
    )
    checksum("model_config", artifacts / "cfg.yaml", MODEL_CONFIG_SHA256)
    routes = {
        "tensorrt_single": (artifacts / "tensorrt_single_fp16_o3.engine",),
        "tensorrt_two_stage": (
            artifacts / "tensorrt_two_stage_feature_fp16_o3.engine",
            artifacts / "tensorrt_two_stage_post_fp16_o3.engine",
        ),
        "tensorrt_plugin": (
            artifacts / "tensorrt_plugin_fp16_o3.engine",
            plugin,
        ),
    }
    for backend, required in routes.items():
        manifest_path = artifacts / f"{backend}_fp16_o3.manifest.json"
        config_path = artifacts / f"{backend}_fp16_o3.yaml"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            load_manifest(
                manifest_path,
                backend=backend,
                height=480,
                width=640,
                max_disp=192,
                valid_iters=8,
                precision="fp16",
                normalization_contract=str(manifest["normalization_contract"]),
                artifact_paths=required,
                input_names=manifest.get("input_names", ()),
                output_names=manifest.get("output_names", ()),
                config_path=config_path,
                builder_optimization_level=3,
                workspace_gib=8.0,
            )
        except Exception as error:
            checks[backend] = {
                "passed": False,
                "error": f"{type(error).__name__}: {str(error)[:300]}",
            }
        else:
            checks[backend] = {
                "passed": True,
                "manifest": manifest_path.name,
                "artifacts": [path.name for path in required],
            }
    return {
        "schema_version": "pointcloud-builder.ffs-assets-check.v1",
        "checks": checks,
        "passed": all(item["passed"] for item in checks.values()),
    }


def _build(root: Path, args: argparse.Namespace) -> None:
    root.mkdir(parents=True, exist_ok=True)
    plugin_command = [
        sys.executable,
        str(REPO_ROOT / "scripts/build_ffs_plugin.py"),
        "--tensorrt-root",
        str(Path(args.tensorrt_root).expanduser().resolve()),
        "--build-dir",
        str(root / "build"),
        "--cuda-arch",
        args.cuda_arch,
    ]
    subprocess.run(plugin_command, check=True, cwd=REPO_ROOT)
    prepare_command = [
        sys.executable,
        str(REPO_ROOT / "scripts/prepare_ffs_artifacts.py"),
        "--checkpoint",
        str(Path(args.checkpoint).expanduser().resolve()),
        "--model-config",
        str(Path(args.model_config).expanduser().resolve()),
        "--artifact-dir",
        str(root / "artifacts"),
        "--plugin-library",
        str(root / "build/libffs_gwc_plugin.so"),
        "--precision",
        "fp16",
        "--builder-optimization-level",
        "3",
        "--workspace-gib",
        "8",
        "--artifact-suffix",
        "fp16_o3",
    ]
    if args.force:
        prepare_command.append("--force")
    subprocess.run(prepare_command, check=True, cwd=REPO_ROOT)


def _private_asset_root(value: str) -> Path:
    root = Path(value).resolve()
    local = (Path.cwd() / ".local").resolve()
    if not root.is_relative_to(local):
        raise ValueError("FFS assets must be stored under .local/")
    return root


if __name__ == "__main__":
    raise SystemExit(main())
