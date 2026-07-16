#!/usr/bin/env python3
"""Prepare trusted FFS assets, export ONNX, and build TensorRT engines."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "ffs_reproduction/artifacts"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ffs_reproduction"))

from builders.trt_builder import build_engine  # noqa: E402
from exporters.onnx_export import export_plugin, export_single, export_two_stage, validate_onnx_contract  # noqa: E402
from pointcloud_builder.ffs.manifest import sha256_file  # noqa: E402


def _portable_path(path: Path, metadata_root: Path) -> str:
    """Return a relocatable path for generated artifact metadata."""

    return Path(os.path.relpath(path.resolve(), metadata_root.resolve())).as_posix()


def _portable_text(value: str, metadata_root: Path) -> str:
    value = value.replace(str(metadata_root.resolve()), ".")
    return value.replace(str(Path.home()), "~")


def _artifact_record(path: Path, metadata_root: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": _portable_path(path, metadata_root),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def _copy_verified(source: Path, destination: Path, metadata_root: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source FFS asset is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    if destination.exists() and sha256_file(destination) != source_hash:
        raise ValueError(f"Existing artifact hash mismatch; refusing overwrite: {destination}")
    if not destination.exists():
        shutil.copy2(source, destination)
    return _artifact_record(destination, metadata_root)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_commit(source_root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()


def _upstream_commit(source_root: Path | None) -> str:
    if source_root is not None:
        return _git_commit(source_root)
    provenance_path = REPO_ROOT / "ffs_reproduction/UPSTREAM_SOURCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    commit = provenance.get("commit")
    if not isinstance(commit, str) or not commit:
        raise ValueError(f"Missing upstream commit in {provenance_path.relative_to(REPO_ROOT)}")
    return commit


def _check_onnx(path: Path, *, custom_plugin: bool = False) -> None:
    import onnx

    model = onnx.load(str(path))
    if custom_plugin:
        if not any(node.op_type == "FFSGWCVolume" for node in model.graph.node):
            raise ValueError(f"Plugin ONNX has no FFSGWCVolume node: {path}")
    else:
        onnx.checker.check_model(model)


def _route_manifest(
    *,
    backend: str,
    height: int,
    width: int,
    max_disp: int,
    valid_iters: int,
    precision: str,
    normalization_contract: str,
    artifacts: list[Path],
    io: dict[str, Any],
    builder_optimization_level: int,
    workspace_gib: float,
    artifact_id: str,
    metadata_root: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "backend": backend,
        "height": height,
        "width": width,
        "max_disp": max_disp,
        "valid_iters": valid_iters,
        "precision": precision,
        "artifact_id": artifact_id,
        "builder_optimization_level": int(builder_optimization_level),
        "workspace_gib": float(workspace_gib),
        "normalization_contract": normalization_contract,
        "input_names": io.get("input_names", []),
        "output_names": io.get("output_names", []),
        "onnx_contract": io.get("onnx_contract"),
        "build_status": "not_attempted",
        "parser_status": "not_attempted",
        "artifacts": [_artifact_record(path, metadata_root) for path in artifacts],
    }
    if extra:
        value.update(extra)
    return value


def _write_route_manifests(artifact_dir: Path, routes: dict[str, Any], artifact_id: str) -> dict[str, dict[str, Path]]:
    paths: dict[str, dict[str, Path]] = {}
    for backend in ("tensorrt_single", "tensorrt_two_stage", "tensorrt_plugin"):
        filename = f"{backend}_{artifact_id}.manifest.json"
        manifest_path = artifact_dir / filename
        yaml_path = artifact_dir / f"{backend}_{artifact_id}.yaml"
        routes[backend]["manifest_path"] = manifest_path.name
        routes[backend]["config_path"] = yaml_path.name
        _write_json(manifest_path, routes[backend])
        yaml_path.write_text(
            yaml.safe_dump(routes[backend], sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )
        paths[backend] = {"manifest": manifest_path, "config": yaml_path}
    return paths


def _check_generated_output_policy(artifact_dir: Path, names: list[str], *, force: bool) -> None:
    """Refuse stale derived artifacts unless replacement is explicit."""

    existing = [artifact_dir / name for name in names if (artifact_dir / name).exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "Derived FFS artifacts already exist; refusing to overwrite them. "
            f"Pass --force only when an explicit rebuild is intended: {joined}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Optional upstream checkout used only to import trusted assets; runtime never depends on it",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--valid-iters", type=int, default=8)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--builder-optimization-level", type=int, choices=range(6), default=3)
    parser.add_argument("--workspace-gib", type=float, default=8.0)
    parser.add_argument(
        "--artifact-suffix",
        default=None,
        help="Variant suffix; defaults to '<precision>_o<builder-optimization-level>'",
    )
    parser.add_argument("--skip-tensorrt", action="store_true", help="Only prepare weights and ONNX; never claim TRT routes built")
    parser.add_argument("--force", action="store_true", help="Explicitly replace existing derived ONNX/engine/manifests")
    args = parser.parse_args()
    if (args.height, args.width) != (480, 640):
        raise ValueError("The current artifact contract is fixed to height=480,width=640")
    if args.max_disp <= 0 or args.max_disp % 4:
        raise ValueError("--max-disp must be positive and divisible by 4")
    if args.workspace_gib <= 0.0:
        raise ValueError("--workspace-gib must be positive")
    artifact_id = args.artifact_suffix or f"{args.precision}_o{args.builder_optimization_level}"
    if not artifact_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in artifact_id):
        raise ValueError("--artifact-suffix may contain only letters, numbers, '_' and '-'")
    artifact_dir = args.artifact_dir.expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    source_root = args.source_root.expanduser().resolve() if args.source_root is not None else None
    default_source_dir = source_root / "weights/20-30-48" if source_root is not None else artifact_dir
    checkpoint = (args.checkpoint or default_source_dir / "model_best_bp2_serialize.pth").expanduser().resolve()
    model_config = (args.model_config or default_source_dir / "cfg.yaml").expanduser().resolve()
    def artifact_path(stem: str, extension: str) -> Path:
        return artifact_dir / f"{stem}_{artifact_id}{extension}"

    generated_names = [
        f"artifact_manifest_{artifact_id}.json",
        f"tensorrt_single_{artifact_id}.onnx",
        f"tensorrt_two_stage_feature_{artifact_id}.onnx",
        f"tensorrt_two_stage_post_{artifact_id}.onnx",
        f"tensorrt_plugin_{artifact_id}.onnx",
        f"tensorrt_single_{artifact_id}.engine",
        f"tensorrt_two_stage_feature_{artifact_id}.engine",
        f"tensorrt_two_stage_post_{artifact_id}.engine",
        f"tensorrt_plugin_{artifact_id}.engine",
        f"tensorrt_single_{artifact_id}.manifest.json",
        f"tensorrt_two_stage_{artifact_id}.manifest.json",
        f"tensorrt_plugin_{artifact_id}.manifest.json",
        f"tensorrt_single_{artifact_id}.yaml",
        f"tensorrt_two_stage_{artifact_id}.yaml",
        f"tensorrt_plugin_{artifact_id}.yaml",
        f"tensorrt_single_{artifact_id}.build_error.json",
        f"tensorrt_two_stage_feature_{artifact_id}.build_error.json",
        f"tensorrt_two_stage_post_{artifact_id}.build_error.json",
        f"tensorrt_plugin_{artifact_id}.build_error.json",
    ]
    _check_generated_output_policy(artifact_dir, generated_names, force=args.force)
    checkpoint_local = artifact_dir / "model_best_bp2_serialize.pth"
    config_local = artifact_dir / "cfg.yaml"
    checkpoint_asset = _copy_verified(checkpoint, checkpoint_local, artifact_dir)
    config_asset = _copy_verified(model_config, config_local, artifact_dir)
    upstream_commit = _upstream_commit(source_root)
    manifest: dict[str, Any] = {
        "asset_origin": "imported_source_root" if source_root is not None else "local_artifact",
        "upstream_commit": upstream_commit,
        "checkpoint": checkpoint_asset,
        "model_config": config_asset,
        "height": args.height,
        "width": args.width,
        "max_disp": args.max_disp,
        "valid_iters": args.valid_iters,
        "precision": args.precision,
        "artifact_id": artifact_id,
        "builder_optimization_level": args.builder_optimization_level,
        "workspace_gib": args.workspace_gib,
        "build_status": "onnx_only" if args.skip_tensorrt else "not_attempted",
        "checkpoint_config": yaml.safe_load(config_local.read_text(encoding="utf-8")),
        "routes": {},
    }
    artifact_manifest_path = artifact_dir / f"artifact_manifest_{artifact_id}.json"
    _write_json(artifact_manifest_path, manifest)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("FFS export/build requires CUDA in the dp3 environment")
    single_onnx = artifact_path("tensorrt_single", ".onnx")
    two_feature_onnx = artifact_path("tensorrt_two_stage_feature", ".onnx")
    two_post_onnx = artifact_path("tensorrt_two_stage_post", ".onnx")
    plugin_onnx = artifact_path("tensorrt_plugin", ".onnx")
    single_io = export_single(checkpoint_local, single_onnx, height=args.height, width=args.width, max_disp=args.max_disp, valid_iters=args.valid_iters, precision=args.precision, device=device)
    two_io = export_two_stage(checkpoint_local, two_feature_onnx, two_post_onnx, height=args.height, width=args.width, max_disp=args.max_disp, valid_iters=args.valid_iters, precision=args.precision, device=device)
    plugin_io = export_plugin(checkpoint_local, plugin_onnx, height=args.height, width=args.width, max_disp=args.max_disp, valid_iters=args.valid_iters, precision=args.precision, device=device)
    single_contract = validate_onnx_contract(
        single_onnx,
        input_names=("left_image", "right_image"),
        output_names=("disparity",),
    )
    feature_contract = validate_onnx_contract(
        two_feature_onnx,
        input_names=("left", "right"),
        output_names=tuple(two_io["feature_output_names"]),
    )
    for path in (single_onnx, two_feature_onnx, two_post_onnx):
        _check_onnx(path)
    plugin_contract = validate_onnx_contract(
        plugin_onnx,
        input_names=("left", "right"),
        output_names=("disp",),
        custom_plugin=True,
    )
    _check_onnx(plugin_onnx, custom_plugin=True)
    manifest["routes"]["tensorrt_single"] = _route_manifest(
        backend="tensorrt_single", height=args.height, width=args.width, max_disp=args.max_disp, valid_iters=args.valid_iters, precision=args.precision,
        normalization_contract=single_io["normalization_contract"], artifacts=[single_onnx], io={**single_io, "onnx_contract": single_contract},
        builder_optimization_level=args.builder_optimization_level, workspace_gib=args.workspace_gib, artifact_id=artifact_id,
        metadata_root=artifact_dir,
    )
    manifest["routes"]["tensorrt_two_stage"] = _route_manifest(
        backend="tensorrt_two_stage", height=args.height, width=args.width, max_disp=args.max_disp, valid_iters=args.valid_iters, precision=args.precision,
        normalization_contract=two_io["normalization_contract"], artifacts=[two_feature_onnx, two_post_onnx], io={**two_io, "onnx_contract": feature_contract},
        builder_optimization_level=args.builder_optimization_level, workspace_gib=args.workspace_gib, artifact_id=artifact_id,
        metadata_root=artifact_dir,
        extra={
            "cv_group": two_io["cv_group"],
            "gwc_normalize": two_io["gwc_normalize"],
            "feature_output_names": two_io["feature_output_names"],
            "post_input_names": two_io["post_input_names"],
            "post_output_names": two_io["post_output_names"],
        },
    )
    manifest["routes"]["tensorrt_plugin"] = _route_manifest(
        backend="tensorrt_plugin", height=args.height, width=args.width, max_disp=args.max_disp, valid_iters=args.valid_iters, precision=args.precision,
        normalization_contract=plugin_io["normalization_contract"], artifacts=[plugin_onnx], io={**plugin_io, "onnx_contract": plugin_contract},
        builder_optimization_level=args.builder_optimization_level, workspace_gib=args.workspace_gib, artifact_id=artifact_id,
        metadata_root=artifact_dir,
        extra={"cv_group": plugin_io["cv_group"], "gwc_normalize": plugin_io["gwc_normalize"]},
    )
    for route in manifest["routes"].values():
        route.update(
            {
                "upstream_commit": upstream_commit,
                "checkpoint": checkpoint_asset,
                "model_config": config_asset,
            }
        )
    route_paths = _write_route_manifests(artifact_dir, manifest["routes"], artifact_id)
    _write_json(artifact_manifest_path, manifest)
    if args.skip_tensorrt:
        print(json.dumps({"artifact_manifest": _portable_path(artifact_manifest_path, REPO_ROOT), "tensorrt": "skipped", "artifact_id": artifact_id}, indent=2))
        return 0

    single_engine = artifact_path("tensorrt_single", ".engine")
    feature_engine = artifact_path("tensorrt_two_stage_feature", ".engine")
    post_engine = artifact_path("tensorrt_two_stage_post", ".engine")
    plugin_engine = artifact_path("tensorrt_plugin", ".engine")
    single_error = artifact_path("tensorrt_single", ".build_error.json")
    feature_error = artifact_path("tensorrt_two_stage_feature", ".build_error.json")
    post_error = artifact_path("tensorrt_two_stage_post", ".build_error.json")
    plugin_error = artifact_path("tensorrt_plugin", ".build_error.json")
    fp16 = args.precision == "fp16"
    build_kwargs = {
        "fp16": fp16,
        "workspace_gib": args.workspace_gib,
        "builder_optimization_level": args.builder_optimization_level,
    }

    def attempt(
        route_name: str,
        onnx_path: Path,
        engine_path: Path,
        error_path: Path,
        *,
        stage_key: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        route = manifest["routes"][route_name]
        status_key = f"{stage_key}_build_status" if stage_key else "build_status"
        parser_key = f"{stage_key}_parser_status" if stage_key else "parser_status"
        try:
            result = build_engine(
                onnx_path,
                engine_path,
                **build_kwargs,
                config_path=route_paths[route_name]["config"],
                error_path=error_path,
                metadata_root=artifact_dir,
                **kwargs,
            )
        except Exception as exc:
            route.update(
                {
                    status_key: "failed",
                    parser_key: "unknown_or_failed",
                    f"{stage_key + '_' if stage_key else ''}build_error_path": _portable_path(error_path, artifact_dir),
                    f"{stage_key + '_' if stage_key else ''}build_error": _portable_text(str(exc), artifact_dir),
                }
            )
            return None
        route.update({status_key: "success", parser_key: "success"})
        if error_path.exists():
            error_path.unlink()
        return result

    single_build = attempt("tensorrt_single", single_onnx, single_engine, single_error)
    two_feature_build = attempt(
        "tensorrt_two_stage", two_feature_onnx, feature_engine, feature_error, stage_key="feature"
    )
    two_post_build = attempt(
        "tensorrt_two_stage", two_post_onnx, post_engine, post_error, stage_key="post"
    )
    two_route = manifest["routes"]["tensorrt_two_stage"]
    two_route["build_status"] = "success" if two_feature_build is not None and two_post_build is not None else "failed"
    two_route["parser_status"] = "success" if two_route["build_status"] == "success" else "partial_or_failed"
    plugin_library = REPO_ROOT / "ffs_reproduction/build/libffs_gwc_plugin.so"
    if plugin_library.is_file():
        plugin_build = attempt(
            "tensorrt_plugin", plugin_onnx, plugin_engine, plugin_error, plugin_library=plugin_library
        )
    else:
        plugin_build = None
        manifest["routes"]["tensorrt_plugin"].update(
            {
                "build_status": "failed",
                "parser_status": "not_attempted",
                "build_error": "Build the SM120 plugin before plugin engine creation: ../build/libffs_gwc_plugin.so",
            }
        )

    single_route = manifest["routes"]["tensorrt_single"]
    if single_build is not None:
        single_route["engine"] = single_build
        single_route["artifacts"].append(_artifact_record(single_engine, artifact_dir))
    if two_feature_build is not None:
        two_route["feature_engine"] = two_feature_build
        two_route["artifacts"].append(_artifact_record(feature_engine, artifact_dir))
    if two_post_build is not None:
        two_route["post_engine"] = two_post_build
        two_route["artifacts"].append(_artifact_record(post_engine, artifact_dir))
    plugin_route = manifest["routes"]["tensorrt_plugin"]
    if plugin_build is not None:
        plugin_route.update({"engine": plugin_build, "plugin_library_sha256": sha256_file(plugin_library)})
        plugin_route["artifacts"].extend(
            [
                _artifact_record(plugin_engine, artifact_dir),
                _artifact_record(plugin_library, artifact_dir),
            ]
        )
    _write_route_manifests(artifact_dir, manifest["routes"], artifact_id)
    successful = all(
        route.get("build_status") == "success" for route in manifest["routes"].values()
    )
    manifest["build_status"] = "success" if successful else "partial_failure"
    _write_json(artifact_manifest_path, manifest)
    result = {"artifact_manifest": _portable_path(artifact_manifest_path, REPO_ROOT), "tensorrt": "built" if successful else "partial_failure", "artifact_id": artifact_id, "routes": manifest["routes"]}
    print(json.dumps(result, indent=2, default=str))
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
