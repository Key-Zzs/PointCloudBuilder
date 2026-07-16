from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from pointcloud_builder.ffs.factory import BACKEND_NAMES, create_backend
from pointcloud_builder.ffs.manifest import load_manifest, sha256_file
from pointcloud_builder.ffs.preprocessing import normalize_disparity_output, prepare_ir_batch
from pointcloud_builder.ffs.vendor_loader import scoped_vendor_imports, vendor_root
from scripts.prepare_ffs_artifacts import _portable_path, _upstream_commit


def _write_minimal_calibration(path):
    intrinsics = {
        "width": 640,
        "height": 480,
        "fx": 392.5,
        "fy": 392.5,
        "cx": 316.8,
        "cy": 235.8,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cameras": {
                    "head_rgb": {
                        "streams": {"color": {"intrinsics": intrinsics}, "infrared1": {"intrinsics": intrinsics}},
                        "extrinsics": {
                            "infrared1_to_color": {
                                "rotation_matrix_row_major": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                "translation_m": [0, 0, 0],
                            }
                        },
                        "baseline": {"recommended_baseline_m": 0.05},
                    }
                }
            }
        )
    )


def test_all_four_backend_names_are_fixed() -> None:
    assert BACKEND_NAMES == ("pytorch", "tensorrt_single", "tensorrt_two_stage", "tensorrt_plugin")


def test_artifact_metadata_is_relocatable_and_provenance_is_local(tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    assert _portable_path(artifact_dir / "route.engine", artifact_dir) == "route.engine"
    assert _portable_path(tmp_path / "build/plugin.so", artifact_dir) == "../build/plugin.so"
    commit = _upstream_commit(None)
    assert len(commit) == 40 and all(character in "0123456789abcdef" for character in commit)


def test_cli_artifact_variant_overrides_yaml_metadata(tmp_path) -> None:
    from scripts.run_v05_ffs_frame import _resolve_config

    config_path = tmp_path / "base.yaml"
    config_path.write_text(
        """
device: cuda
camera:
  name: head
  depth_scale: 0.001
  aligned_depth_to_color: false
  color_intrinsics: {width: 640, height: 480, fx: 100, fy: 100, cx: 320, cy: 240}
  depth_intrinsics: {width: 640, height: 480, fx: 100, fy: 100, cx: 320, cy: 240}
pointcloud: {use_rgb: false, output_format: xyz}
depth_source:
  mode: ffs_stereo
  ffs:
    backend: pytorch
    artifact_id: fp16_o3
    precision: fp16
    builder_optimization_level: 3
    workspace_gib: 8.0
    baseline_m: 0.05
"""
    )
    _write_minimal_calibration(tmp_path / "dataset/meta/realsense_calibration.json")
    config = _resolve_config(
        config_path,
        tmp_path / "dataset",
        "head",
        "pytorch",
        artifact_id="fp32_o0",
        precision="fp32",
        builder_optimization_level=0,
        workspace_gib=2.0,
    )
    ffs = config.depth_source.ffs
    assert ffs is not None
    assert ffs.artifact_id == "fp32_o0"
    assert ffs.precision == "fp32"
    assert ffs.builder_optimization_level == 0
    assert ffs.workspace_gib == 2.0


def test_v05_config_resolves_repository_local_model_assets(tmp_path) -> None:
    from scripts.run_v05_ffs_frame import _resolve_config

    dataset = tmp_path / "dataset"
    _write_minimal_calibration(dataset / "meta/realsense_calibration.json")
    config = _resolve_config(
        Path("ffs_reproduction/configs/v05_ffs.yaml").resolve(),
        dataset,
        "head",
        "pytorch",
    )
    ffs = config.depth_source.ffs
    assert ffs is not None
    artifact_dir = Path("ffs_reproduction/artifacts").resolve()
    assert Path(ffs.checkpoint_path) == artifact_dir / "model_best_bp2_serialize.pth"
    assert Path(ffs.model_config_path) == artifact_dir / "cfg.yaml"


def test_grayscale_ir_expands_to_three_channels() -> None:
    image = prepare_ir_batch(torch.ones((480, 640), dtype=torch.uint8), name="left_ir", height=480, width=640, device=torch.device("cpu"))
    assert tuple(image.shape) == (1, 3, 480, 640)
    assert image.dtype == torch.float32
    assert torch.equal(image[:, 0], image[:, 1]) and torch.equal(image[:, 1], image[:, 2])


def test_backend_assets_fail_fast_without_fallback() -> None:
    base = {
        "height": 480,
        "width": 640,
        "max_disp": 192,
        "valid_iters": 8,
        "precision": "fp16",
        "manifest_path": None,
        "engine_path": None,
        "feature_engine_path": None,
        "post_engine_path": None,
        "plugin_library_path": None,
        "checkpoint_path": None,
    }
    for backend in BACKEND_NAMES:
        config = type("Config", (), {**base, "backend": backend})()
        with pytest.raises((ValueError, RuntimeError), match="requires"):
            create_backend(config, device=torch.device("cuda"))


def test_manifest_mismatch_is_rejected(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "backend": "tensorrt_single",
                "height": 480,
                "width": 640,
                "max_disp": 416,
                "valid_iters": 8,
                "precision": "fp16",
                "normalization_contract": "external_imagenet_0_255",
                "input_names": ["left_image", "right_image"],
                "output_names": ["disparity"],
            }
        )
    )
    with pytest.raises(ValueError, match="max_disp"):
        load_manifest(
            manifest,
            backend="tensorrt_single",
            height=480,
            width=640,
            max_disp=192,
            valid_iters=8,
            precision="fp16",
            normalization_contract="external_imagenet_0_255",
            input_names=("left_image", "right_image"),
            output_names=("disparity",),
        )


def test_manifest_engine_precision_mismatch_is_rejected(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "backend": "tensorrt_single",
                "height": 480,
                "width": 640,
                "max_disp": 192,
                "valid_iters": 8,
                "precision": "fp16",
                "normalization_contract": "external_imagenet_0_255",
                "input_names": ["left_image", "right_image"],
                "output_names": ["disparity"],
                "engine": {"fp16": False},
            }
        )
    )
    with pytest.raises(ValueError, match="engine.fp16"):
        load_manifest(
            manifest,
            backend="tensorrt_single",
            height=480,
            width=640,
            max_disp=192,
            valid_iters=8,
            precision="fp16",
            normalization_contract="external_imagenet_0_255",
            input_names=("left_image", "right_image"),
            output_names=("disparity",),
        )


def test_vendor_import_context_does_not_pollute_global_modules() -> None:
    before_path = list(sys.path)
    before = {name for name in sys.modules if name == "core" or name.startswith("core.") or name == "Utils" or name.startswith("foundation_stereo_ori")}
    with scoped_vendor_imports(vendor_root()):
        importlib.import_module("core.foundation_stereo")
    after = {name for name in sys.modules if name == "core" or name.startswith("core.") or name == "Utils" or name.startswith("foundation_stereo_ori")}
    assert after == before
    assert sys.path == before_path


def test_plain_import_does_not_load_heavy_optional_modules() -> None:
    code = "import sys; import pointcloud_builder; print([x for x in sys.modules if x == 'tensorrt' or x.startswith('onnx') or x.startswith('open3d') or x == 'core'])"
    output = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert "tensorrt" not in output and "open3d" not in output and "core" not in output


def test_disparity_output_contract_is_full_resolution() -> None:
    value = normalize_disparity_output(torch.ones((1, 1, 480, 640), dtype=torch.float16), height=480, width=640, device=torch.device("cpu"))
    assert tuple(value.shape) == (480, 640)
    assert value.dtype == torch.float32


def _write_route_contract(path, *, precision="fp32", max_disp=192, valid_iters=8):
    path.write_text(
        json.dumps(
            {
                "backend": "tensorrt_single",
                "height": 480,
                "width": 640,
                "max_disp": max_disp,
                "valid_iters": valid_iters,
                "precision": precision,
                "normalization_contract": "external_imagenet_0_255",
                "input_names": ["left_image", "right_image"],
                "output_names": ["disparity"],
            }
        )
        + "\n"
    )


def _write_engine_manifest(path, engine, *, config_path=None, precision="fp32", max_disp=192, valid_iters=8):
    value = {
        "backend": "tensorrt_single",
        "height": 480,
        "width": 640,
        "max_disp": max_disp,
        "valid_iters": valid_iters,
        "precision": precision,
        "normalization_contract": "external_imagenet_0_255",
        "input_names": ["left_image", "right_image"],
        "output_names": ["disparity"],
        "artifacts": [{"path": str(engine), "sha256": sha256_file(engine), "size": engine.stat().st_size}],
    }
    if config_path is not None:
        value["config_path"] = str(config_path)
    path.write_text(json.dumps(value) + "\n")


def test_engine_config_is_resolved_by_explicit_basename_and_contract(tmp_path) -> None:
    engine = tmp_path / "fast_foundationstereo_fp32_o0.engine"
    config = tmp_path / "fast_foundationstereo_fp32_o0.yaml"
    manifest = tmp_path / "route.manifest.json"
    engine.write_bytes(b"engine")
    _write_route_contract(config)
    _write_engine_manifest(manifest, engine, config_path=config)
    loaded = load_manifest(
        manifest,
        backend="tensorrt_single",
        height=480,
        width=640,
        max_disp=192,
        valid_iters=8,
        precision="fp32",
        normalization_contract="external_imagenet_0_255",
        artifact_paths=(engine,),
        input_names=("left_image", "right_image"),
        output_names=("disparity",),
    )
    assert loaded["resolved_config_path"] == str(config.resolve())


def test_engine_config_missing_or_ambiguous_fails_fast(tmp_path) -> None:
    engine = tmp_path / "unusual.engine"
    manifest = tmp_path / "route.manifest.json"
    engine.write_bytes(b"engine")
    _write_engine_manifest(manifest, engine)
    with pytest.raises(FileNotFoundError, match="No FFS config"):
        load_manifest(
            manifest,
            backend="tensorrt_single",
            height=480,
            width=640,
            max_disp=192,
            valid_iters=8,
            precision="fp32",
            normalization_contract="external_imagenet_0_255",
            artifact_paths=(engine,),
            input_names=("left_image", "right_image"),
            output_names=("disparity",),
        )
    (tmp_path / "one.yaml").write_text("backend: tensorrt_single\n")
    (tmp_path / "two.yaml").write_text("backend: tensorrt_single\n")
    with pytest.raises(ValueError, match="ambiguous"):
        load_manifest(
            manifest,
            backend="tensorrt_single",
            height=480,
            width=640,
            max_disp=192,
            valid_iters=8,
            precision="fp32",
            normalization_contract="external_imagenet_0_255",
            artifact_paths=(engine,),
            input_names=("left_image", "right_image"),
            output_names=("disparity",),
        )


def test_engine_config_contract_and_hash_mismatch_fail(tmp_path) -> None:
    engine = tmp_path / "route.engine"
    config = tmp_path / "route.yaml"
    manifest = tmp_path / "route.manifest.json"
    engine.write_bytes(b"engine")
    _write_route_contract(config, precision="fp16")
    _write_engine_manifest(manifest, engine, config_path=config, precision="fp32")
    with pytest.raises(ValueError, match="manifest/config mismatch|engine/config mismatch"):
        load_manifest(
            manifest,
            backend="tensorrt_single",
            height=480,
            width=640,
            max_disp=192,
            valid_iters=8,
            precision="fp32",
            normalization_contract="external_imagenet_0_255",
            artifact_paths=(engine,),
            input_names=("left_image", "right_image"),
            output_names=("disparity",),
        )
    _write_route_contract(config)
    _write_engine_manifest(manifest, engine, config_path=config)
    engine.write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA-256"):
        load_manifest(
            manifest,
            backend="tensorrt_single",
            height=480,
            width=640,
            max_disp=192,
            valid_iters=8,
            precision="fp32",
            normalization_contract="external_imagenet_0_255",
            artifact_paths=(engine,),
            input_names=("left_image", "right_image"),
            output_names=("disparity",),
        )
