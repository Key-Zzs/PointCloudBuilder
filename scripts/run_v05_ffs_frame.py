#!/usr/bin/env python3
"""Run one read-only v05 raw-sidecar frame through one FFS backend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTER_REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(OUTER_REPO_ROOT / "tools"))

import lerobot_rgbd_source  # noqa: E402
from pointcloud_builder import PointCloudBuilder  # noqa: E402
from pointcloud_builder.config import parse_config  # noqa: E402
from pointcloud_builder.ffs.calibration import load_realsense_calibration  # noqa: E402
from visualize_ffs_stereo_pipeline import save_pipeline_artifacts  # noqa: E402


CAMERA_KEYS = {"head": "head_rgb", "left_wrist": "left_wrist_rgb", "right_wrist": "right_wrist_rgb"}
VIDEO_KEYS = {
    "head": "observation.images.head_rgb",
    "left_wrist": "observation.images.left_wrist_rgb",
    "right_wrist": "observation.images.right_wrist_rgb",
}


def _intrinsics(stream: dict[str, Any]) -> dict[str, Any]:
    value = stream["intrinsics"]
    return {key: value[key] for key in ("width", "height", "fx", "fy", "cx", "cy")}


def _calibration_camera_config(calibration: dict[str, Any], camera: str) -> dict[str, Any]:
    value = calibration["cameras"][CAMERA_KEYS[camera]]
    extrinsics = value["extrinsics"]["infrared1_to_color"]
    return {
        "color_intrinsics": _intrinsics(value["streams"]["color"]),
        "depth_intrinsics": _intrinsics(value["streams"]["infrared1"]),
        "depth_to_color_extrinsics": {
            "rotation_matrix_row_major": extrinsics["rotation_matrix_row_major"],
            "translation_m": extrinsics["translation_m"],
        },
        "baseline_m": float(value["baseline"]["recommended_baseline_m"]),
    }


def _resolve_config(
    config_path: Path,
    dataset_root: Path,
    camera: str,
    backend: str,
    artifact_id: str | None = None,
    precision: str | None = None,
    builder_optimization_level: int | None = None,
    workspace_gib: float | None = None,
) -> Any:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Builder config must be a mapping: {config_path}")
    calibration_path = dataset_root / "meta/realsense_calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration_camera = _calibration_camera_config(calibration, camera)
    raw.setdefault("camera", {})
    raw["camera"].update(
        {
            "aligned_depth_to_color": False,
            "color_intrinsics": calibration_camera["color_intrinsics"],
            "depth_intrinsics": calibration_camera["depth_intrinsics"],
            "depth_to_color_extrinsics": calibration_camera["depth_to_color_extrinsics"],
        }
    )
    source = raw.setdefault("depth_source", {})
    if source.get("mode") != "ffs_stereo" or not isinstance(source.get("ffs"), dict):
        raise ValueError("--builder-config must define depth_source.mode=ffs_stereo and depth_source.ffs")
    ffs = source["ffs"]
    ffs.update(
        {
            "backend": backend,
            "calibration_path": str(calibration_path),
            "calibration_camera": camera,
            "baseline_m": calibration_camera["baseline_m"],
            "width": 640,
            "height": 480,
        }
    )
    if precision is not None:
        ffs["precision"] = precision
    if builder_optimization_level is not None:
        ffs["builder_optimization_level"] = builder_optimization_level
    if workspace_gib is not None:
        ffs["workspace_gib"] = workspace_gib
    artifact_dir = (REPO_ROOT / "ffs_reproduction/artifacts").resolve()
    artifact_id = str(artifact_id or ffs.get("artifact_id", "fp16_o3"))
    # The CLI-selected variant is authoritative for every backend, including
    # the PyTorch reference metadata.  Keeping the YAML default here would
    # label an fp32 diagnostic run as the fp16 artifact.
    ffs["artifact_id"] = artifact_id
    defaults = {
        "pytorch": {"checkpoint_path": artifact_dir / "model_best_bp2_serialize.pth"},
        "tensorrt_single": {
            "engine_path": artifact_dir / f"tensorrt_single_{artifact_id}.engine",
            "manifest_path": artifact_dir / f"tensorrt_single_{artifact_id}.manifest.json",
            "config_path": artifact_dir / f"tensorrt_single_{artifact_id}.yaml",
        },
        "tensorrt_two_stage": {
            "feature_engine_path": artifact_dir / f"tensorrt_two_stage_feature_{artifact_id}.engine",
            "post_engine_path": artifact_dir / f"tensorrt_two_stage_post_{artifact_id}.engine",
            "manifest_path": artifact_dir / f"tensorrt_two_stage_{artifact_id}.manifest.json",
            "config_path": artifact_dir / f"tensorrt_two_stage_{artifact_id}.yaml",
        },
        "tensorrt_plugin": {
            "engine_path": artifact_dir / f"tensorrt_plugin_{artifact_id}.engine",
            "plugin_library_path": REPO_ROOT / "ffs_reproduction/build/libffs_gwc_plugin.so",
            "manifest_path": artifact_dir / f"tensorrt_plugin_{artifact_id}.manifest.json",
            "config_path": artifact_dir / f"tensorrt_plugin_{artifact_id}.yaml",
        },
    }
    for key, value in defaults[backend].items():
        ffs.setdefault(key, str(value))
    for key in (
        "checkpoint_path",
        "model_config_path",
        "engine_path",
        "feature_engine_path",
        "post_engine_path",
        "plugin_library_path",
        "manifest_path",
        "config_path",
        "calibration_path",
    ):
        value = ffs.get(key)
        if value and not Path(value).expanduser().is_absolute():
            base = REPO_ROOT if str(value).startswith("ffs_reproduction/") else config_path.parent
            ffs[key] = str((base / value).resolve())
    return parse_config(raw)


def _decode_rgb(dataset_root: Path, camera: str, frame_index: int) -> np.ndarray | None:
    """Decode one offline RGB video frame; never used by the live builder."""

    try:
        import av
    except ImportError:
        return None
    videos = sorted((dataset_root / "videos" / VIDEO_KEYS[camera]).glob("**/*.mp4"))
    if not videos:
        return None
    with av.open(str(videos[0])) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index == frame_index:
                return frame.to_ndarray(format="rgb24")
    raise IndexError(f"RGB video has no frame {frame_index}: {videos[0]}")


def _parity(native_depth: np.ndarray, ffs_depth: np.ndarray, scale: float) -> dict[str, Any]:
    native_m = native_depth.astype(np.float32) * float(scale)
    native_valid = np.isfinite(native_m) & (native_m > 0.0)
    ffs_valid = np.isfinite(ffs_depth) & (ffs_depth > 0.0)
    overlap = native_valid & ffs_valid
    result: dict[str, Any] = {
        "native_depth_valid_ratio": float(native_valid.mean()),
        "ffs_depth_valid_ratio": float(ffs_valid.mean()),
        "valid_overlap_ratio_over_image": float(overlap.mean()),
        "valid_overlap_ratio_over_native": float(overlap.sum() / max(int(native_valid.sum()), 1)),
        "comparison_is_not_absolute_ground_truth": True,
    }
    if overlap.any():
        error = np.abs(ffs_depth[overlap] - native_m[overlap])
        result.update(
            {
                "depth_abs_error_mean_m": float(error.mean()),
                "depth_abs_error_median_m": float(np.median(error)),
                "depth_abs_error_p95_m": float(np.percentile(error, 95)),
                "depth_relative_error_mean": float((error / np.maximum(native_m[overlap], 1e-6)).mean()),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--camera", choices=("head", "left_wrist", "right_wrist"), required=True)
    parser.add_argument("--global-frame-index", type=int, required=True)
    parser.add_argument("--backend", choices=("pytorch", "tensorrt_single", "tensorrt_two_stage", "tensorrt_plugin"), required=True)
    parser.add_argument("--builder-config", type=Path, required=True)
    parser.add_argument("--artifact-id", default=None, help="Explicit engine/config variant, e.g. fp32_o0")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default=None)
    parser.add_argument("--builder-optimization-level", type=int, choices=range(6), default=None)
    parser.add_argument("--workspace-gib", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    data_paths = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    row_count = sum(pq.ParquetFile(path).metadata.num_rows for path in data_paths)
    source = lerobot_rgbd_source.open_rgbd_sidecar_source(root, source="auto", info=info, parquet_row_count=row_count)
    frame = source.read_frame_at(
        data_paths,
        camera=args.camera,
        row_index=args.global_frame_index,
        columns=["global_frame_index"],
        include_ir=True,
    )
    observed_index = int(frame.row["global_frame_index"])
    if observed_index != args.global_frame_index:
        raise ValueError(f"Requested global frame {args.global_frame_index}, reader returned {observed_index}")
    calibration_path = root / "meta/realsense_calibration.json"
    calibration = load_realsense_calibration(calibration_path, args.camera, rectification_mode="auto")
    config = _resolve_config(
        args.builder_config.expanduser().resolve(), root, args.camera, args.backend, args.artifact_id,
        args.precision, args.builder_optimization_level, args.workspace_gib,
    )
    rgb = _decode_rgb(root, args.camera, args.global_frame_index) if config.pointcloud.use_rgb else None
    input_frame = {
        "depth": torch.as_tensor(frame.depth),
        "left_ir": torch.as_tensor(frame.ir_pair.left_ir),
        "right_ir": torch.as_tensor(frame.ir_pair.right_ir),
        "rgb": rgb,
        "timestamp": frame.ir_pair.timestamp,
        "global_frame_index": observed_index,
    }
    if rgb is None:
        input_frame.pop("rgb")
    builder = PointCloudBuilder(config)
    perception, meta = builder.build_perception_stages(input_frame)
    output_dir = args.output_dir.expanduser().resolve() / args.backend
    summary = save_pipeline_artifacts(perception, meta, output_dir, no_show=args.no_show)
    parity = _parity(frame.depth, perception["depth"].detach().cpu().numpy(), source.depth_scale_m_per_unit(args.camera))
    parity.update(
        {
            "camera": args.camera,
            "global_frame_index": observed_index,
            "calibration_sha256": frame.ir_pair.calibration_sha256,
            "rectification": calibration.metadata,
            "raw_points": int(perception["raw"].shape[0]),
            "cropped_points": int(perception["cropped"].shape[0]),
            "sampled_points": int(perception["sampled"].shape[0]),
            "crop_removed_points": int(perception["raw"].shape[0] - perception["cropped"].shape[0]),
            "sampling_removed_or_padded_points": int(perception["cropped"].shape[0] - perception["sampled"].shape[0]),
            "native_depth_used_for_builder": False,
        }
    )
    (output_dir / "parity.json").write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "parity": parity}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
