#!/usr/bin/env python3
"""Run one v05 frame through selected backends and compare to PyTorch."""

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
from run_v05_ffs_frame import _resolve_config  # noqa: E402


BACKENDS = ("pytorch", "tensorrt_single", "tensorrt_two_stage", "tensorrt_plugin")


def _metrics(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> dict[str, Any]:
    ref_disp, cand_disp = reference["disparity"], candidate["disparity"]
    ref_depth, cand_depth = reference["depth"], candidate["depth"]
    ref_valid = np.isfinite(ref_disp) & (ref_disp > 0)
    cand_valid = np.isfinite(cand_disp) & (cand_disp > 0)
    overlap = ref_valid & cand_valid
    result: dict[str, Any] = {
        "valid_overlap_ratio": float(overlap.sum() / max(int(ref_valid.sum()), 1)),
        "finite_positive_ratio": float(cand_valid.mean()),
        "reference_valid_count": int(ref_valid.sum()),
        "candidate_valid_count": int(cand_valid.sum()),
    }
    if overlap.any():
        disp_error = np.abs(cand_disp[overlap] - ref_disp[overlap])
        depth_valid = overlap & (ref_depth > 0) & (cand_depth > 0) & np.isfinite(ref_depth) & np.isfinite(cand_depth)
        result.update(
            {
                "disparity_mae_px": float(disp_error.mean()),
                "disparity_median_px": float(np.median(disp_error)),
                "disparity_p95_px": float(np.percentile(disp_error, 95)),
                "disparity_max_px": float(disp_error.max()),
            }
        )
        if depth_valid.any():
            depth_error = np.abs(cand_depth[depth_valid] - ref_depth[depth_valid])
            result.update(
                {
                    "depth_mae_m": float(depth_error.mean()),
                    "depth_median_m": float(np.median(depth_error)),
                    "depth_p95_m": float(np.percentile(depth_error, 95)),
                    "depth_relative_error_mean": float((depth_error / np.maximum(ref_depth[depth_valid], 1e-6)).mean()),
                }
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--camera", choices=("head", "left_wrist", "right_wrist"), required=True)
    parser.add_argument("--global-frame-index", type=int, required=True)
    parser.add_argument("--builder-config", type=Path, required=True)
    parser.add_argument("--artifact-id", default=None, help="Explicit engine/config variant, e.g. fp32_o0")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default=None)
    parser.add_argument("--builder-optimization-level", type=int, choices=range(6), default=None)
    parser.add_argument("--workspace-gib", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=BACKENDS, action="append", dest="backends")
    parser.add_argument("--thresholds", type=Path, default=REPO_ROOT / "ffs_reproduction/configs/parity_thresholds.yaml")
    args = parser.parse_args()
    backends = tuple(args.backends or BACKENDS)
    root = args.dataset_root.expanduser().resolve()
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    paths = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    row_count = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
    source = lerobot_rgbd_source.open_rgbd_sidecar_source(root, source="auto", info=info, parquet_row_count=row_count)
    row = source.read_frame_at(paths, camera=args.camera, row_index=args.global_frame_index, columns=["global_frame_index"], include_ir=True)
    frame = {"left_ir": row.ir_pair.left_ir, "right_ir": row.ir_pair.right_ir, "timestamp": row.ir_pair.timestamp, "global_frame_index": args.global_frame_index}
    outputs: dict[str, dict[str, np.ndarray]] = {}
    records: dict[str, Any] = {}
    for backend in backends:
        config = _resolve_config(
            args.builder_config.expanduser().resolve(), root, args.camera, backend, args.artifact_id,
            args.precision, args.builder_optimization_level, args.workspace_gib,
        )
        builder = PointCloudBuilder(config)
        perception, meta = builder.build_perception_stages(frame)
        outputs[backend] = {
            "disparity": perception["disparity"].detach().cpu().numpy().astype(np.float32),
            "depth": perception["depth"].detach().cpu().numpy().astype(np.float32),
        }
        records[backend] = {
            "metadata": meta,
            "output_shape": list(perception["sampled"].shape),
            "output_dtype": str(perception["sampled"].dtype),
            "output_device": str(perception["sampled"].device),
        }
        del builder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if "pytorch" not in outputs:
        raise ValueError("PyTorch must be included as the numerical reference")
    reference = outputs["pytorch"]
    thresholds = yaml.safe_load(args.thresholds.read_text(encoding="utf-8"))
    result: dict[str, Any] = {"reference": "pytorch", "thresholds": thresholds, "backends": records, "comparisons": {}}
    for backend, values in outputs.items():
        if backend == "pytorch":
            continue
        metrics = _metrics(reference, values)
        result["comparisons"][backend] = metrics
        checks = {
            "valid_overlap_ratio": metrics.get("valid_overlap_ratio", 0.0) >= float(thresholds["valid_overlap_ratio_min"]),
            "finite_positive_ratio": metrics.get("finite_positive_ratio", 0.0) >= float(thresholds["finite_positive_ratio_min"]),
            "disparity_mae_px": metrics.get("disparity_mae_px", float("inf")) <= float(thresholds["disparity_mae_max_px"]),
            "disparity_p95_px": metrics.get("disparity_p95_px", float("inf")) <= float(thresholds["disparity_p95_max_px"]),
            "depth_mae_m": metrics.get("depth_mae_m", float("inf")) <= float(thresholds["depth_mae_max_m"]),
            "depth_relative_error_mean": metrics.get("depth_relative_error_mean", float("inf")) <= float(thresholds["depth_relative_error_max"]),
        }
        result["comparisons"][backend]["checks"] = checks
        if not all(checks.values()):
            raise RuntimeError(f"Parity thresholds failed for {backend}: {checks}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
