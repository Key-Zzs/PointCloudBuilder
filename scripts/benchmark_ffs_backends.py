#!/usr/bin/env python3
"""Benchmark all configured FFS routes on one identical stereo frame."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from pointcloud_builder import PointCloudBuilder
from pointcloud_builder.config import parse_config
from pointcloud_builder.frame_io import load_frame


BACKENDS = ("pytorch", "tensorrt_single", "tensorrt_two_stage", "tensorrt_plugin")


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def _config_for(
    path: Path,
    backend: str,
    artifact_id: str | None = None,
    precision: str | None = None,
    builder_optimization_level: int | None = None,
    workspace_gib: float | None = None,
):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("depth_source"), dict) or not isinstance(raw["depth_source"].get("ffs"), dict):
        raise ValueError("Benchmark config must contain depth_source.ffs")
    raw["depth_source"]["ffs"]["backend"] = backend
    artifact_dir = (Path(__file__).resolve().parents[1] / "ffs_reproduction/artifacts").resolve()
    artifact_id = str(artifact_id or raw["depth_source"]["ffs"].get("artifact_id", "fp16_o3"))
    raw["depth_source"]["ffs"]["artifact_id"] = artifact_id
    if precision is not None:
        raw["depth_source"]["ffs"]["precision"] = precision
    if builder_optimization_level is not None:
        raw["depth_source"]["ffs"]["builder_optimization_level"] = builder_optimization_level
    if workspace_gib is not None:
        raw["depth_source"]["ffs"]["workspace_gib"] = workspace_gib
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
            "plugin_library_path": Path(__file__).resolve().parents[1] / "ffs_reproduction/build/libffs_gwc_plugin.so",
            "manifest_path": artifact_dir / f"tensorrt_plugin_{artifact_id}.manifest.json",
            "config_path": artifact_dir / f"tensorrt_plugin_{artifact_id}.yaml",
        },
    }
    for key, value in defaults[backend].items():
        raw["depth_source"]["ffs"].setdefault(key, str(value))
    return parse_config(raw)


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-id", default=None, help="Explicit engine/config variant, e.g. fp32_o0")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default=None)
    parser.add_argument("--builder-optimization-level", type=int, choices=range(6), default=None)
    parser.add_argument("--workspace-gib", type=float, default=None)
    parser.add_argument("--backend", choices=BACKENDS, action="append", dest="backends")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--no-enforce-minimum", action="store_true", help="For development-only smoke tests")
    args = parser.parse_args()
    if not args.no_enforce_minimum and (args.warmup < 20 or args.runs < 100):
        raise ValueError("Final benchmark requires warmup>=20 and runs>=100")
    backends = tuple(args.backends or BACKENDS)
    frame = load_frame(args.input)
    result: dict[str, Any] = {"config": str(args.config.resolve()), "input": str(args.input.resolve()), "warmup": args.warmup, "runs": args.runs, "backends": {}}
    for backend_name in backends:
        config = _config_for(
            args.config.resolve(), backend_name, args.artifact_id, args.precision,
            args.builder_optimization_level, args.workspace_gib,
        )
        builder = PointCloudBuilder(config)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for _ in range(args.warmup):
            builder.build_perception_stages(frame)
        _synchronize()
        host_total: list[float] = []
        gpu_stage_values: dict[str, list[float]] = {}
        last_meta: dict[str, Any] = {}
        last_perception: dict[str, torch.Tensor] = {}
        for _ in range(args.runs):
            _synchronize()
            start = time.perf_counter()
            perception, meta = builder.build_perception_stages(frame)
            _synchronize()
            host_total.append((time.perf_counter() - start) * 1000.0)
            last_meta = meta
            last_perception = perception
            timing = meta.get("ffs", {}).get("timing_ms", {})
            for key, value in timing.items():
                if isinstance(value, (int, float)):
                    gpu_stage_values.setdefault(key, []).append(float(value))
        peak_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        result["backends"][backend_name] = {
            "backend": backend_name,
            "artifact_id": config.depth_source.ffs.artifact_id if config.depth_source.ffs is not None else None,
            "precision": config.depth_source.ffs.precision if config.depth_source.ffs is not None else None,
            "builder_optimization_level": config.depth_source.ffs.builder_optimization_level if config.depth_source.ffs is not None else None,
            "workspace_gib": config.depth_source.ffs.workspace_gib if config.depth_source.ffs is not None else None,
            "host_total": _stats(host_total),
            "gpu_or_event_stages": {key: _stats(values) for key, values in gpu_stage_values.items()},
            "peak_memory_bytes": peak_memory,
            "peak_memory_mib": peak_memory / (1024.0 * 1024.0),
            "output_shape": list(last_perception["sampled"].shape),
            "output_dtype": str(last_perception["sampled"].dtype),
            "output_device": str(last_perception["sampled"].device),
            "metadata": last_meta,
        }
        del builder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
