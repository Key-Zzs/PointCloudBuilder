#!/usr/bin/env python3
"""Evaluate the frozen M7 warmup/steady-state memory plateau gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any


MIB = 1024 * 1024
FIELDS = ("rss_bytes", "cuda_allocated_bytes", "cuda_reserved_bytes")
DELTA_LIMITS = {
    "rss_bytes": 256 * MIB,
    "cuda_allocated_bytes": 64 * MIB,
    "cuda_reserved_bytes": 256 * MIB,
}
SLOPE_LIMITS_PER_100_FRAMES = {
    "rss_bytes": 5 * MIB,
    "cuda_allocated_bytes": 2 * MIB,
    "cuda_reserved_bytes": 8 * MIB,
}


def analyze_memory_plateau(
    samples: list[dict[str, Any]], *, warmup_frames: int = 300
) -> dict[str, Any]:
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be non-negative")
    ordered = sorted(samples, key=lambda item: int(item["frame_index"]))
    indices = [int(item["frame_index"]) for item in ordered]
    if len(indices) != len(set(indices)):
        raise ValueError("memory samples contain duplicate frame indices")
    if any(not math.isfinite(float(item[field])) or float(item[field]) < 0 for item in ordered for field in FIELDS):
        raise ValueError("memory sample values must be finite and non-negative")
    steady = [item for item in ordered if int(item["frame_index"]) > warmup_frames]
    if len(steady) < 8:
        raise ValueError("at least eight steady-state memory samples are required")
    quartile_size = max(1, len(steady) // 4)
    metrics: dict[str, dict[str, Any]] = {}
    for field in FIELDS:
        first = [float(item[field]) for item in steady[:quartile_size]]
        last = [float(item[field]) for item in steady[-quartile_size:]]
        delta = statistics.median(last) - statistics.median(first)
        slope = _least_squares_slope(
            [float(item["frame_index"]) / 100.0 for item in steady],
            [float(item[field]) for item in steady],
        )
        delta_limit = DELTA_LIMITS[field]
        slope_limit = SLOPE_LIMITS_PER_100_FRAMES[field]
        metrics[field] = {
            "first_quartile_median_bytes": statistics.median(first),
            "last_quartile_median_bytes": statistics.median(last),
            "median_delta_bytes": delta,
            "median_delta_limit_bytes": delta_limit,
            "slope_bytes_per_100_frames": slope,
            "slope_limit_bytes_per_100_frames": slope_limit,
            "delta_passed": delta <= delta_limit,
            "slope_passed": slope <= slope_limit,
            "passed": delta <= delta_limit and slope <= slope_limit,
        }
    passed = all(item["passed"] for item in metrics.values())
    return {
        "schema_version": "pointcloud-builder.memory-plateau.v1",
        "sample_count": len(ordered),
        "steady_sample_count": len(steady),
        "first_frame": indices[0] if indices else None,
        "last_frame": indices[-1] if indices else None,
        "warmup_frames": warmup_frames,
        "metrics": metrics,
        "plateau": passed,
        "status": "PASS" if passed else "FAIL",
    }


def _least_squares_slope(xs: list[float], ys: list[float]) -> float:
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator == 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=False)) / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--warmup-frames", type=int, default=300)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    samples = raw["samples"] if isinstance(raw, dict) else raw
    if not isinstance(samples, list):
        raise ValueError("input must be a sample list or an object with samples")
    report = analyze_memory_plateau(samples, warmup_frames=args.warmup_frames)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["plateau"]:
        raise SystemExit("memory plateau gate failed")


if __name__ == "__main__":
    main()
