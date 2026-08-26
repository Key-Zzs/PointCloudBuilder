"""Fail-closed live mapping performance gates."""

from __future__ import annotations

import math
import statistics
from typing import Any


def evaluate_rss_plateau(
    samples_mb: tuple[tuple[int, float], ...] | list[tuple[int, float]],
    *,
    warmup_fraction: float = 0.20,
    median_delta_limit_mb: float = 256.0,
    slope_limit_mb_per_100_frames: float = 5.0,
) -> dict[str, Any]:
    """Apply the established RSS delta/slope policy to mapper child samples."""

    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("RSS warmup_fraction must be in [0, 1)")
    ordered = sorted((int(index), float(value)) for index, value in samples_mb)
    if (
        len(ordered) < 32
        or len({index for index, _ in ordered}) != len(ordered)
        or any(
            index < 0 or not math.isfinite(value) or value < 0
            for index, value in ordered
        )
    ):
        return {
            "evaluated": False,
            "passed": False,
            "reason": "at least 32 unique finite RSS samples are required",
            "sample_count": len(ordered),
        }
    warmup_count = int(len(ordered) * warmup_fraction)
    steady = ordered[warmup_count:]
    quartile = max(1, len(steady) // 4)
    first_median = statistics.median(value for _, value in steady[:quartile])
    last_median = statistics.median(value for _, value in steady[-quartile:])
    delta_mb = last_median - first_median
    xs = [index / 100.0 for index, _ in steady]
    ys = [value for _, value in steady]
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    slope = (
        0.0
        if denominator == 0
        else sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
        / denominator
    )
    gates = {
        "median_delta_le_256mb": delta_mb <= median_delta_limit_mb,
        "slope_le_5mb_per_100_frames": slope <= slope_limit_mb_per_100_frames,
    }
    return {
        "evaluated": True,
        "passed": all(gates.values()),
        "sample_count": len(ordered),
        "steady_sample_count": len(steady),
        "first_quartile_median_mb": first_median,
        "last_quartile_median_mb": last_median,
        "median_delta_mb": delta_mb,
        "median_delta_limit_mb": median_delta_limit_mb,
        "slope_mb_per_100_frames": slope,
        "slope_limit_mb_per_100_frames": slope_limit_mb_per_100_frames,
        "gates": gates,
    }
