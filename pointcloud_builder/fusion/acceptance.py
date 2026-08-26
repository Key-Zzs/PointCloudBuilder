"""Fail-closed interference and repeatability verdict logic."""

from __future__ import annotations

import math
from typing import Any


def evaluate_interference(
    single: dict[str, dict[str, float | None]],
    concurrent: dict[str, dict[str, float | None]],
) -> dict[str, Any]:
    if set(single) != set(concurrent) or not single:
        raise ValueError("single and concurrent metrics require the same cameras")
    cameras = {}
    for name in sorted(single):
        baseline = single[name]
        simultaneous = concurrent[name]
        valid_single = _ratio(baseline.get("valid_ratio"), f"{name}.single.valid_ratio")
        valid_ab = _number(
            simultaneous.get("valid_ratio"), f"{name}.concurrent.valid_ratio"
        )
        if not 0.0 <= valid_ab <= 1.0:
            raise ValueError(f"{name}.concurrent.valid_ratio must be in [0, 1]")
        p95_single = _number(baseline.get("board_p95_m"), f"{name}.single.board_p95_m")
        p95_ab = _number(
            simultaneous.get("board_p95_m"), f"{name}.concurrent.board_p95_m"
        )
        if p95_single < 0.0 or p95_ab < 0.0:
            raise ValueError(f"{name} board p95 metrics must be non-negative")
        valid_drop = valid_single - valid_ab
        p95_ratio = p95_ab / p95_single if p95_single > 0 else float("inf")
        advisory = bool(valid_drop > 0.10 or p95_ratio > 2.0)
        gross = bool(valid_ab < 0.50 or p95_ab > 0.040)
        cameras[name] = {
            "single": dict(baseline),
            "concurrent": dict(simultaneous),
            "valid_ratio_drop": valid_drop,
            "board_p95_ratio": p95_ratio,
            "advisory": advisory,
            "gross_failure": gross,
        }
    gross = any(item["gross_failure"] for item in cameras.values())
    advisory = any(item["advisory"] for item in cameras.values())
    return {
        "cameras": cameras,
        "advisory": advisory,
        "gross_failure": gross,
        "status": "FAIL_GROSS" if gross else ("PASS_WITH_WARNING" if advisory else "PASS"),
        "passed": not gross,
    }


def evaluate_repeatability(run_1: dict[str, Any], run_2: dict[str, Any]) -> dict[str, Any]:
    dimensions_1 = tuple(
        _number(value, f"run_1.cube_dimensions_m[{index}]")
        for index, value in enumerate(run_1["cube_dimensions_m"])
    )
    dimensions_2 = tuple(
        _number(value, f"run_2.cube_dimensions_m[{index}]")
        for index, value in enumerate(run_2["cube_dimensions_m"])
    )
    if len(dimensions_1) != 3 or len(dimensions_2) != 3:
        raise ValueError("repeatability requires three cube dimensions per run")
    dimension_deltas = tuple(
        abs(left - right) for left, right in zip(dimensions_1, dimensions_2, strict=True)
    )
    board_delta = abs(
        _number(run_1["board_median_abs_z_m"], "run_1.board_median_abs_z_m")
        - _number(run_2["board_median_abs_z_m"], "run_2.board_median_abs_z_m")
    )
    overlap_delta = abs(
        _number(
            run_1["overlap_symmetric_median_m"],
            "run_1.overlap_symmetric_median_m",
        )
        - _number(
            run_2["overlap_symmetric_median_m"],
            "run_2.overlap_symmetric_median_m",
        )
    )
    fatal_errors = bool(run_1.get("worker_fatal_error") or run_2.get("worker_fatal_error"))
    passed = bool(
        max(dimension_deltas) <= 0.005
        and board_delta <= 0.005
        and overlap_delta <= 0.005
        and not fatal_errors
    )
    return {
        "cube_dimension_deltas_m": dimension_deltas,
        "maximum_cube_dimension_delta_m": max(dimension_deltas),
        "board_median_abs_z_delta_m": board_delta,
        "overlap_symmetric_median_delta_m": overlap_delta,
        "worker_fatal_error": fatal_errors,
        "passed": passed,
    }


def _number(value: float | None, label: str) -> float:
    if value is None:
        raise ValueError(f"{label} is required")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _ratio(value: float | None, label: str) -> float:
    result = _number(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result
