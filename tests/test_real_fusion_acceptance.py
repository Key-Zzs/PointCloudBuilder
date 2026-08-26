from __future__ import annotations

import math

import pytest

from pointcloud_builder.fusion import evaluate_interference, evaluate_repeatability


def test_interference_advisory_does_not_fail_without_gross_threshold() -> None:
    result = evaluate_interference(
        {
            "camera_a": {"valid_ratio": 0.90, "board_p95_m": 0.010},
            "camera_b": {"valid_ratio": 0.92, "board_p95_m": 0.010},
        },
        {
            "camera_a": {"valid_ratio": 0.78, "board_p95_m": 0.012},
            "camera_b": {"valid_ratio": 0.91, "board_p95_m": 0.022},
        },
    )
    assert result["advisory"]
    assert not result["gross_failure"]
    assert result["status"] == "PASS_WITH_WARNING"
    assert result["passed"]


@pytest.mark.parametrize(
    ("valid", "p95"),
    ((0.49, 0.010), (0.80, 0.041)),
)
def test_interference_gross_threshold_fails(valid: float, p95: float) -> None:
    result = evaluate_interference(
        {"camera_a": {"valid_ratio": 0.90, "board_p95_m": 0.010}},
        {"camera_a": {"valid_ratio": valid, "board_p95_m": p95}},
    )
    assert result["gross_failure"]
    assert not result["passed"]


def test_interference_gross_equality_boundaries_pass() -> None:
    result = evaluate_interference(
        {"camera_a": {"valid_ratio": 0.90, "board_p95_m": 0.020}},
        {"camera_a": {"valid_ratio": 0.50, "board_p95_m": 0.040}},
    )
    assert result["passed"]


def test_repeatability_enforces_five_mm_and_fatal_error_gates() -> None:
    first = {
        "cube_dimensions_m": (0.070, 0.069, 0.071),
        "board_median_abs_z_m": 0.002,
        "overlap_symmetric_median_m": 0.008,
        "worker_fatal_error": False,
    }
    second = {
        "cube_dimensions_m": (0.073, 0.071, 0.068),
        "board_median_abs_z_m": 0.004,
        "overlap_symmetric_median_m": 0.010,
        "worker_fatal_error": False,
    }
    assert evaluate_repeatability(first, second)["passed"]
    second["cube_dimensions_m"] = (0.076, 0.071, 0.068)
    assert not evaluate_repeatability(first, second)["passed"]


def test_repeatability_keeps_height_separate_from_xy_dimensions() -> None:
    first = {
        "cube_dimensions_m": (0.080, 0.060, 0.055),
        "board_median_abs_z_m": 0.002,
        "overlap_symmetric_median_m": 0.008,
        "worker_fatal_error": False,
    }
    second = {
        "cube_dimensions_m": (0.080, 0.055, 0.061),
        "board_median_abs_z_m": 0.002,
        "overlap_symmetric_median_m": 0.008,
        "worker_fatal_error": False,
    }
    assert not evaluate_repeatability(first, second)["passed"]


@pytest.mark.parametrize("invalid", (math.nan, math.inf, -math.inf))
def test_interference_rejects_non_finite_metrics(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        evaluate_interference(
            {"camera_a": {"valid_ratio": 0.90, "board_p95_m": 0.010}},
            {"camera_a": {"valid_ratio": invalid, "board_p95_m": 0.010}},
        )
    with pytest.raises(ValueError, match="finite"):
        evaluate_interference(
            {"camera_a": {"valid_ratio": 0.90, "board_p95_m": 0.010}},
            {"camera_a": {"valid_ratio": 0.80, "board_p95_m": invalid}},
        )


@pytest.mark.parametrize("invalid", (-0.01, 1.01))
def test_interference_rejects_out_of_range_ratios(invalid: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluate_interference(
            {"camera_a": {"valid_ratio": 0.90, "board_p95_m": 0.010}},
            {"camera_a": {"valid_ratio": invalid, "board_p95_m": 0.010}},
        )
