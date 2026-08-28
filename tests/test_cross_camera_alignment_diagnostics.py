from __future__ import annotations

import numpy as np

from pointcloud_builder.diagnostics.cross_camera_alignment import (
    apply_transform,
    binned_residuals,
    correlation_summary,
    frozen_overlap_mask,
    normalized_image_coordinates,
    point_to_plane_observability,
    robust_point_to_point_icp,
)


def _scene(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    plane = np.column_stack(
        (rng.uniform(-0.25, 0.25, 2500), rng.uniform(-0.20, 0.20, 2500), np.zeros(2500))
    )
    box = rng.uniform((-0.05, -0.04, 0.02), (0.08, 0.07, 0.15), (1800, 3))
    background = rng.uniform((-0.3, 0.16, 0.05), (0.3, 0.22, 0.35), (800, 3))
    return np.concatenate((plane, box, background))


def _transform() -> np.ndarray:
    angles = np.deg2rad((0.35, -0.25, 0.5))
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    result = np.eye(4)
    result[:3, :3] = (
        (cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy),
        (cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx),
        (-sy, cy * sx, cx * cy),
    )
    result[:3, 3] = (0.005, -0.002, 0.001)
    return result


def test_pure_extrinsic_error_is_recovered() -> None:
    anchor = _scene()
    injected = _transform()
    moving = apply_transform(anchor, np.linalg.inv(injected))
    fit = robust_point_to_point_icp(
        anchor,
        moving,
        maximum_correspondence_distance_m=0.025,
        trim_fraction=0.9,
    )["T_anchor_from_moving_residual"]
    assert np.linalg.norm(fit[:3, 3] - injected[:3, 3]) < 3e-4
    delta = fit[:3, :3] @ injected[:3, :3].T
    angle_error = np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))
    assert np.rad2deg(angle_error) < 0.06


def test_point_to_plane_observability_rejects_single_plane() -> None:
    rng = np.random.default_rng(10)
    plane = np.column_stack(
        (rng.uniform(-1, 1, 1000), rng.uniform(-1, 1, 1000), np.zeros(1000))
    )
    plane_normals = np.tile((0.0, 0.0, 1.0), (plane.shape[0], 1))
    assert not point_to_plane_observability(plane, plane_normals)["passed"]
    scene = _scene()
    normals = rng.normal(size=scene.shape)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    assert point_to_plane_observability(scene, normals)["passed"]


def test_constant_residual_correlation_is_explicitly_undefined() -> None:
    result = correlation_summary(np.arange(20.0), np.ones(20))
    assert result["defined"] is False
    assert result["reason"] == "constant_y"


def test_overlap_cohort_is_frozen_from_before_correction() -> None:
    before = np.asarray((0.002, 0.029, 0.031, np.inf))
    after = np.asarray((0.001, 0.040, 0.0005, 0.001))
    cohort = frozen_overlap_mask(before)
    assert cohort.tolist() == [True, True, False, False]
    assert after[cohort].tolist() == [0.001, 0.04]


def test_intrinsic_like_radial_error_detector_finds_edge_trend() -> None:
    rng = np.random.default_rng(4)
    uv = rng.uniform((0, 0), (640, 480), (5000, 2))
    coords = normalized_image_coordinates(
        uv, width=640, height=480, fx=600, fy=600, cx=320, cy=240
    )
    residual = 0.0004 + 0.008 * coords["radius"] ** 2
    bins = binned_residuals(coords["radius"], residual, bins=5)
    trend = correlation_summary(coords["radius"], residual * 1000)
    assert bins[-1]["median_mm"] > 3 * bins[0]["median_mm"]
    assert trend["spearman_rho"] > 0.95


def test_depth_dependent_error_detector_recovers_slope() -> None:
    depth = np.linspace(0.4, 1.8, 500)
    residual_mm = 0.8 + 4.5 * depth
    trend = correlation_summary(depth, residual_mm)
    assert abs(trend["theil_sen_slope"] - 4.5) < 1e-6
    assert trend["pearson_r"] > 0.999


def test_timestamp_static_and_dynamic_cases() -> None:
    rng = np.random.default_rng(9)
    skew_ms = np.linspace(0.1, 30.0, 300)
    static = 1.5 + rng.normal(0, 0.08, skew_ms.size)
    dynamic = 1.5 + 0.7 * skew_ms + rng.normal(0, 0.08, skew_ms.size)
    static_result = correlation_summary(skew_ms, static)
    dynamic_result = correlation_summary(skew_ms, dynamic)
    assert abs(static_result["pearson_r"]) < 0.15
    assert dynamic_result["pearson_r"] > 0.99
    assert abs(dynamic_result["theil_sen_slope"] - 0.7) < 0.02
