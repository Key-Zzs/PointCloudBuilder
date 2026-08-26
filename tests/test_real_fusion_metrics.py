from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from pointcloud_builder.fusion import (
    VoxelFusionConfig,
    board_surface_metrics,
    contribution_metrics,
    cube_box_voxel_count,
    detect_cube,
    fusion_geometry_metrics,
    symmetric_overlap_metrics,
    voxel_fuse_workspace_clouds,
)
from pointcloud_builder.rig.types import WorkspaceCloud
from pointcloud_builder.workspace import ExpectedPlaneRegion, WorkspacePointCloud


def _plane_region() -> ExpectedPlaneRegion:
    return ExpectedPlaneRegion(
        frame="workspace",
        x=(0.015, 0.195),
        y=(0.015, 0.135),
        expected_z_m=0.0,
        z_search_range_m=(-0.08, 0.08),
    )


def _plane(offset: float = 0.0) -> torch.Tensor:
    x, y = torch.meshgrid(
        torch.linspace(0.015, 0.195, 50),
        torch.linspace(0.015, 0.135, 35),
        indexing="ij",
    )
    z = torch.full_like(x, offset)
    return torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)


def _cube_surface(
    *,
    yaw_deg: float = 27.0,
    center_xy: tuple[float, float] = (0.105, 0.075),
    base_z_m: float = 0.0,
) -> torch.Tensor:
    values = torch.linspace(-0.035, 0.035, 29)
    u, v = torch.meshgrid(values, values, indexing="ij")
    faces = []
    for fixed_axis in range(3):
        for sign in (-1.0, 1.0):
            points = torch.zeros((u.numel(), 3), dtype=torch.float64)
            points[:, fixed_axis] = sign * 0.035
            free = [axis for axis in range(3) if axis != fixed_axis]
            points[:, free[0]] = u.flatten()
            points[:, free[1]] = v.flatten()
            faces.append(points)
    cube = torch.cat(faces)
    angle = math.radians(yaw_deg)
    rotation = torch.tensor(
        ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))),
        dtype=torch.float64,
    )
    cube[:, :2] = cube[:, :2] @ rotation.T
    cube += torch.tensor(
        (center_xy[0], center_xy[1], base_z_m + 0.035), dtype=torch.float64
    )
    return cube.to(torch.float32)


def test_board_surface_metrics_reports_thickness_and_outliers() -> None:
    cloud = WorkspacePointCloud(points=_plane(0.002), frame="workspace")
    metrics = board_surface_metrics(cloud, _plane_region())
    assert metrics["median_abs_z_m"] == pytest.approx(0.002, abs=1e-6)
    assert metrics["surface_thickness_m"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["outlier_ratio"] == 0.0


def test_board_outlier_ratio_does_not_pre_filter_gross_z_outliers() -> None:
    points = torch.cat((_plane(), torch.tensor(((0.050, 0.050, 0.100),))), dim=0)
    metrics = board_surface_metrics(
        WorkspacePointCloud(points=points, frame="workspace"), _plane_region()
    )
    assert metrics["outlier_ratio"] > 0.0


def test_symmetric_overlap_accepts_five_mm_offset() -> None:
    a = _plane()
    b = a + torch.tensor((0.004, 0.0, 0.0))
    metrics = symmetric_overlap_metrics(
        a,
        b,
        roi=((0.0, 0.22), (0.0, 0.16), (-0.01, 0.01)),
    )
    assert metrics["target_passed"]
    assert metrics["symmetric"]["p95_m"] < 0.01


def test_detect_cube_recovers_yawed_seventy_mm_fixture() -> None:
    points = torch.cat((_plane(), _cube_surface()), dim=0)
    metrics = detect_cube(points, board_p95_abs_z_m=0.002)
    assert metrics.passed
    assert not metrics.ambiguous
    assert metrics.dimensions_m[:2] == pytest.approx((0.070, 0.070), abs=0.006)
    # Candidate extraction removes the plane and its near-plane samples.  The
    # frozen height is nevertheless exactly p99(z)-p01(z) of the retained cluster.
    assert metrics.dimensions_m[2] == pytest.approx(0.0575, abs=1e-5)
    assert metrics.support_plane_gap_m == pytest.approx(0.0125, abs=1e-5)


def test_detect_cube_fails_closed_for_floating_cube() -> None:
    points = torch.cat((_plane(), _cube_surface(base_z_m=0.020)), dim=0)
    metrics = detect_cube(points, board_p95_abs_z_m=0.002)
    assert metrics.support_plane_gap_m == pytest.approx(0.020, abs=1e-5)
    assert metrics.dimensions_m[2] == pytest.approx(0.070, abs=1e-5)
    assert not metrics.passed


def test_detect_cube_requires_review_for_close_score_candidates() -> None:
    points = torch.cat(
        (
            _plane(),
            _cube_surface(center_xy=(0.105, 0.075)),
            _cube_surface(center_xy=(0.255, 0.075)),
        ),
        dim=0,
    )
    metrics = detect_cube(points, board_p95_abs_z_m=0.002)
    assert metrics.candidate_count == 2
    assert metrics.ambiguous
    assert not metrics.passed


def test_cube_joint_observation_excludes_board_points() -> None:
    cube = detect_cube(
        torch.cat((_plane(), _cube_surface()), dim=0), board_p95_abs_z_m=0.002
    ).to_dict()
    assert cube_box_voxel_count(_plane(), cube, minimum_z_m=0.010) == 0
    assert cube_box_voxel_count(
        _cube_surface(), cube, minimum_z_m=0.010
    ) >= 20


def test_fusion_geometry_and_contribution_are_snapshot_scoped() -> None:
    a = WorkspacePointCloud(points=_plane(), frame="workspace")
    b = WorkspacePointCloud(points=_plane(0.001), frame="workspace")
    result = voxel_fuse_workspace_clouds(
        (WorkspaceCloud("camera_a", a), WorkspaceCloud("camera_b", b)),
        VoxelFusionConfig(enabled=True, voxel_size_m=0.005, deterministic=True),
    )
    concatenated = WorkspacePointCloud(
        points=torch.cat((a.points, b.points)), frame="workspace"
    )
    geometry = fusion_geometry_metrics(
        concatenated,
        result.cloud,
        board_region=_plane_region(),
        voxel_size_m=0.005,
    )
    contributions = contribution_metrics(result.provenance)
    assert geometry["thickness_gate_passed"]
    assert geometry["board_shift_gate_passed"]
    assert contributions["passed"]
    assert set(contributions["per_camera"]) == {"camera_a", "camera_b"}


def test_contribution_accepts_exact_five_percent_boundary() -> None:
    provenance = SimpleNamespace(
        output_voxel_count=20,
        input_point_count=20,
        per_camera_input_count={"camera_a": 1, "camera_b": 19},
        per_camera_unique_voxel_contribution={"camera_a": 1, "camera_b": 19},
        multi_camera_voxel_count=0,
    )
    assert contribution_metrics(provenance)["passed"]
