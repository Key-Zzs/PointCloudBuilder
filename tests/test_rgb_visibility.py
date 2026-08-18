"""Synthetic contracts for the RGB-view visibility gate."""

from __future__ import annotations

import numpy as np
import torch

from pointcloud_builder import PointCloudBuilder
from pointcloud_builder.camera_model import CameraIntrinsics
from pointcloud_builder.config import parse_config
from pointcloud_builder.projection import ColorViewVisibilityFilter, lift_binary_mask, project_points_to_color_image


INTRINSICS = CameraIntrinsics(width=4, height=4, fx=1.0, fy=1.0, cx=0.0, cy=0.0)


def _mask() -> torch.Tensor:
    mask = torch.zeros((4, 4), dtype=torch.bool)
    mask[1, 1] = True
    mask[1, 2] = True
    return mask


def _project(points: torch.Tensor):
    return project_points_to_color_image(points, extrinsics=None, intrinsics=INTRINSICS)


def test_visibility_a_front_point_wins_same_rgb_pixel() -> None:
    points = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    lifted, selected = lift_binary_mask(
        points,
        _project(points),
        _mask(),
        visibility_filter=ColorViewVisibilityFilter(epsilon_z=0.005),
    )
    assert selected.tolist() == [True, False]
    assert torch.equal(lifted, points[:1])


def test_visibility_b_keeps_within_epsilon_and_rejects_far_back_point() -> None:
    points = torch.tensor([[1.0, 1.0, 1.0], [1.003, 1.003, 1.003], [1.2, 1.2, 1.2]])
    projection = _project(points)
    visible = ColorViewVisibilityFilter(epsilon_z=0.005).visible_mask(projection)
    assert visible.tolist() == [True, True, False]


def test_visibility_c_rejects_point_behind_color_camera() -> None:
    points = torch.tensor([[1.0, 1.0, -1.0], [1.0, 1.0, 1.0]])
    visible = ColorViewVisibilityFilter().visible_mask(_project(points))
    assert visible.tolist() == [False, True]


def test_visibility_d_rejects_out_of_bounds_pixel() -> None:
    points = torch.tensor([[4.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    projection = _project(points)
    assert projection.valid.tolist() == [False, True]
    assert ColorViewVisibilityFilter().visible_mask(projection).tolist() == [False, True]


def test_visibility_e_tests_different_rgb_pixels_independently() -> None:
    points = torch.tensor([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [2.0, 1.0, 1.0],
        [4.0, 2.0, 2.0],
    ])
    visible = ColorViewVisibilityFilter().visible_mask(_project(points))
    assert visible.tolist() == [True, False, True, False]


def test_visibility_f_disabled_reproduces_legacy_lifting() -> None:
    points = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    projection = _project(points)
    legacy_points, legacy_selected = lift_binary_mask(points, projection, _mask())
    disabled_points, disabled_selected = lift_binary_mask(
        points,
        projection,
        _mask(),
        visibility_filter=ColorViewVisibilityFilter(enabled=False),
    )
    assert torch.equal(disabled_selected, legacy_selected)
    assert torch.equal(disabled_points, legacy_points)


def test_visibility_g_xyz_only_builder_can_still_use_rgb_projection() -> None:
    raw = {
        "device": "cpu",
        "camera": {
            "name": "synthetic",
            "depth_scale": 1.0,
            "aligned_depth_to_color": True,
            "color_intrinsics": {"width": 4, "height": 4, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
            "depth_intrinsics": {"width": 4, "height": 4, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        },
        "pointcloud": {"use_rgb": False, "output_format": "xyz"},
        "sampling": {"enabled": False, "mode": "fps", "num_points": 2},
    }
    builder = PointCloudBuilder(parse_config(raw))
    points = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    projection = builder.project_points_to_color_image(points, source_frame="raw_dense")
    lifted, selected = lift_binary_mask(
        points,
        projection,
        _mask(),
        visibility_filter=builder.config.visibility_filter,
    )
    assert builder.config.pointcloud.use_rgb is False
    assert selected.tolist() == [True, False]
    assert torch.equal(lifted, points[:1])


def test_visibility_contract_is_shared_by_dense_and_sparse_modes() -> None:
    from pointcloud_builder.instance import build_instance_dense, build_instance_sparse
    from pointcloud_builder.sampling import SamplingConfig
    from pointcloud_builder.segmentation.types import InstanceMask

    points = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    item = InstanceMask(0, 0, "track", "object", mask, (1, 1, 1, 1), 1.0, "text", "object", True)
    visibility_filter = ColorViewVisibilityFilter(epsilon_z=0.005)
    dense = build_instance_dense(
        raw_dense_points=points,
        projection=_project(points),
        masks=[item],
        sampling_config=SamplingConfig(mode="fps", num_points=1),
        support_plane=None,
        visibility_filter=visibility_filter,
    )
    sparse = build_instance_sparse(
        workspace_sampled_points=points,
        projection=_project(points),
        masks=[item],
        sampling_config=SamplingConfig(mode="fps", num_points=1),
        visibility_filter=visibility_filter,
    )
    assert torch.equal(dense.instances[0].visibility_mask, sparse.instances[0].visibility_mask)
    assert dense.instances[0].visibility_rejected_count == 1
    assert sparse.instances[0].visibility_rejected_count == 1
