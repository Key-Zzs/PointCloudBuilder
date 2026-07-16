from __future__ import annotations

import torch

from pointcloud_builder.ffs.geometry import disparity_to_depth


def test_disparity_to_depth_formula_and_invalid_depth_zero() -> None:
    disparity = torch.tensor([[2.0, 4.0], [float("nan"), 0.0]], dtype=torch.float32)
    depth, valid, counts = disparity_to_depth(
        disparity,
        fx_px=100.0,
        baseline_m=0.05,
        remove_invisible=False,
    )
    assert torch.allclose(depth[0, 0], torch.tensor(2.5))
    assert torch.allclose(depth[0, 1], torch.tensor(1.25))
    assert bool(valid[0, 0]) and bool(valid[0, 1])
    assert not bool(valid[1, 0]) and not bool(valid[1, 1])
    assert depth[1, 0] == 0 and depth[1, 1] == 0
    assert counts["non_finite"] == 1
    assert counts["non_positive_or_epsilon"] == 1


def test_invisible_right_coordinate_is_removed() -> None:
    disparity = torch.full((2, 5), 2.0)
    depth, valid, counts = disparity_to_depth(
        disparity,
        fx_px=100.0,
        baseline_m=0.05,
        remove_invisible=True,
    )
    assert counts["invisible_right_coordinate"] == 4
    assert int(valid.sum()) == 6
    assert torch.all(depth[0, :2] == 0)
    assert torch.all(depth[1, :2] == 0)
    assert torch.all(depth[:, 2:] > 0)
