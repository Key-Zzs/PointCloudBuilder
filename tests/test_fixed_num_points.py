from __future__ import annotations

import torch

from pointcloud_builder.config import SamplingConfig
from pointcloud_builder.sampling import sample_point_cloud


def test_fixed_num_points_when_input_is_smaller() -> None:
    points = torch.rand((3, 3), dtype=torch.float32)
    config = SamplingConfig(mode="stride", num_points=16, stride=2)
    sampled, _ = sample_point_cloud(points, config)
    assert sampled.shape == (16, 3)


def test_fixed_num_points_when_input_is_larger() -> None:
    points = torch.rand((64, 6), dtype=torch.float32)
    config = SamplingConfig(mode="fps", num_points=16)
    sampled, meta = sample_point_cloud(points, config)
    assert sampled.shape == (16, 6)
    assert meta["sampled_count"] == 16
