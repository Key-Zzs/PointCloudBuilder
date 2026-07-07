from __future__ import annotations

import torch

from pointcloud_builder.config import SamplingConfig
from pointcloud_builder.sampling import sample_points


def test_sampling_returns_fixed_size() -> None:
    points = torch.rand((10, 3), dtype=torch.float32)
    config = SamplingConfig(mode="random", num_points=32)
    sampled, colors, meta = sample_points(points, config)
    assert sampled.shape == (32, 3)
    assert colors is None
    assert meta["sampled_count"] == 32
