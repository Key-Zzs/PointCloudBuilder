from __future__ import annotations

import torch

from pointcloud_builder.config import CropConfig, SamplingConfig
from pointcloud_builder.crop import crop_points
from pointcloud_builder.sampling import sample_points


def test_empty_crop_does_not_crash_sampling() -> None:
    points = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32)
    crop = CropConfig(enabled=True, x=(-1.0, 1.0), y=(-1.0, 1.0), z=(0.1, 1.0))
    sampling = SamplingConfig(mode="voxel_random", num_points=8)
    cropped, colors, _ = crop_points(points, crop)
    sampled, sampled_colors, meta = sample_points(cropped, sampling, colors)
    assert sampled.shape == (8, 3)
    assert sampled_colors is None
    assert meta["empty_input"] is True
