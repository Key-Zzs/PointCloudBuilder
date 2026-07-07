from __future__ import annotations

import torch

from pointcloud_builder.config import CropConfig
from pointcloud_builder.crop import crop_points


def test_crop_filters_points() -> None:
    points = torch.tensor([[0.0, 0.0, 0.5], [10.0, 0.0, 0.5]], dtype=torch.float32)
    config = CropConfig(enabled=True, x=(-1.0, 1.0), y=(-1.0, 1.0), z=(0.1, 1.0))
    cropped, _, mask = crop_points(points, config)
    assert cropped.shape == (1, 3)
    assert mask.tolist() == [True, False]
