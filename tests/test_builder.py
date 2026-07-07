from __future__ import annotations

import torch

from pointcloud_builder import PointCloudBuilder


def test_builder_from_yaml_instantiates() -> None:
    builder = PointCloudBuilder.from_yaml("configs/example_head_depth_raw.yaml")
    assert builder.camera.width == 640
    assert builder.config.sampling.mode == "voxel_fps"


def test_builder_returns_fixed_size_tensor() -> None:
    builder = PointCloudBuilder.from_yaml("configs/example_head_depth_raw.yaml")
    frame = {
        "depth": torch.ones((builder.camera.height, builder.camera.width), dtype=torch.float32),
    }
    pc, meta = builder.from_recorded_frame(frame)
    assert pc.shape == (builder.config.sampling.num_points, 3)
    assert meta["sampled_count"] == builder.config.sampling.num_points
