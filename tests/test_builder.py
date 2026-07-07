from __future__ import annotations

import torch

from pointcloud_builder import PointCloudBuilder


def test_builder_from_yaml_instantiates() -> None:
    builder = PointCloudBuilder.from_yaml("configs/example_head_depth_raw.yaml")
    assert builder.camera.width == 640
    assert builder.camera.active_intrinsics.fx == 600.0
    assert builder.config.pointcloud.use_rgb is False


def test_builder_returns_raw_point_cloud() -> None:
    builder = PointCloudBuilder.from_yaml("configs/example_head_depth_raw.yaml")
    frame = {
        "depth": torch.ones((builder.camera.height, builder.camera.width), dtype=torch.float32),
    }
    pc, meta = builder.from_recorded_frame(frame)
    assert pc.shape == (builder.camera.height * builder.camera.width, 3)
    assert meta["stage"] == "raw"
    assert meta["num_raw_points"] == builder.camera.height * builder.camera.width
