from __future__ import annotations

import torch

from pointcloud_builder import PointCloudBuilder


def test_builder_from_yaml_instantiates() -> None:
    builder = PointCloudBuilder.from_yaml("configs/example_head_depth_raw.yaml")
    assert builder.camera.width == 640
    assert builder.camera.active_intrinsics.fx == 600.0
    assert builder.config.pointcloud.use_rgb is False
    assert builder.config.crop.enabled is True
    assert builder.config.sampling.enabled is True


def test_builder_returns_sampled_fixed_size_point_cloud() -> None:
    builder = PointCloudBuilder.from_yaml("configs/example_head_depth_raw.yaml")
    frame = {
        "depth": torch.ones((builder.camera.height, builder.camera.width), dtype=torch.float32),
    }
    pc, meta = builder.from_recorded_frame(frame)
    assert pc.shape == (builder.config.sampling.num_points, 3)
    assert meta["stage"] == "sampled"
    assert meta["num_raw_points"] == builder.camera.height * builder.camera.width
    assert meta["num_cropped_points"] <= meta["num_raw_points"]
    assert meta["num_sampled_points"] == builder.config.sampling.num_points
    assert meta["crop_enabled"] is True
    assert meta["sampling_enabled"] is True
    assert meta["sampling_mode"] == builder.config.sampling.mode
    assert meta["target_num_points"] == builder.config.sampling.num_points
    assert "input_empty" in meta
    assert "padded" in meta
    assert "pad_mode" in meta
    assert "voxel_size" in meta


def test_recorded_and_live_outputs_have_same_shape() -> None:
    builder = PointCloudBuilder.from_yaml("configs/example_head_aligned.yaml")
    frame = {
        "depth": torch.ones((builder.camera.height, builder.camera.width), dtype=torch.float32),
        "rgb": torch.ones((builder.camera.height, builder.camera.width, 3), dtype=torch.uint8),
    }
    recorded, recorded_meta = builder.from_recorded_frame(frame)
    live, live_meta = builder.from_live_frame(frame)
    assert recorded.shape == live.shape
    assert recorded.shape == (builder.config.sampling.num_points, 6)
    assert recorded_meta["num_sampled_points"] == live_meta["num_sampled_points"]


def test_cuda_output_does_not_crash_when_available() -> None:
    if not torch.cuda.is_available():
        return
    builder = PointCloudBuilder.from_yaml("configs/example_head_depth_raw.yaml")
    frame = {
        "depth": torch.ones((builder.camera.height, builder.camera.width), dtype=torch.float32),
    }
    pc, meta = builder.from_live_frame(frame)
    assert pc.is_cuda
    assert pc.shape == (builder.config.sampling.num_points, 3)
    assert meta["device"].startswith("cuda")
