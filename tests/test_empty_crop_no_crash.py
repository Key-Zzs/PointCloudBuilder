from __future__ import annotations

import torch

from pointcloud_builder import PointCloudBuilder
from pointcloud_builder.camera_model import CameraIntrinsics
from pointcloud_builder.config import (
    CameraConfig,
    CropConfig,
    PointCloudBuilderConfig,
    PointCloudConfig,
    SamplingConfig,
)
from pointcloud_builder.crop import crop_point_cloud
from pointcloud_builder.sampling import sample_point_cloud


def test_empty_crop_does_not_crash_sampling() -> None:
    points = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32)
    crop = CropConfig(enabled=True, x=(-1.0, 1.0), y=(-1.0, 1.0), z=(0.1, 1.0))
    sampling = SamplingConfig(mode="voxel_random", num_points=8)
    cropped, _ = crop_point_cloud(points, crop)
    sampled, meta = sample_point_cloud(cropped, sampling)
    assert sampled.shape == (8, 3)
    assert meta["empty_input"] is True


def test_builder_empty_crop_returns_fixed_size_without_crashing() -> None:
    intrinsics = CameraIntrinsics(width=3, height=3, fx=1.0, fy=1.0, cx=1.0, cy=1.0)
    config = PointCloudBuilderConfig(
        device="cpu",
        camera=CameraConfig(
            name="test",
            depth_scale=1.0,
            aligned_depth_to_color=False,
            color_intrinsics=intrinsics,
            depth_intrinsics=intrinsics,
        ),
        pointcloud=PointCloudConfig(use_rgb=False, output_format="xyz"),
        crop=CropConfig(enabled=True, frame="camera", x=(10.0, 11.0), y=(10.0, 11.0), z=(10.0, 11.0)),
        sampling=SamplingConfig(enabled=True, mode="voxel_random", num_points=8, pad_mode="repeat"),
    )
    builder = PointCloudBuilder(config)
    pc, meta = builder.from_live_frame({"depth": torch.ones((3, 3), dtype=torch.float32)})
    assert pc.shape == (8, 3)
    assert torch.allclose(pc, torch.zeros((8, 3)))
    assert meta["crop_empty"] is True
    assert meta["input_empty"] is True
    assert meta["num_sampled_points"] == 8
