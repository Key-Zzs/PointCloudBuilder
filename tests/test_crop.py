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


def make_crop() -> CropConfig:
    return CropConfig(
        enabled=True,
        frame="camera",
        x=(-0.5, 0.5),
        y=(-0.5, 0.5),
        z=(0.05, 1.5),
    )


def test_crop_xyz_points_correctly() -> None:
    point_cloud = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.6, 0.0, 1.0],
            [0.0, -0.6, 1.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    cropped, mask = crop_point_cloud(point_cloud, make_crop())
    assert cropped.shape == (1, 3)
    assert torch.allclose(cropped, point_cloud[:1])
    assert mask.tolist() == [True, False, False, False]


def test_crop_xyzrgb_preserves_rgb_columns() -> None:
    point_cloud = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.1, 0.2, 0.3],
            [0.8, 0.0, 1.0, 0.4, 0.5, 0.6],
        ],
        dtype=torch.float32,
    )
    cropped, mask = crop_point_cloud(point_cloud, make_crop())
    assert cropped.shape == (1, 6)
    assert torch.allclose(cropped[0, 3:], torch.tensor([0.1, 0.2, 0.3]))
    assert mask.tolist() == [True, False]


def test_crop_keeps_boundary_points() -> None:
    crop = make_crop()
    point_cloud = torch.tensor(
        [
            [crop.x[0], 0.0, 1.0],
            [crop.x[1], 0.0, 1.0],
            [0.0, crop.y[0], 1.0],
            [0.0, crop.y[1], 1.0],
            [0.0, 0.0, crop.z[0]],
            [0.0, 0.0, crop.z[1]],
        ],
        dtype=torch.float32,
    )
    cropped, mask = crop_point_cloud(point_cloud, crop)
    assert cropped.shape == (6, 3)
    assert mask.tolist() == [True, True, True, True, True, True]


def test_empty_crop_returns_empty_tensor_without_crashing() -> None:
    point_cloud = torch.tensor([[2.0, 2.0, 2.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    cropped, mask = crop_point_cloud(point_cloud, make_crop())
    assert cropped.shape == (0, 6)
    assert mask.tolist() == [False]


def test_crop_disabled_returns_point_cloud_unchanged() -> None:
    point_cloud = torch.rand((8, 6), dtype=torch.float32)
    crop = CropConfig(enabled=False, frame="camera", x=(-0.1, 0.1), y=(-0.1, 0.1), z=(0.1, 0.2))
    cropped, mask = crop_point_cloud(point_cloud, crop)
    assert cropped.data_ptr() == point_cloud.data_ptr()
    assert torch.allclose(cropped, point_cloud)
    assert mask.tolist() == [True] * 8


def test_cpu_and_cuda_results_match_when_cuda_is_available() -> None:
    if not torch.cuda.is_available():
        return
    point_cloud_cpu = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.8, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.01, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    cropped_cpu, mask_cpu = crop_point_cloud(point_cloud_cpu, make_crop())
    cropped_cuda, mask_cuda = crop_point_cloud(point_cloud_cpu.cuda(), make_crop())
    assert torch.allclose(cropped_cpu, cropped_cuda.cpu())
    assert torch.equal(mask_cpu, mask_cuda.cpu())


def test_recorded_and_live_frames_use_same_crop_logic() -> None:
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
        crop=CropConfig(enabled=True, frame="camera", x=(-0.1, 0.1), y=(-0.1, 0.1), z=(0.5, 1.5)),
        sampling=SamplingConfig(enabled=True, mode="stride", num_points=1),
    )
    builder = PointCloudBuilder(config)
    frame = {"depth": torch.ones((3, 3), dtype=torch.float32)}
    recorded, recorded_meta = builder.from_recorded_frame(frame)
    live, live_meta = builder.from_live_frame(frame)
    assert torch.allclose(recorded, live)
    assert recorded.shape == (1, 3)
    assert recorded_meta["num_raw_points"] == 9
    assert recorded_meta["num_cropped_points"] == 1
    assert recorded_meta["num_sampled_points"] == 1
    assert recorded_meta["stage"] == "sampled"
    assert live_meta["num_cropped_points"] == recorded_meta["num_cropped_points"]
