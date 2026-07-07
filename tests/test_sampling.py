from __future__ import annotations

import pytest
import torch

from pointcloud_builder.config import SamplingConfig
from pointcloud_builder.sampling import sample_point_cloud, sample_points


@pytest.mark.parametrize("mode", ["fps", "stride", "random", "voxel", "voxel_random", "voxel_fps"])
def test_sampling_modes_return_fixed_num_points(mode: str) -> None:
    point_cloud = torch.rand((32, 3), dtype=torch.float32)
    config = SamplingConfig(
        enabled=True,
        mode=mode,  # type: ignore[arg-type]
        num_points=8,
        stride=2,
        voxel_size=0.05,
        seed=42,
        deterministic=True,
        pad_mode="repeat",
    )
    sampled, meta = sample_point_cloud(point_cloud, config)
    assert sampled.shape == (8, 3)
    assert meta["sampled_count"] == 8
    assert meta["sampling_mode"] == mode


def test_sampling_preserves_xyzrgb_shape() -> None:
    xyz = torch.rand((16, 3), dtype=torch.float32)
    rgb = torch.rand((16, 3), dtype=torch.float32)
    point_cloud = torch.cat([xyz, rgb], dim=-1)
    sampled, meta = sample_point_cloud(
        point_cloud,
        SamplingConfig(enabled=True, mode="random", num_points=10, seed=1, deterministic=True),
    )
    assert sampled.shape == (10, 6)
    assert meta["sampled_count"] == 10


def test_repeat_padding_when_input_is_smaller() -> None:
    point_cloud = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    sampled, meta = sample_point_cloud(
        point_cloud,
        SamplingConfig(enabled=True, mode="stride", num_points=5, pad_mode="repeat"),
    )
    assert sampled.shape == (5, 3)
    assert torch.allclose(sampled[2], sampled[0])
    assert torch.allclose(sampled[3], sampled[1])
    assert meta["padded"] is True


def test_zero_padding_when_input_is_smaller() -> None:
    point_cloud = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    sampled, meta = sample_point_cloud(
        point_cloud,
        SamplingConfig(enabled=True, mode="stride", num_points=4, pad_mode="zero"),
    )
    assert sampled.shape == (4, 3)
    assert torch.allclose(sampled[1:], torch.zeros((3, 3)))
    assert meta["padded"] is True


def test_empty_input_returns_fixed_zero_tensor() -> None:
    point_cloud = torch.empty((0, 6), dtype=torch.float32)
    sampled, meta = sample_point_cloud(
        point_cloud,
        SamplingConfig(enabled=True, mode="voxel_random", num_points=7, pad_mode="repeat"),
    )
    assert sampled.shape == (7, 6)
    assert torch.allclose(sampled, torch.zeros((7, 6)))
    assert meta["input_empty"] is True
    assert meta["empty_input"] is True


def test_sample_points_backward_compatible_wrapper() -> None:
    points = torch.rand((10, 3), dtype=torch.float32)
    sampled, colors, meta = sample_points(points, SamplingConfig(mode="random", num_points=32))
    assert sampled.shape == (32, 3)
    assert colors is None
    assert meta["sampled_count"] == 32


def test_cuda_sampling_does_not_crash_when_available() -> None:
    if not torch.cuda.is_available():
        return
    point_cloud = torch.rand((64, 6), dtype=torch.float32, device="cuda")
    sampled, meta = sample_point_cloud(
        point_cloud,
        SamplingConfig(enabled=True, mode="voxel_fps", num_points=16, voxel_size=0.02),
    )
    assert sampled.is_cuda
    assert sampled.shape == (16, 6)
    assert meta["sampled_count"] == 16
