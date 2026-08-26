from __future__ import annotations

import pytest
import torch

from pointcloud_builder.config import SamplingConfig
from pointcloud_builder.fusion import (
    VoxelFusionConfig,
    sample_fused_cloud,
    voxel_fuse_workspace_clouds,
)
from pointcloud_builder.rig import WorkspaceCloud
from pointcloud_builder.workspace import WorkspacePointCloud


def _cloud(name: str, points: torch.Tensor) -> WorkspaceCloud:
    return WorkspaceCloud(name, WorkspacePointCloud(points=points, frame="workspace"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"voxel_size_m": float("nan")},
        {"voxel_size_m": float("inf")},
        {"origin": (0.0, float("nan"), 0.0)},
        {"origin": (0.0, 0.0, float("inf"))},
    ],
)
def test_fusion_config_rejects_nonfinite_values(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="finite"):
        VoxelFusionConfig(enabled=True, **kwargs)


def test_voxel_key_uses_floor_for_negative_coordinates_and_centroid_mean() -> None:
    a = _cloud(
        "camera_a",
        torch.tensor([[-0.001, 0.0, 0.0], [0.002, 0.0, 0.0], [0.021, 0.0, 0.0]]),
    )
    b = _cloud("camera_b", torch.tensor([[0.004, 0.0, 0.0], [0.022, 0.0, 0.0]]))
    result = voxel_fuse_workspace_clouds(
        [b, a], VoxelFusionConfig(enabled=True, voxel_size_m=0.01)
    )
    torch.testing.assert_close(
        result.provenance.voxel_keys,
        torch.tensor([[-1, 0, 0], [0, 0, 0], [2, 0, 0]]),
    )
    torch.testing.assert_close(
        result.cloud.points[:, 0], torch.tensor([-0.001, 0.003, 0.0215]), atol=1e-7, rtol=0
    )
    assert result.provenance.input_point_count == 5
    assert result.provenance.output_voxel_count == 3
    assert result.provenance.per_camera_input_count == {"camera_a": 3, "camera_b": 2}
    assert result.provenance.per_camera_unique_voxel_contribution == {
        "camera_a": 3,
        "camera_b": 2,
    }
    assert result.provenance.multi_camera_voxel_count == 2
    assert result.provenance.per_voxel_source_camera_count.tolist() == [1, 2, 2]
    assert result.provenance.per_voxel_point_count.tolist() == [1, 2, 2]


def test_voxel_fusion_supports_xyzrgb_arithmetic_mean() -> None:
    points = torch.tensor(
        [[0.001, 0.0, 0.0, 1.0, 0.0, 0.0], [0.003, 0.0, 0.0, 0.0, 0.0, 1.0]]
    )
    result = voxel_fuse_workspace_clouds(
        [_cloud("camera_a", points)],
        VoxelFusionConfig(enabled=True, voxel_size_m=0.01),
    )
    torch.testing.assert_close(
        result.cloud.points,
        torch.tensor([[0.002, 0.0, 0.0, 0.5, 0.0, 0.5]]),
        atol=1e-7,
        rtol=0,
    )


def test_empty_one_camera_all_empty_and_different_counts_are_supported() -> None:
    empty = _cloud("camera_a", torch.empty((0, 3)))
    nonempty = _cloud("camera_b", torch.tensor([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]]))
    mixed = voxel_fuse_workspace_clouds(
        [empty, nonempty], VoxelFusionConfig(enabled=True, voxel_size_m=0.01)
    )
    assert mixed.cloud.points.shape == (2, 3)
    assert mixed.provenance.per_camera_input_count == {"camera_a": 0, "camera_b": 2}
    all_empty = voxel_fuse_workspace_clouds(
        [empty], VoxelFusionConfig(enabled=True, voxel_size_m=0.01)
    )
    assert all_empty.cloud.points.shape == (0, 3)
    assert all_empty.provenance.output_voxel_count == 0
    with pytest.raises(ValueError, match="at least one"):
        voxel_fuse_workspace_clouds([], VoxelFusionConfig(enabled=True))


def test_camera_order_permutation_is_bitwise_invariant_on_cpu() -> None:
    generator = torch.Generator().manual_seed(9)
    a = _cloud("camera_a", torch.rand((137, 3), generator=generator) - 0.5)
    b = _cloud("camera_b", torch.rand((83, 3), generator=generator) - 0.5)
    config = VoxelFusionConfig(
        enabled=True, voxel_size_m=0.05, origin=(-0.5, -0.5, -0.5), deterministic=True
    )
    forward = voxel_fuse_workspace_clouds([a, b], config)
    reverse = voxel_fuse_workspace_clouds([b, a], config)
    assert torch.equal(forward.cloud.points, reverse.cloud.points)
    assert torch.equal(forward.provenance.voxel_keys, reverse.provenance.voxel_keys)
    assert forward.provenance.to_summary() == reverse.provenance.to_summary()


@pytest.mark.parametrize("mode", ["voxel", "voxel_random", "voxel_fps", "fps"])
def test_global_sampling_modes_are_fixed_size_and_deterministic(mode: str) -> None:
    points = torch.linspace(-0.5, 0.5, 300).unsqueeze(1).repeat(1, 3)
    cloud = WorkspacePointCloud(points=points, frame="workspace")
    config = SamplingConfig(
        enabled=True,
        mode=mode,  # type: ignore[arg-type]
        num_points=64,
        voxel_size=0.01,
        deterministic=True,
        seed=5,
    )
    first = sample_fused_cloud(cloud, config)
    second = sample_fused_cloud(cloud, config)
    assert first.points.shape == (64, 3)
    assert torch.equal(first.points, second.points)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_fusion_matches_cpu_within_float_tolerance() -> None:
    generator = torch.Generator().manual_seed(11)
    points_a = torch.rand((120, 3), generator=generator) - 0.5
    points_b = torch.rand((73, 3), generator=generator) - 0.5
    config = VoxelFusionConfig(enabled=True, voxel_size_m=0.04, origin=(-0.5, -0.5, -0.5))
    cpu = voxel_fuse_workspace_clouds(
        [_cloud("camera_a", points_a), _cloud("camera_b", points_b)], config
    )
    cuda = voxel_fuse_workspace_clouds(
        [_cloud("camera_b", points_b.cuda()), _cloud("camera_a", points_a.cuda())], config
    )
    torch.testing.assert_close(cuda.cloud.points.cpu(), cpu.cloud.points, atol=3e-6, rtol=1e-6)
    assert torch.equal(cuda.provenance.voxel_keys.cpu(), cpu.provenance.voxel_keys)
