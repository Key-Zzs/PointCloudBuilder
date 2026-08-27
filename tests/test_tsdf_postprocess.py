from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from pointcloud_builder.config import CropConfig, SamplingConfig
from pointcloud_builder.mapping.config import (
    TsdfPostprocessConfig,
    parse_tsdf_config,
)
from pointcloud_builder.mapping.postprocess import postprocess_extracted_cloud
from pointcloud_builder.mapping.open3d.extraction import _canonical_point_order


def _base_config() -> dict:
    return {
        "schema_version": "pointcloud-builder.tsdf.v1",
        "backend": {"type": "open3d_tensor", "device": "CPU:0"},
        "volume": {
            "voxel_size_m": 0.01,
            "block_resolution": 8,
            "block_count": 100,
            "trunc_voxel_multiplier": 4.0,
        },
        "depth": {"minimum_m": 0.1, "maximum_m": 2.0},
        "integration": {
            "source": "ffs_stereo",
            "frame_stride": 1,
            "maximum_weight": None,
            "queue_capacity": 2,
            "maximum_update_hz": 5.0,
            "maximum_mesh_hz": 1.0,
        },
        "extraction": {
            "weight_threshold": 1.0,
            "point_cloud": True,
            "triangle_mesh": True,
        },
        "dynamic": {
            "mode": "build_static",
            "residual_threshold_m": 0.02,
            "persistence_frames": 10,
            "consistency_tolerance_m": 0.01,
            "integrate_background_consistent": True,
            "integrate_persistent_new_surface": True,
        },
    }


def _postprocess(*, crop: bool, sampling: bool) -> TsdfPostprocessConfig:
    return TsdfPostprocessConfig(
        crop=CropConfig(
            enabled=crop,
            frame="workspace",
            x=(-0.5, 0.5),
            y=(-0.5, 0.5),
            z=(-0.5, 0.5),
        ),
        sampling=SamplingConfig(
            enabled=sampling,
            mode="voxel_fps",
            num_points=4,
            voxel_size=0.01,
            seed=42,
            deterministic=True,
            pad_mode="repeat",
        ),
    )


@pytest.mark.parametrize(
    ("crop", "sampling", "cropped_count", "sampled_count"),
    [
        (False, False, 3, 3),
        (True, False, 2, 2),
        (False, True, 3, 4),
        (True, True, 2, 4),
    ],
)
def test_tsdf_postprocess_combinations_reuse_workspace_contracts(
    crop: bool, sampling: bool, cropped_count: int, sampled_count: int
) -> None:
    points = np.asarray(
        [[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.75, 0.0, 0.0]],
        dtype=np.float32,
    )
    result = postprocess_extracted_cloud(
        points,
        workspace_frame="workspace",
        config=_postprocess(crop=crop, sampling=sampling),
    )
    assert result.raw.frame == result.cropped.frame == result.sampled.frame == "workspace"
    assert result.raw.points.shape == (3, 3)
    assert result.cropped.points.shape == (cropped_count, 3)
    assert result.sampled.points.shape == (sampled_count, 3)
    assert torch.isfinite(result.sampled.points).all()


@pytest.mark.parametrize("pad_mode", ["repeat", "zero"])
def test_tsdf_postprocess_empty_and_small_cloud_follow_existing_padding(
    pad_mode: str,
) -> None:
    config = _postprocess(crop=False, sampling=True)
    config = replace(
        config,
        sampling=replace(config.sampling, pad_mode=pad_mode),  # type: ignore[arg-type]
    )
    empty = postprocess_extracted_cloud(
        np.empty((0, 3), dtype=np.float32),
        workspace_frame="workspace",
        config=config,
    )
    assert empty.sampled.points.shape == (4, 3)
    assert torch.count_nonzero(empty.sampled.points) == 0

    one = postprocess_extracted_cloud(
        np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32),
        workspace_frame="workspace",
        config=config,
    )
    assert one.sampled.points.shape == (4, 3)
    if pad_mode == "repeat":
        assert torch.all(one.sampled.points == one.sampled.points[0])
    else:
        assert torch.count_nonzero(one.sampled.points[1:]) == 0


def test_tsdf_postprocess_sampling_is_deterministic() -> None:
    points = np.random.default_rng(7).normal(size=(32, 3)).astype(np.float32)
    config = _postprocess(crop=False, sampling=True)
    first = postprocess_extracted_cloud(
        points, workspace_frame="workspace", config=config
    )
    second = postprocess_extracted_cloud(
        points, workspace_frame="workspace", config=config
    )
    assert torch.equal(first.sampled.points, second.sampled.points)


def test_backend_point_order_is_canonical_before_deterministic_sampling() -> None:
    points = np.asarray(
        [[0.2, 0.0, 0.0], [0.1, 0.2, 0.0], [0.1, 0.1, 0.3]],
        dtype=np.float32,
    )
    assert np.array_equal(
        _canonical_point_order(points),
        _canonical_point_order(points[::-1]),
    )


def test_old_tsdf_config_defaults_both_postprocess_stages_off() -> None:
    config = parse_tsdf_config(_base_config())
    assert not config.postprocess.crop.enabled
    assert not config.postprocess.sampling.enabled


def test_tsdf_postprocess_config_is_strict_and_workspace_only() -> None:
    raw = _base_config()
    raw["postprocess"] = {
        "crop": {
            "enabled": True,
            "frame": "camera",
            "x": [-1.0, 1.0],
            "y": [-1.0, 1.0],
            "z": [-1.0, 1.0],
        }
    }
    with pytest.raises(ValueError, match="workspace"):
        parse_tsdf_config(raw)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_tsdf_postprocess_cuda_matches_cpu() -> None:
    points = np.random.default_rng(11).normal(size=(64, 3)).astype(np.float32)
    config = _postprocess(crop=True, sampling=True)
    cpu = postprocess_extracted_cloud(
        points, workspace_frame="workspace", config=config, device="CPU:0"
    )
    cuda = postprocess_extracted_cloud(
        points, workspace_frame="workspace", config=config, device="CUDA:0"
    )
    assert torch.equal(cpu.sampled.points, cuda.sampled.points.cpu())
