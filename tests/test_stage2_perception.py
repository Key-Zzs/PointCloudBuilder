"""CPU-only contracts for additive Paper-A Stage-2 perception primitives."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from pointcloud_builder import PointCloudBuilder
from pointcloud_builder.camera_model import CameraIntrinsics
from pointcloud_builder.config import SamplingConfig, parse_config
from pointcloud_builder.instance import build_instance_dense, build_instance_sparse
from pointcloud_builder.projection import lift_binary_mask, project_points_to_color_image
from pointcloud_builder.segmentation import SegmentationSidecar, decode_rle, encode_rle
from pointcloud_builder.segmentation.types import InstanceMask, SegmentationProvenance
from pointcloud_builder.support_plane import (
    estimate_episode_support_plane,
    estimate_support_plane,
    filter_support_plane,
    load_support_plane,
    save_support_plane,
)


def _raw_config(*, support: bool = False) -> dict[str, object]:
    config: dict[str, object] = {
        "device": "cpu",
        "camera": {
            "name": "synthetic",
            "depth_scale": 1.0,
            "aligned_depth_to_color": True,
            "color_intrinsics": {"width": 4, "height": 4, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
            "depth_intrinsics": {"width": 4, "height": 4, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        },
        "pointcloud": {"use_rgb": False, "output_format": "xyz"},
        "crop": {"enabled": True, "x": [-10, 10], "y": [-10, 10], "z": [0.1, 4.0]},
        "sampling": {"enabled": True, "mode": "voxel_random", "num_points": 8, "seed": 7, "deterministic": True},
    }
    if support:
        config["pipeline"] = {"profile": "table_filtered"}
        config["support_plane"] = {"enabled": True, "model_source": "estimate_episode", "distance_threshold_m": 0.01}
    return config


def _frame() -> dict[str, object]:
    depth = np.asarray(
        [[1.0, 1.0, 1.0, 1.0], [1.0, 1.1, 1.1, 1.0], [1.0, 1.1, 1.1, 1.0], [1.0, 1.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    return {"depth": depth, "global_frame_index": 3}


def test_legacy_parity_is_unchanged_when_stage2_features_are_disabled() -> None:
    config = parse_config(_raw_config())
    first = PointCloudBuilder(config)
    second = PointCloudBuilder(config)
    cloud_a, meta_a = first.from_recorded_frame(_frame())
    cloud_b, meta_b = second.from_recorded_frame(_frame())
    stages, stage_meta = first.build_stages(_frame())
    assert torch.equal(cloud_a, cloud_b)
    assert meta_a == meta_b
    assert {key: value for key, value in stage_meta.items() if key != "mode"} == {key: value for key, value in meta_a.items() if key != "mode"}
    assert tuple(stages) == ("raw", "cropped", "sampled")
    assert torch.equal(cloud_a, stages["sampled"])


def test_support_plane_tilt_noise_consensus_filter_and_cache(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    xy = rng.uniform(-0.3, 0.3, size=(700, 2))
    # z = 0.7 + 0.15x - 0.08y is not horizontal in the camera frame.
    plane = np.column_stack((xy, 0.7 + 0.15 * xy[:, 0] - 0.08 * xy[:, 1] + rng.normal(0, 0.001, len(xy))))
    object_points = rng.normal([0.0, 0.0, 0.95], 0.01, size=(40, 3))
    fitted = estimate_episode_support_plane(
        [(0, np.vstack((plane, object_points))), (12, np.vstack((plane + rng.normal(0, 0.0005, plane.shape), object_points)))],
        distance_threshold_m=0.006,
    )
    assert fitted.inlier_ratio > 0.85
    assert fitted.residual_p95 < 0.005
    remaining, keep = filter_support_plane(torch.as_tensor(np.vstack((plane, object_points)), dtype=torch.float32), fitted)
    assert int(keep.sum()) >= 35
    assert int(keep.sum()) < 100
    cache = tmp_path / "episode_0_plane.json"
    save_support_plane(cache, fitted, episode_index=0)
    assert load_support_plane(cache).to_dict() == fitted.to_dict()


def test_projection_and_mask_lifting_work_for_xyz_only() -> None:
    points = torch.tensor([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [-1.0, 0.0, 1.0]])
    projection = project_points_to_color_image(points, extrinsics=None, intrinsics=CameraIntrinsics(4, 4, 1.0, 1.0, 0.0, 0.0))
    mask = torch.zeros((4, 4), dtype=torch.bool)
    mask[1, 1] = True
    lifted, selected = lift_binary_mask(points, projection, mask)
    assert selected.tolist() == [True, False, False]
    assert torch.equal(lifted, points[:1])


def _mask(frame: int, concept: str, track: str, region: tuple[slice, slice]) -> InstanceMask:
    binary = np.zeros((4, 4), dtype=bool)
    binary[region] = True
    return InstanceMask(frame, 0, track, concept, binary, (0, 0, 3, 3), 0.9, "text", concept, True)


def test_mask_lifting_modes_record_variable_instances_and_information_gap() -> None:
    points = torch.tensor([[1.0, 1.0, 1.1], [1.0, 2.0, 1.1], [2.0, 1.0, 1.1], [2.0, 2.0, 1.1]])
    projection = project_points_to_color_image(points, extrinsics=None, intrinsics=CameraIntrinsics(4, 4, 1.0, 1.0, 0.0, 0.0))
    masks = [_mask(0, "ring", "r-1", (slice(1, 3), slice(1, 3))), _mask(0, "peg", "p-1", (slice(1, 2), slice(1, 2)))]
    sample = SamplingConfig(mode="fps", num_points=8, deterministic=True, seed=1)
    dense = build_instance_dense(raw_dense_points=points, projection=projection, masks=masks, sampling_config=sample, support_plane=None, expected_instances={"ring": 1, "peg": 1})
    sparse = build_instance_sparse(workspace_sampled_points=points[:1], projection=project_points_to_color_image(points[:1], extrinsics=None, intrinsics=CameraIntrinsics(4, 4, 1.0, 1.0, 0.0, 0.0)), masks=masks, sampling_config=sample, expected_instances={"ring": 1, "peg": 1})
    assert not dense.expected_instance_violations and len(dense.instances) == 2
    assert dense.instances[0].source_dense_point_count == 4
    assert sparse.instances[0].source_dense_point_count == 1
    assert sparse.instances[1].source_dense_point_count == 1
    assert dense.instances[0].sampled_points.shape == (8, 3)
    assert dense.instances[0].sampled_unique_point_count == 4
    assert dense.instances[0].source_stage_point_count == 4
    assert dense.instances[0].object_selected_count == 4
    assert dense.instances[0].object_selected_ratio == pytest.approx(1.0)
    assert dense.instances[0].padded_count == 4
    # Mode2 counts the policy-view sample *before* mask selection and makes
    # repeat padding explicit instead of masquerading it as object support.
    assert sparse.instances[0].source_stage_point_count == 1
    assert sparse.instances[0].object_selected_count == 1
    assert sparse.instances[0].object_selected_ratio == pytest.approx(1.0)
    assert sparse.instances[0].padded_count == 7


def test_missing_or_multiple_expected_instances_are_never_silently_selected() -> None:
    points = torch.tensor([[1.0, 1.0, 1.0]])
    projection = project_points_to_color_image(points, extrinsics=None, intrinsics=CameraIntrinsics(4, 4, 1.0, 1.0, 0.0, 0.0))
    result = build_instance_dense(
        raw_dense_points=points,
        projection=projection,
        masks=[_mask(0, "ring", "r-1", (slice(1, 2), slice(1, 2))), _mask(0, "ring", "r-2", (slice(1, 2), slice(1, 2)))],
        sampling_config=SamplingConfig(mode="fps", num_points=2),
        support_plane=None,
        expected_instances={"ring": 1, "peg": 1},
    )
    assert len(result.instances) == 2
    assert any("ring" in value and "observed=2" in value for value in result.expected_instance_violations)
    assert any("peg" in value and "observed=0" in value for value in result.expected_instance_violations)


@pytest.mark.parametrize("starts", [False, True])
def test_rle_roundtrip(starts: bool) -> None:
    mask = np.asarray([[starts, starts, not starts], [not starts, True, False]], dtype=bool)
    runs = encode_rle(mask)
    assert np.array_equal(decode_rle(runs, shape=mask.shape, starts_with_one=bool(mask.flat[0])), mask)


def test_sidecar_frame_join_and_provenance(tmp_path: Path) -> None:
    zarr = pytest.importorskip("zarr")
    del zarr
    sidecar = tmp_path / "masks.zarr"
    record = _mask(7, "ring", "sam-track-9", (slice(1, 2), slice(1, 2)))
    provenance = SegmentationProvenance("sidecar", "abc", "facebook/sam3.1", "def", "ghi", "jkl")
    SegmentationSidecar.write(sidecar, masks=[record], provenance=provenance, expected_frame_indices=[7])
    restored = SegmentationSidecar.index_by_frame(sidecar, expected_frame_indices=[7])
    assert restored[7][0].track_id == "sam-track-9"
    with pytest.raises(ValueError, match="source frame mismatch"):
        SegmentationSidecar.index_by_frame(sidecar, expected_frame_indices=[7, 8])


def test_streaming_sidecar_writes_without_retaining_mask_video(tmp_path: Path) -> None:
    pytest.importorskip("zarr")
    sidecar = tmp_path / "stream.zarr"
    provenance = SegmentationProvenance("sidecar", "abc", "facebook/sam3.1", "def", "ghi", "jkl")
    with SegmentationSidecar.open_stream_writer(sidecar, provenance=provenance, expected_frame_indices=[0, 1]) as writer:
        writer.add(_mask(0, "ring", "sam-track-1", (slice(1, 2), slice(1, 2))))
        writer.add(_mask(1, "ring", "sam-track-1", (slice(2, 3), slice(2, 3))))
    indexed = SegmentationSidecar.index_by_frame(sidecar, expected_frame_indices=[0, 1])
    assert sorted(indexed) == [0, 1]


def test_builder_requires_episode_plane_and_leaves_legacy_metadata_untouched() -> None:
    legacy = PointCloudBuilder(parse_config(_raw_config()))
    _, legacy_meta = legacy.from_recorded_frame(_frame())
    stage2 = PointCloudBuilder(parse_config(_raw_config(support=True)))
    with pytest.raises(ValueError, match="requires depth_source.mode=ffs_stereo"):
        stage2.build_unfiltered_perception_stages(_frame())
    with pytest.raises(RuntimeError, match="no episode/precomputed model"):
        stage2.from_recorded_frame(_frame())
    points, _ = legacy.build_stages(_frame())
    plane = estimate_support_plane(points["cropped"], distance_threshold_m=0.01)
    stage2.set_support_plane(plane)
    _, stage2_meta = stage2.from_recorded_frame(_frame())
    assert "support_plane" not in legacy_meta
    assert stage2_meta["support_plane"]["enabled"] is True


def test_segmentation_execution_is_optional_and_explicitly_typed() -> None:
    sidecar = parse_config(_raw_config())
    assert sidecar.segmentation.execution == "sidecar"
    in_process_raw = _raw_config()
    in_process_raw["segmentation"] = {"execution": "in_process"}
    assert parse_config(in_process_raw).segmentation.execution == "in_process"
    invalid_raw = _raw_config()
    invalid_raw["segmentation"] = {"execution": "subprocess"}
    with pytest.raises(ValueError, match="segmentation.execution"):
        parse_config(invalid_raw)
