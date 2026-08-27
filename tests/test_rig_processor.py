from __future__ import annotations

from dataclasses import replace

import torch

from pointcloud_builder.rig import (
    build_synthetic_rig,
    create_synthetic_scene,
    parse_rig_config,
)
from pointcloud_builder.rig.frame_matcher import match_exact_index
from pointcloud_builder.rig.processor import RigFrameProcessor
from pointcloud_builder.rig.types import CameraFrameEnvelope, RigFrameSet


def _config(names: tuple[str, ...]) -> dict:
    return {
        "schema_version": "pointcloud-builder.rig.v1",
        "output_frame": "workspace",
        "cameras": [
            {
                "name": name,
                "enabled": True,
                "source": {
                    "type": "synthetic",
                    "capture_artifact": f"synthetic://{name}",
                    "provision_artifact": f"synthetic://{name}/bundle",
                },
                "depth": {"mode": "native"},
                "pipeline_config": None,
                "local_crop": {"enabled": False},
            }
            for name in names
        ],
        "timing": {
            "mode": "exact_index",
            "maximum_skew_ms": 5.0,
            "reference_camera": "camera_a",
        },
        "workspace_crop": {
            "enabled": True,
            "x": [-0.72, 0.72],
            "y": [-0.58, 0.58],
            "z": [-0.01, 0.55],
        },
        "fusion": {
            "enabled": True,
            "voxel_size_m": 0.015,
            "origin": [-0.75, -0.60, -0.02],
            "deterministic": True,
        },
        "sampling": {
            "enabled": True,
            "mode": "voxel_fps",
            "num_points": 1024,
            "voxel_size": 0.01,
            "deterministic": True,
            "seed": 23,
        },
    }


def test_shared_processor_matches_offline_pipeline_outputs() -> None:
    names = ("camera_a", "camera_b", "camera_c")
    config = parse_rig_config(_config(names))
    rig = build_synthetic_rig(config, create_synthetic_scene(names))
    sources = {name: runtime.source for name, runtime in rig.runtimes.items()}
    frame_set = match_exact_index(sources, 1, reference_camera="camera_a")

    direct = RigFrameProcessor(config, rig.runtimes).process_frame_set(frame_set)
    offline = rig.build(1)

    assert direct.canonical_camera_order == offline.canonical_camera_order == names
    assert torch.equal(direct.concatenated.points, offline.concatenated.points)
    assert torch.equal(
        direct.workspace_cropped.points, offline.workspace_cropped.points
    )
    assert torch.equal(direct.fused.points, offline.fused.points)
    assert torch.equal(direct.sampled.points, offline.sampled.points)
    assert direct.sampled.metadata == offline.sampled.metadata
    assert direct.per_camera_stage_statistics == offline.per_camera_stage_statistics
    assert direct.processing_metadata == offline.processing_metadata
    assert (
        direct.fusion_provenance.to_summary() == offline.fusion_provenance.to_summary()
    )
    for direct_camera, offline_camera in zip(
        direct.per_camera_workspace, offline.per_camera_workspace, strict=True
    ):
        assert direct_camera.camera_name == offline_camera.camera_name
        assert torch.equal(direct_camera.cloud.points, offline_camera.cloud.points)


def test_processor_reports_scalar_stage_counts_and_m6_stage_order() -> None:
    names = ("camera_a", "camera_b")
    config = parse_rig_config(_config(names))
    rig = build_synthetic_rig(config, create_synthetic_scene(names))
    result = rig.build(0)

    statistics = result.per_camera_stage_statistics
    assert tuple(sorted(statistics)) == names
    assert result.sampled.metadata["per_camera_stage_statistics"] == statistics
    assert result.sampled.metadata["processing_metadata"] == result.processing_metadata
    assert not _contains_tensor(statistics)
    assert not _contains_tensor(result.processing_metadata)

    raw_count = sum(item["workspace_raw_point_count"] for item in statistics.values())
    cropped_count = sum(
        item["workspace_cropped_point_count"] for item in statistics.values()
    )
    assert raw_count == result.concatenated.points.shape[0]
    assert cropped_count == result.workspace_cropped.points.shape[0]
    assert cropped_count == result.fusion_provenance.input_point_count
    assert result.processing_metadata == {
        "processor": "RigFrameProcessor",
        "camera_count": 2,
        "canonical_camera_order": ["camera_a", "camera_b"],
        "per_camera_processing": "sequential_canonical_order",
        "concatenation_input_stage": "per_camera_workspace_raw",
        "workspace_crop_stage": (
            "once_per_camera_after_workspace_transform_before_concatenation"
        ),
        "fusion_input_stage": "workspace_cropped_concatenation",
        "global_sampling_input_stage": "fused",
        "geometry_aggregation": "centroid",
        "rgb_aggregation": "mean",
        "point_counts": {
            "concatenated": result.concatenated.points.shape[0],
            "workspace_cropped": result.workspace_cropped.points.shape[0],
            "fused": result.fused.points.shape[0],
            "sampled": result.sampled.points.shape[0],
        },
    }


def test_rig_frame_set_old_constructor_derives_compatible_match_metadata() -> None:
    envelope = CameraFrameEnvelope(
        camera_name="camera_a",
        frame_index=7,
        host_receive_timestamp_ns=1_234_567_890,
        frame=object(),
    )
    original_envelopes = {"camera_a": envelope}
    original_deltas = {"camera_a": -3.5}
    frame_set = RigFrameSet(
        original_envelopes,
        "camera_a",
        original_deltas,
        3.5,
        (),
    )
    original_envelopes.clear()
    original_deltas.clear()

    assert frame_set.matched_camera_names == ("camera_a",)
    assert frame_set.match_sequence_index is None
    assert frame_set.match_timestamp_ns == 1_234_567_890
    assert frame_set.per_camera_delta_ms == {"camera_a": -3.5}
    assert frame_set.per_camera_absolute_delta_ms == {"camera_a": 3.5}
    assert frame_set.matching_policy is None
    assert frame_set.metadata == {}

    enriched = replace(
        frame_set,
        match_sequence_index=11,
        matching_policy="nearest_host_timestamp",
        metadata={"source": "live"},
    )
    assert enriched.match_sequence_index == 11
    assert enriched.matching_policy == "nearest_host_timestamp"
    assert enriched.metadata == {"source": "live"}


def test_processor_rejects_unmatched_or_incomplete_frame_sets() -> None:
    names = ("camera_a", "camera_b")
    config = parse_rig_config(_config(names))
    rig = build_synthetic_rig(config, create_synthetic_scene(names))
    sources = {name: runtime.source for name, runtime in rig.runtimes.items()}
    complete = match_exact_index(sources, 0, reference_camera="camera_a")
    processor = RigFrameProcessor(config, rig.runtimes)

    unmatched = replace(complete, unmatched_cameras=("camera_b",))
    try:
        processor.process_frame_set(unmatched)
    except ValueError as error:
        assert "unmatched cameras" in str(error)
    else:
        raise AssertionError("processor accepted an unmatched frame set")

    incomplete = replace(
        complete,
        envelopes={"camera_a": complete.envelopes["camera_a"]},
    )
    try:
        processor.process_frame_set(incomplete)
    except ValueError as error:
        assert "frame set cameras mismatch" in str(error)
    else:
        raise AssertionError("processor accepted an incomplete frame set")


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_tensor(item) for item in value)
    return False
