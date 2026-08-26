from __future__ import annotations

from copy import deepcopy

import torch

from pointcloud_builder.fusion import synthetic_geometry_metrics
from pointcloud_builder.rig import build_synthetic_rig, create_synthetic_scene, parse_rig_config


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


def test_rig_fusion_exposes_all_stages_and_never_samples_per_camera_early() -> None:
    names = ("camera_a", "camera_b", "camera_c")
    result = build_synthetic_rig(parse_rig_config(_config(names)), create_synthetic_scene(names)).build(1)
    assert len(result.per_camera_camera_frame) == len(result.per_camera_workspace) == 3
    assert result.concatenated.points.shape[0] == 3 * 96 * 72
    assert result.workspace_cropped.points.shape[0] == sum(
        item.cloud.points.shape[0] for item in result.per_camera_workspace
    )
    assert result.fusion_provenance.input_point_count == result.workspace_cropped.points.shape[0]
    assert result.fused.points.shape[0] == result.fusion_provenance.output_voxel_count
    assert result.fused.points.shape[0] < 0.60 * result.workspace_cropped.points.shape[0]
    assert result.sampled.points.shape == (1024, 3)
    assert all(
        not runtime.pipeline.context.builder.config.sampling.enabled
        for runtime in build_synthetic_rig(
            parse_rig_config(_config(names)), create_synthetic_scene(names)
        ).runtimes.values()
    )


def test_fused_and_sampled_outputs_ignore_yaml_camera_order() -> None:
    names = ("camera_a", "camera_b", "camera_c")
    scene = create_synthetic_scene(names)
    raw = _config(names)
    reversed_raw = deepcopy(raw)
    reversed_raw["cameras"] = list(reversed(reversed_raw["cameras"]))
    forward = build_synthetic_rig(parse_rig_config(raw), scene).build(0)
    reverse = build_synthetic_rig(parse_rig_config(reversed_raw), scene).build(0)
    assert torch.equal(forward.fused.points, reverse.fused.points)
    assert torch.equal(forward.sampled.points, reverse.sampled.points)
    assert forward.fusion_provenance.to_summary() == reverse.fusion_provenance.to_summary()


def test_synthetic_fusion_reduces_duplicates_without_geometry_regression() -> None:
    names = ("camera_a", "camera_b", "camera_c")
    result = build_synthetic_rig(parse_rig_config(_config(names)), create_synthetic_scene(names)).build(2)
    metrics = synthetic_geometry_metrics(
        result.workspace_cropped.points, result.fused.points, voxel_size_m=0.015
    )
    assert metrics["duplicate_surface_thickness_after_m"] <= metrics[
        "duplicate_surface_thickness_before_m"
    ]
    assert metrics["voxel_occupancy_reduction"] > 0.35
    assert metrics["point_to_plane_after_median_m"] < 0.0015
    assert metrics["plane_signed_bias_shift_m"] < 0.001
    assert metrics["box_top_systematic_shift_m"] < 0.005
    assert metrics["nearest_neighbor_residual_p95_m"] < 0.04
    assert metrics["completeness_at_1_5_voxels"] > 0.90


def test_one_camera_fallback_and_two_camera_fusion() -> None:
    scene = create_synthetic_scene(("camera_a", "camera_b"))
    for names in (("camera_a",), ("camera_a", "camera_b")):
        result = build_synthetic_rig(parse_rig_config(_config(names)), scene).build(0)
        assert result.fusion_provenance.input_point_count == result.workspace_cropped.points.shape[0]
        assert result.sampled.points.shape == (1024, 3)
        plane = result.fused.points[result.fused.points[:, 2].abs() < 0.01]
        assert float(plane[:, 2].abs().median()) < 0.0015
