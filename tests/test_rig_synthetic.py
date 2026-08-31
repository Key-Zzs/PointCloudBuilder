from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest
import torch

from pointcloud_builder.mapping.depth_packet import canonical_bundle_sha256
from pointcloud_builder.rig import (
    build_synthetic_rig,
    create_synthetic_scene,
    parse_rig_config,
)
from pointcloud_builder.rig_calibration.deployment import (
    canonical_json_sha256,
    deployment_fingerprint,
)


def _raw(names: tuple[str, ...], *, timing_mode: str = "exact_index") -> dict:
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
            "mode": timing_mode,
            "maximum_skew_ms": 5.0,
            "reference_camera": "camera_a",
        },
        "workspace_crop": {
            "enabled": True,
            "x": [-0.72, 0.72],
            "y": [-0.58, 0.58],
            "z": [-0.01, 0.55],
        },
        "fusion": {"enabled": False},
        "sampling": {
            "enabled": True,
            "mode": "voxel_fps",
            "num_points": 1024,
            "voxel_size": 0.01,
            "deterministic": True,
            "seed": 17,
        },
    }


def _assert_scene_geometry(result) -> None:
    assert result.workspace_cropped.frame == "workspace"
    for item in result.per_camera_workspace_clouds:
        points = item.cloud.points[:, :3]
        plane = points[points[:, 2].abs() < 0.004]
        top = points[
            (points[:, 0].abs() < 0.16)
            & (points[:, 1].abs() < 0.10)
            & (points[:, 2] > 0.20)
        ]
        assert len(plane) > 500
        assert len(top) > 20
        assert float(plane[:, 2].abs().median()) < 0.0015
        assert abs(float(top[:, 2].median()) - 0.25) < 0.002
        known_plane = torch.tensor((0.50, 0.40, 0.0), dtype=points.dtype)
        assert float(torch.linalg.norm(plane - known_plane, dim=1).min()) < 0.035
        known_box = torch.tensor((0.0, 0.0, 0.25), dtype=points.dtype)
        assert float(torch.linalg.norm(top - known_box, dim=1).min()) < 0.035


def test_full_pipeline_supports_one_two_three_and_four_analytic_cameras() -> None:
    all_names = ("camera_a", "camera_b", "camera_c", "camera_d")
    scene = create_synthetic_scene(all_names)
    for count in (1, 2, 3, 4):
        names = all_names[:count]
        result = build_synthetic_rig(parse_rig_config(_raw(names)), scene).build(1)
        assert result.canonical_camera_order == tuple(sorted(names))
        assert [item.camera_name for item in result.per_camera_workspace_clouds] == sorted(names)
        assert result.sampled.points.shape == (1024, 3)
        assert result.sampled.metadata["pre_sampling_count"] == sum(
            item.cloud.points.shape[0] for item in result.per_camera_workspace_clouds
        )
        _assert_scene_geometry(result)


def test_three_camera_deployed_multipose_runtime_and_provenance(tmp_path) -> None:
    names = ("camera_a", "camera_b", "camera_c")
    scene = create_synthetic_scene(names)
    target_identity = {"kind": "synthetic-grid"}
    cameras = {}
    for name in names:
        identity = scene.bundles[name].device.to_dict()
        cameras[name] = {
            "camera_id": name,
            "camera_identity": identity,
            "camera_identity_sha256": canonical_json_sha256(identity),
            "camera_bundle_sha256": canonical_bundle_sha256(scene.bundles[name]),
            "projection_frame": f"{name}/color_optical",
            "T_workspace_from_camera": scene.poses[name].tolist(),
        }
    artifact = {
        "schema_version": "pointcloud-builder.rig-calibration-deployment.v1",
        "status": "deployed",
        "workspace_frame": "workspace",
        "target_identity": target_identity,
        "target_identity_sha256": canonical_json_sha256(target_identity),
        "solution_fingerprint": "a" * 64,
        "validation_sha256": "b" * 64,
        "physical_acceptance_sha256": "c" * 64,
        "source_receipts": {
            "solution": {},
            "validation": {},
            "physical_acceptance": {},
        },
        "cameras": cameras,
        "creation_metadata": {"synthetic": True},
    }
    artifact["rig_calibration_fingerprint"] = deployment_fingerprint(artifact)
    deployment = tmp_path / "rig_calibration.json"
    deployment.write_text(json.dumps(artifact), encoding="utf-8")
    raw = _raw(names)
    raw["rig_calibration"] = {
        "type": "validated_multipose",
        "artifact": str(deployment),
    }
    result = build_synthetic_rig(parse_rig_config(raw), scene).build(0)
    _assert_scene_geometry(result)
    for observation in result.depth_frame_set.observations:
        assert observation.calibration_mode == "validated_multipose_deployment"
        assert observation.rig_calibration_fingerprint == artifact[
            "rig_calibration_fingerprint"
        ]
        assert observation.solution_fingerprint == "a" * 64
        assert observation.camera_bundle_sha256 == cameras[observation.camera_name][
            "camera_bundle_sha256"
        ]
    assert all(
        value["production_applied"] is True
        for value in result.per_camera_provenance.values()
    )


def test_analytic_renderer_produces_pose_specific_depth_images() -> None:
    scene = create_synthetic_scene(("camera_a", "camera_b", "camera_c"))
    depths = [scene.frames[name][0].depth.data for name in sorted(scene.frames)]
    assert not np.array_equal(depths[0], depths[1])
    assert not np.array_equal(depths[1], depths[2])
    assert not np.array_equal(depths[0], depths[2])


def test_missing_bundle_and_output_frame_mismatch_fail_closed() -> None:
    scene = create_synthetic_scene(("camera_a",))
    with pytest.raises(ValueError, match="missing bundle"):
        build_synthetic_rig(
            parse_rig_config(_raw(("camera_a", "camera_b"))), scene
        )
    raw = _raw(("camera_a",))
    raw["output_frame"] = "world"
    with pytest.raises(ValueError, match="output frame differs"):
        build_synthetic_rig(parse_rig_config(raw), scene)


def test_yaml_camera_order_does_not_change_identity_or_concatenated_output() -> None:
    names = ("camera_a", "camera_b", "camera_c")
    scene = create_synthetic_scene(names)
    raw = _raw(names)
    reversed_raw = deepcopy(raw)
    reversed_raw["cameras"] = list(reversed(reversed_raw["cameras"]))
    forward = build_synthetic_rig(parse_rig_config(raw), scene).build(0)
    reverse = build_synthetic_rig(parse_rig_config(reversed_raw), scene).build(0)
    assert forward.canonical_camera_order == reverse.canonical_camera_order == names
    assert forward.sampled.metadata == reverse.sampled.metadata
    assert torch.equal(
        forward.sampled.points,
        reverse.sampled.points,
    )
    for left, right in zip(
        forward.per_camera_workspace_clouds, reverse.per_camera_workspace_clouds, strict=True
    ):
        assert left.camera_name == right.camera_name
        assert torch.equal(left.cloud.points, right.cloud.points)


def test_nearest_host_timestamp_pipeline_reports_reference_deltas() -> None:
    names = ("camera_a", "camera_b", "camera_c")
    scene = create_synthetic_scene(
        names, timestamp_offsets_ns={"camera_b": 4_000_000, "camera_c": -3_000_000}
    )
    result = build_synthetic_rig(
        parse_rig_config(_raw(names, timing_mode="nearest_host_timestamp")), scene
    ).build(1)
    assert result.frame_match.reference_camera == "camera_a"
    assert result.frame_match.per_camera_delta_ms == {
        "camera_a": 0.0,
        "camera_b": 4.0,
        "camera_c": -3.0,
    }
    assert result.frame_match.maximum_skew_ms == 4.0
