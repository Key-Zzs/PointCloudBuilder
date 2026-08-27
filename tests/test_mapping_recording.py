from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pointcloud_builder.mapping.config import parse_tsdf_config
from pointcloud_builder.mapping.performance import evaluate_rss_plateau
from pointcloud_builder.mapping.recording import (
    RigDepthRecordingWriter,
    iter_rig_depth_recording,
    validate_rig_depth_recording,
)
from pointcloud_builder.rig import (
    build_synthetic_rig,
    create_synthetic_scene,
    parse_rig_config,
)


def _rig_config() -> dict:
    names = ("camera_a", "camera_b")
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
        "fusion": {"enabled": True, "voxel_size_m": 0.015},
        "sampling": {
            "enabled": True,
            "mode": "voxel_fps",
            "num_points": 1024,
            "voxel_size": 0.01,
            "deterministic": True,
            "seed": 11,
        },
    }


def _depth_frame_sets(count: int = 2):
    scene = create_synthetic_scene(("camera_a", "camera_b"), frame_count=count)
    rig = build_synthetic_rig(parse_rig_config(_rig_config()), scene)
    return [rig.build(index).depth_frame_set for index in range(count)]


def test_same_pass_depth_contract_preserves_native_rays_and_pose() -> None:
    frame_set = _depth_frame_sets(1)[0]
    assert [item.camera_name for item in frame_set.observations] == [
        "camera_a",
        "camera_b",
    ]
    for item in frame_set.observations:
        assert item.depth.dtype == np.uint16
        assert item.depth_unit == "raw_units"
        assert item.depth_scale_m_per_unit == 0.001
        assert item.source_frame.endswith("depth_optical")
        assert item.workspace_frame == "workspace"
        assert item.T_workspace_from_camera.shape == (4, 4)
        assert len(item.provision_sha256) == 64


def test_recording_is_atomic_checksums_exact_and_replayable(tmp_path: Path) -> None:
    output = tmp_path / "recording"
    frames = _depth_frame_sets(2)
    writer = RigDepthRecordingWriter(output, depth_source="native")
    for frame in frames:
        writer.append(frame)
    writer.finalize(report={"synthetic": True})
    assert output.is_dir()
    manifest = validate_rig_depth_recording(output)
    assert manifest["matched_set_count"] == 2
    replayed = list(iter_rig_depth_recording(output))
    for expected, actual in zip(frames, replayed, strict=True):
        assert expected.matched_set_index == actual.matched_set_index
        for left, right in zip(expected.observations, actual.observations, strict=True):
            assert np.array_equal(left.depth, right.depth)
            assert np.array_equal(
                left.T_workspace_from_camera, right.T_workspace_from_camera
            )
    assert not any(
        str(output) in path.read_text(errors="ignore")
        for path in output.rglob("*.json")
    )


def test_recording_rejects_tamper_and_cross_file_forgery(tmp_path: Path) -> None:
    output = tmp_path / "recording"
    writer = RigDepthRecordingWriter(output, depth_source="native")
    writer.append(_depth_frame_sets(1)[0])
    writer.finalize()
    meta = output / "frames/set_000000/camera_a_meta.json"
    value = json.loads(meta.read_text(encoding="utf-8"))
    value["depth"] = "frames/set_000000/camera_b_depth.npy"
    meta.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        validate_rig_depth_recording(output)


def test_ffs_recording_requires_backend_provenance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="backend provenance"):
        RigDepthRecordingWriter(tmp_path / "ffs", depth_source="ffs_stereo")


def test_tsdf_config_is_strict_and_reports_default_memory() -> None:
    raw = {
        "schema_version": "pointcloud-builder.tsdf.v1",
        "backend": {"type": "open3d_tensor", "device": "CPU:0"},
        "volume": {
            "voxel_size_m": 0.005,
            "block_resolution": 16,
            "block_count": 50_000,
            "trunc_voxel_multiplier": 4.0,
        },
        "depth": {"minimum_m": 0.1, "maximum_m": 2.0},
        "integration": {
            "source": "native",
            "frame_stride": 1,
            "maximum_weight": None,
            "queue_capacity": 2,
            "maximum_update_hz": 5.0,
            "maximum_mesh_hz": 1.0,
        },
        "extraction": {
            "weight_threshold": 3.0,
            "point_cloud": True,
            "triangle_mesh": True,
        },
        "dynamic": {
            "mode": "frozen_static",
            "residual_threshold_m": 0.02,
            "persistence_frames": 10,
            "consistency_tolerance_m": 0.01,
            "integrate_background_consistent": True,
            "integrate_persistent_new_surface": True,
        },
    }
    config = parse_tsdf_config(raw)
    assert config.estimated_attribute_bytes == 2_457_600_000
    forged = json.loads(json.dumps(raw))
    forged["integration"]["source"] = "sampled_cloud"
    with pytest.raises(ValueError, match="source"):
        parse_tsdf_config(forged)
    forged = json.loads(json.dumps(raw))
    forged["volume"]["block_count"] = True
    with pytest.raises(ValueError, match="block_count"):
        parse_tsdf_config(forged)
    forged = json.loads(json.dumps(raw))
    forged["dynamic"]["integrate_background_consistent"] = "yes"
    with pytest.raises(ValueError, match="switches"):
        parse_tsdf_config(forged)
    forged = json.loads(json.dumps(raw))
    forged["integration"]["maximum_update_hz"] = float("inf")
    with pytest.raises(ValueError, match="maximum_update_hz"):
        parse_tsdf_config(forged)
    forged = json.loads(json.dumps(raw))
    forged["dynamic"]["residual_threshold_m"] = float("nan")
    with pytest.raises(ValueError, match="residual_threshold_m"):
        parse_tsdf_config(forged)
    raw["volume"]["unexpected"] = True
    with pytest.raises(ValueError, match="unknown"):
        parse_tsdf_config(raw)


def test_mapper_rss_plateau_gate_rejects_sustained_growth() -> None:
    flat = [(index, 900.0) for index in range(100)]
    assert evaluate_rss_plateau(flat)["passed"]
    growing = [(index, 900.0 + 0.06 * index) for index in range(100)]
    report = evaluate_rss_plateau(growing)
    assert report["evaluated"]
    assert not report["passed"]
    assert not report["gates"]["slope_le_5mb_per_100_frames"]
    assert not evaluate_rss_plateau(flat[:20])["evaluated"]
