from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pointcloud_builder.mapping.artifact import (
    _volume_statistics_equal,
    load_tsdf_map_artifact,
    validate_tsdf_map_artifact,
    write_tsdf_map_artifact,
)
from pointcloud_builder.mapping.config import parse_tsdf_config
from pointcloud_builder.mapping.open3d import FeatureNotSupportedError, Open3dTsdfMap
from pointcloud_builder.mapping.types import RigDepthFrameSet
from pointcloud_builder.rig import (
    build_synthetic_rig,
    create_synthetic_scene,
    parse_rig_config,
)

pytest.importorskip("open3d")


def test_volume_statistics_allow_only_float32_reduction_roundoff() -> None:
    base = {
        "shape": [100, 8, 8, 8, 1],
        "dtype": "Float32",
        "minimum": -1.0,
        "maximum": 1.0,
        "mean": 0.345662921667099,
        "nonzero_count": 40000,
    }
    reordered = {**base, "mean": base["mean"] + 2.0e-7}
    changed_values = {**base, "mean": base["mean"] + 2.0e-5}

    assert _volume_statistics_equal(base, reordered)
    assert not _volume_statistics_equal(base, changed_values)
    assert not _volume_statistics_equal(
        base, {**reordered, "nonzero_count": base["nonzero_count"] - 1}
    )


def _rig():
    names = ("camera_a", "camera_b")
    raw = {
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
            "seed": 5,
        },
    }
    scene = create_synthetic_scene(names, frame_count=2)
    return build_synthetic_rig(parse_rig_config(raw), scene)


def _config():
    return parse_tsdf_config(
        {
            "schema_version": "pointcloud-builder.tsdf.v1",
            "backend": {"type": "open3d_tensor", "device": "CPU:0"},
            "volume": {
                "voxel_size_m": 0.01,
                "block_resolution": 8,
                "block_count": 2000,
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
    )


def _scene_surface_error(points: np.ndarray) -> float:
    plane = np.abs(points[:, 2])
    lower = np.asarray((-0.18, -0.12, 0.0))
    upper = np.asarray((0.18, 0.12, 0.25))
    inside_other_axes = []
    face_distances = []
    for axis in range(3):
        others = [item for item in range(3) if item != axis]
        inside = np.ones(len(points), dtype=bool)
        for other in others:
            inside &= (points[:, other] >= lower[other] - 0.02) & (
                points[:, other] <= upper[other] + 0.02
            )
        inside_other_axes.append(inside)
        face_distances.append(
            np.minimum(
                np.abs(points[:, axis] - lower[axis]),
                np.abs(points[:, axis] - upper[axis]),
            )
        )
    box = np.full(len(points), np.inf)
    for inside, distance in zip(inside_other_axes, face_distances, strict=True):
        box = np.minimum(box, np.where(inside, distance, np.inf))
    return float(np.median(np.minimum(plane, box)))


def test_open3d_extrinsic_direction_and_depth_scale_have_synthetic_parity() -> None:
    frame_set = _rig().build(0).depth_frame_set
    correct = Open3dTsdfMap(_config(), workspace_frame="workspace")
    correct.integrate(frame_set)
    correct_error = _scene_surface_error(correct.extract().points)

    wrong_observations = tuple(
        replace(
            item,
            T_workspace_from_camera=np.linalg.inv(item.T_workspace_from_camera),
        )
        for item in frame_set.observations
    )
    wrong_set = replace(frame_set, observations=wrong_observations)
    wrong = Open3dTsdfMap(_config(), workspace_frame="workspace")
    wrong.integrate(wrong_set)
    wrong_error = _scene_surface_error(wrong.extract().points)
    assert correct_error <= 0.01
    assert wrong_error >= correct_error + 0.05
    correct.close()
    wrong.close()


def test_save_load_artifact_geometry_parity_and_lifecycle(tmp_path: Path) -> None:
    rig = _rig()
    mapper = Open3dTsdfMap(_config(), workspace_frame="workspace")
    integrations = [
        mapper.integrate(rig.build(index).depth_frame_set) for index in range(2)
    ]
    with pytest.raises(ValueError, match="frozen"):
        write_tsdf_map_artifact(
            tmp_path / "not-frozen",
            mapper=mapper,
            source_recording_sha256="a" * 64,
            integration_metrics={},
        )
    mapper.freeze()
    output = tmp_path / "map"
    artifact = write_tsdf_map_artifact(
        output,
        mapper=mapper,
        source_recording_sha256="a" * 64,
        integration_metrics={
            "frames": len(integrations),
            "p50_ms": float(np.median([item.integration_ms for item in integrations])),
        },
    )
    assert artifact.root == output
    validate_tsdf_map_artifact(output)
    loaded = load_tsdf_map_artifact(output)
    assert loaded.state.lifecycle == "frozen"
    assert loaded.state.active_block_count == mapper.state.active_block_count
    assert loaded.extract().point_count == mapper.extract().point_count
    loaded.unfreeze()
    loaded.reset()
    assert loaded.state.active_block_count == 0
    assert loaded.state.map_revision >= 2
    with pytest.raises(FeatureNotSupportedError):
        loaded.invalidate_aabb(np.zeros(3), np.ones(3))
    loaded.close()
    mapper.close()


def test_nonidentity_unrectified_distortion_is_rejected() -> None:
    frame_set = _rig().build(0).depth_frame_set
    bad = replace(
        frame_set.observations[0],
        distortion_model="inverse-brown-conrady",
        distortion_coeffs=(0.1, 0.0, 0.0, 0.0, 0.0),
        rectified=False,
    )
    forged = RigDepthFrameSet(
        matched_set_index=frame_set.matched_set_index,
        host_timestamp_ns=frame_set.host_timestamp_ns,
        maximum_skew_ms=frame_set.maximum_skew_ms,
        observations=(bad, frame_set.observations[1]),
    )
    mapper = Open3dTsdfMap(_config(), workspace_frame="workspace")
    with pytest.raises(ValueError, match="distortion"):
        mapper.integrate(forged)
    mapper.close()
