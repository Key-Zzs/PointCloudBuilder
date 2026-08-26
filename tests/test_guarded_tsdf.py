from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pointcloud_builder.mapping.config import TsdfDynamicConfig
from pointcloud_builder.mapping.config import parse_tsdf_config
from pointcloud_builder.mapping.guarded import GuardedDepthFilter
from pointcloud_builder.mapping.types import RigDepthFrameSet, RigDepthObservation
from pointcloud_builder.rig import (
    build_synthetic_rig,
    create_synthetic_scene,
    parse_rig_config,
)


def _observation() -> RigDepthObservation:
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
        "workspace_crop": {"enabled": False},
        "fusion": {"enabled": True, "voxel_size_m": 0.015},
        "sampling": {
            "enabled": True,
            "mode": "voxel_fps",
            "num_points": 256,
            "voxel_size": 0.01,
            "deterministic": True,
            "seed": 3,
        },
    }
    rig = build_synthetic_rig(
        parse_rig_config(raw),
        create_synthetic_scene(names, frame_count=1, width=32, height=24),
    )
    return rig.build(0).depth_frame_set.observations[0]


def _metric_observation(
    base: RigDepthObservation, depth_m: np.ndarray
) -> RigDepthObservation:
    depth = np.asarray(depth_m, dtype=np.float32)
    return replace(
        base,
        depth=depth,
        depth_unit="meters",
        depth_scale_m_per_unit=1.0,
        valid_mask=depth > 0,
        depth_source="ffs_stereo",
        rectified=True,
    )


def test_transient_and_moving_surface_is_masked_but_persistent_surface_enters() -> None:
    base = _observation()
    shape = base.depth.shape
    predicted = np.full(shape, 1.0, dtype=np.float32)
    observed = predicted.copy()
    roi = np.s_[8:14, 10:18]
    config = TsdfDynamicConfig(
        mode="guarded_continuous",
        residual_threshold_m=0.02,
        persistence_frames=4,
        consistency_tolerance_m=0.01,
        integrate_background_consistent=True,
        integrate_persistent_new_surface=True,
    )
    guarded = GuardedDepthFilter(config)

    # A transient cube, then a moving cube, never reaches persistence.
    transient_integrated = 0
    for depth in (0.7, 0.65, 0.8):
        frame = observed.copy()
        frame[roi] = depth
        decision = guarded.apply(_metric_observation(base, frame), predicted)
        transient_integrated += int((decision.observation.depth[roi] > 0).sum())
        assert decision.report.transient_dynamic_pixels >= 48
    assert transient_integrated == 0

    # Disappearance restores the predicted static plane immediately.
    restored = guarded.apply(_metric_observation(base, observed), predicted)
    assert np.all(restored.observation.depth[roi] == 1.0)

    # One fixed new surface enters only on the preregistered persistence frame.
    integrated_counts = []
    for _ in range(4):
        frame = observed.copy()
        frame[roi] = 0.72
        decision = guarded.apply(_metric_observation(base, frame), predicted)
        integrated_counts.append(int((decision.observation.depth[roi] > 0).sum()))
    assert integrated_counts == [0, 0, 0, 48]
    assert decision.report.newly_persistent_pixels == 48
    assert decision.report.persistent_candidate_pixels == 48


def test_guarded_filter_reset_clears_pixel_persistence() -> None:
    base = _observation()
    predicted = np.ones(base.depth.shape, dtype=np.float32)
    observed = np.full(base.depth.shape, 0.7, dtype=np.float32)
    guarded = GuardedDepthFilter(
        TsdfDynamicConfig(mode="guarded_continuous", persistence_frames=2)
    )
    guarded.apply(_metric_observation(base, observed), predicted)
    guarded.reset()
    decision = guarded.apply(_metric_observation(base, observed), predicted)
    assert decision.report.persistent_candidate_pixels == 0


def test_slowly_drifting_surface_cannot_accumulate_pairwise_persistence() -> None:
    base = _observation()
    predicted = np.ones(base.depth.shape, dtype=np.float32)
    guarded = GuardedDepthFilter(
        TsdfDynamicConfig(
            mode="guarded_continuous",
            residual_threshold_m=0.02,
            persistence_frames=4,
            consistency_tolerance_m=0.01,
        )
    )
    roi = np.s_[8:14, 10:18]
    integrated = []
    for depth_m in (0.700, 0.706, 0.712, 0.718, 0.724):
        observed = predicted.copy()
        observed[roi] = depth_m
        decision = guarded.apply(_metric_observation(base, observed), predicted)
        integrated.append(int((decision.observation.depth[roi] > 0).sum()))
    assert integrated == [0, 0, 0, 0, 0]
    assert decision.report.persistent_candidate_pixels == 0
    assert decision.report.metrics["transient_dynamic_ratio"] > 0


def test_guarded_sequence_has_zero_tsdf_ghosts_and_integrates_persistent_surface() -> (
    None
):
    pytest.importorskip("open3d")
    from pointcloud_builder.mapping.open3d import Open3dTsdfMap

    base = _observation()
    base = _metric_observation(base, base.metric_depth)
    config = parse_tsdf_config(
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
                "source": "ffs_stereo",
                "frame_stride": 1,
                "maximum_weight": None,
                "queue_capacity": 2,
                "maximum_update_hz": 20.0,
                "maximum_mesh_hz": 1.0,
            },
            "extraction": {
                "weight_threshold": 1.0,
                "point_cloud": True,
                "triangle_mesh": True,
            },
            "dynamic": {
                "mode": "guarded_continuous",
                "residual_threshold_m": 0.02,
                "persistence_frames": 4,
                "consistency_tolerance_m": 0.01,
                "integrate_background_consistent": True,
                "integrate_persistent_new_surface": True,
            },
        }
    )

    def frame(observation: RigDepthObservation, index: int) -> RigDepthFrameSet:
        return RigDepthFrameSet(
            matched_set_index=index,
            host_timestamp_ns=index,
            maximum_skew_ms=0.0,
            observations=(observation,),
        )

    mapper = Open3dTsdfMap(config, workspace_frame=base.workspace_frame)
    mapper.integrate(frame(base, 0))
    mapper.integrate(frame(base, 1))
    baseline_points = mapper.extract().points
    # Open3D 0.19 lazily allocates the range map on the first raycast.
    mapper.raycast_depth(base)
    predicted = mapper.raycast_depth(base)
    valid = predicted > 0.4
    roi = None
    for row in range(valid.shape[0] - 5):
        for column in range(valid.shape[1] - 7):
            candidate = valid[row : row + 6, column : column + 8]
            if candidate.all():
                yy, xx = np.mgrid[row : row + 6, column : column + 8]
                roi = (yy * valid.shape[1] + xx).reshape(-1)
                break
        if roi is not None:
            break
    assert roi is not None and len(roi) == 48

    def workspace_targets(depth_flat: np.ndarray) -> np.ndarray:
        rows, columns = np.divmod(roi, predicted.shape[1])
        z = depth_flat[roi]
        intrinsics = base.intrinsics
        camera = np.column_stack(
            (
                (columns - intrinsics.cx) * z / intrinsics.fx,
                (rows - intrinsics.cy) * z / intrinsics.fy,
                z,
                np.ones_like(z),
            )
        )
        return (base.T_workspace_from_camera @ camera.T).T[:, :3]

    def nearest(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
        return np.sqrt(
            np.square(query[:, None, :] - reference[None, :, :]).sum(axis=2)
        ).min(axis=1)

    guarded = GuardedDepthFilter(config.dynamic)
    false_integrated_pixels = 0
    transient_pixels = 0
    ghost_targets = []
    for index, offset in enumerate((0.15, 0.21, 0.27), start=2):
        observed = predicted.copy().reshape(-1)
        observed[roi] -= offset
        ghost_targets.append(workspace_targets(observed))
        decision = guarded.apply(
            _metric_observation(base, observed.reshape(predicted.shape)), predicted
        )
        false_integrated_pixels += int(
            (decision.observation.depth.reshape(-1)[roi] > 0).sum()
        )
        transient_pixels += len(roi)
        mapper.integrate(frame(decision.observation, index))
    after_transient_points = mapper.extract().points
    ghost_distances = nearest(
        after_transient_points, np.concatenate(ghost_targets, axis=0)
    )
    ghost_voxel_ratio = float((ghost_distances <= 0.03).mean())
    false_integration_ratio = false_integrated_pixels / transient_pixels
    assert ghost_voxel_ratio == 0.0
    assert false_integration_ratio == 0.0

    restored = guarded.apply(_metric_observation(base, predicted), predicted)
    assert np.all(restored.observation.depth.reshape(-1)[roi] > 0)
    mapper.integrate(frame(restored.observation, 5))
    restored_points = mapper.extract().points
    baseline_sample = baseline_points[
        np.linspace(0, len(baseline_points) - 1, min(500, len(baseline_points)))
        .round()
        .astype(int)
    ]
    restored_sample = restored_points[
        np.linspace(0, len(restored_points) - 1, min(500, len(restored_points)))
        .round()
        .astype(int)
    ]
    recovery_distances = np.concatenate(
        (
            nearest(baseline_sample, restored_points),
            nearest(restored_sample, baseline_points),
        )
    )
    assert float(np.quantile(recovery_distances, 0.95)) <= 0.01

    persistent_depth = predicted.copy().reshape(-1)
    persistent_depth[roi] -= 0.20
    for index in range(6, 18):
        decision = guarded.apply(
            _metric_observation(base, persistent_depth.reshape(predicted.shape)),
            predicted,
        )
        mapper.integrate(frame(decision.observation, index))
    persistent_points = mapper.extract().points
    persistent_completeness = float(
        (nearest(workspace_targets(persistent_depth), persistent_points) <= 0.03).mean()
    )
    assert persistent_completeness == 1.0
    mapper.close()
