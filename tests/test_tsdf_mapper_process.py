from __future__ import annotations

from dataclasses import replace
import threading
import time

import pytest

from pointcloud_builder.mapping.process import AsyncTsdfMapper, MapperProcessConfig
from pointcloud_builder.rig import (
    build_synthetic_rig,
    create_synthetic_scene,
    parse_rig_config,
)

from test_tsdf_mapping import _config

pytest.importorskip("open3d")


def _frame_set():
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
            "seed": 7,
        },
    }
    rig = build_synthetic_rig(
        parse_rig_config(raw), create_synthetic_scene(names, frame_count=1)
    )
    return rig.build(0).depth_frame_set


def test_async_mapper_integrates_extracts_and_obeys_lifecycle() -> None:
    config = replace(
        _config(),
        integration=replace(
            _config().integration,
            maximum_update_hz=20.0,
            maximum_mesh_hz=20.0,
        ),
    )
    mapper = AsyncTsdfMapper(MapperProcessConfig(config, "workspace"))
    mapper.start()
    assert mapper.submit(_frame_set())
    snapshot = None
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and snapshot is None:
        snapshot = mapper.poll_snapshot()
        time.sleep(0.02)
    assert snapshot is not None
    assert snapshot.extraction.point_count > 0
    assert snapshot.map_state.active_block_count > 0
    assert mapper.freeze().lifecycle == "frozen"
    assert mapper.unfreeze().lifecycle == "integrating"
    reset = mapper.reset()
    assert reset.lifecycle == "created"
    assert reset.active_block_count == 0
    telemetry = mapper.close()
    assert telemetry.child_error is None
    assert not telemetry.running


def test_async_mapper_latest_only_queue_is_bounded_and_nonblocking() -> None:
    config = replace(
        _config(),
        integration=replace(
            _config().integration,
            queue_capacity=1,
            maximum_update_hz=5.0,
            maximum_mesh_hz=1.0,
        ),
    )
    mapper = AsyncTsdfMapper(MapperProcessConfig(config, "workspace"))
    mapper.start()
    base = _frame_set()
    durations = []
    for index in range(30):
        frame = replace(base, matched_set_index=index)
        started = time.perf_counter()
        mapper.submit(frame)
        durations.append((time.perf_counter() - started) * 1000.0)
    telemetry = mapper.close()
    assert telemetry.submitted_frame_sets == 30
    assert telemetry.producer_dropped_frame_sets > 0
    assert telemetry.maximum_queue_depth <= 1
    assert sorted(durations)[28] <= 5.0
    assert telemetry.child_error is None


def test_reset_ack_is_a_barrier_against_pre_reset_queued_frames() -> None:
    config = replace(
        _config(),
        integration=replace(
            _config().integration,
            queue_capacity=2,
            maximum_update_hz=20.0,
            maximum_mesh_hz=1.0,
        ),
    )
    mapper = AsyncTsdfMapper(MapperProcessConfig(config, "workspace"))
    mapper.start()
    base = _frame_set()
    for index in range(40):
        mapper.submit(replace(base, matched_set_index=index))
    reset = mapper.reset()
    assert reset.active_block_count == 0
    time.sleep(0.3)
    frozen = mapper.freeze()
    assert frozen.active_block_count == 0
    assert frozen.integrated_frame_sets == 0
    telemetry = mapper.close()
    assert telemetry.child_control_discarded_frame_sets >= 0
    assert telemetry.child_error is None


def test_lifecycle_barrier_is_bounded_during_concurrent_submission() -> None:
    config = replace(
        _config(),
        integration=replace(
            _config().integration,
            queue_capacity=2,
            maximum_update_hz=20.0,
            maximum_mesh_hz=1.0,
        ),
    )
    mapper = AsyncTsdfMapper(MapperProcessConfig(config, "workspace"))
    mapper.start()
    base = _frame_set()
    stopped = threading.Event()

    def produce() -> None:
        index = 0
        while not stopped.is_set():
            mapper.submit(replace(base, matched_set_index=index))
            index += 1
            time.sleep(0.001)

    producer = threading.Thread(target=produce)
    producer.start()
    time.sleep(0.05)
    started = time.monotonic()
    mapper.reset(timeout_s=3.0)
    assert time.monotonic() - started < 3.0
    stopped.set()
    producer.join(timeout=2.0)
    assert not producer.is_alive()
    reset = mapper.reset(timeout_s=3.0)
    assert reset.active_block_count == 0
    telemetry = mapper.close()
    assert telemetry.producer_dropped_frame_sets > 0
    assert telemetry.child_error is None


def test_mapper_start_failure_reaps_non_daemon_child() -> None:
    config = replace(_config(), backend=replace(_config().backend, device="INVALID:0"))
    mapper = AsyncTsdfMapper(MapperProcessConfig(config, "workspace"))
    with pytest.raises(RuntimeError):
        mapper.start(timeout_s=3.0)
    assert not mapper.running
    assert mapper.telemetry().child_error is not None
