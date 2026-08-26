from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from camera_rig.api import CameraFrame, StreamFrame

from pointcloud_builder.rig.config import RigLiveConfig, RigTimingConfig, parse_rig_config
from pointcloud_builder.rig.live import (
    LiveRigAcquisition,
    LiveRigPipeline,
    LiveRigWorkerFailure,
)
from pointcloud_builder.rig.synthetic import build_synthetic_rig, create_synthetic_scene


class _Session:
    def __init__(
        self,
        name: str,
        *,
        offset_ns: int = 0,
        frames: tuple[CameraFrame, ...] | None = None,
        fail_after: int | None = None,
    ) -> None:
        self.name = name
        self.offset_ns = offset_ns
        self.frames = frames
        self.fail_after = fail_after
        self.index = 0
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def capture(self) -> CameraFrame:
        if self.fail_after is not None and self.index >= self.fail_after:
            raise RuntimeError("synthetic capture failure")
        time.sleep(0.001)
        if self.frames is not None:
            original = self.frames[self.index % len(self.frames)]
            frame = CameraFrame(
                camera_name=original.camera_name,
                serial=original.serial,
                streams=original.streams,
                host_receive_timestamp_ns=(
                    1_000_000_000 + self.index * 33_333_333 + self.offset_ns
                ),
            )
        else:
            streams = {
                name: StreamFrame(
                    stream_name=name,
                    data=(
                        np.zeros((1, 1, 3), dtype=np.uint8)
                        if name == "color"
                        else np.zeros((1, 1), dtype=np.uint16 if name == "depth" else np.uint8)
                    ),
                    frame_number=self.index,
                )
                for name in ("color", "depth", "ir_left", "ir_right")
            }
            frame = CameraFrame(
                camera_name=self.name,
                serial=f"FAKE-{self.name}",
                streams=streams,
                host_receive_timestamp_ns=(
                    1_000_000_000 + self.index * 33_333_333 + self.offset_ns
                ),
            )
        self.index += 1
        return frame

    def close(self) -> None:
        self.closed = True


def _camera_config(name: str):
    return SimpleNamespace(camera=SimpleNamespace(name=name))


def _acquisition(*, fail_b_after: int | None = None):
    sessions: dict[str, list[_Session]] = {"camera_a": [], "camera_b": []}

    def factory(name: str, offset: int, fail_after: int | None = None):
        def create(_):
            session = _Session(name, offset_ns=offset, fail_after=fail_after)
            sessions[name].append(session)
            return session

        return create

    acquisition = LiveRigAcquisition(
        {name: _camera_config(name) for name in sessions},
        timing=RigTimingConfig(
            mode="nearest_host_timestamp",
            maximum_skew_ms=20.0,
            reference_camera="camera_a",
        ),
        live_config=RigLiveConfig(
            buffer_capacity=2,
            matcher_wait_timeout_s=0.05,
            join_timeout_s=2.0,
            telemetry_history_capacity=32,
        ),
        session_factories={
            "camera_a": factory("camera_a", 0),
            "camera_b": factory("camera_b", 4_000_000, fail_b_after),
        },
    )
    return acquisition, sessions


def test_acquisition_matches_cleans_up_and_reports_bounded_telemetry() -> None:
    acquisition, sessions = _acquisition()
    matches = []
    with acquisition:
        while len(matches) < 20:
            matched = acquisition.next_frame_set()
            if matched is not None:
                matches.append(matched)
    assert all(
        len({item.envelopes[name].frame_index for item in matches}) == len(matches)
        for name in ("camera_a", "camera_b")
    )
    report = acquisition.report()
    assert report["workers_alive"] == []
    assert report["worker_errors"] == []
    assert report["matcher"]["matched_sets"] == 20
    assert report["matcher"]["frame_reuse_violations"] == 0
    assert report["matcher"]["absolute_skew_ms"]["camera_b"]["p95"] <= 20.0
    assert report["buffers"]["camera_a"]["maximum_depth"] <= 2
    assert report["buffers"]["camera_b"]["maximum_depth"] <= 2
    assert all(values[0].closed for values in sessions.values())


def test_worker_failure_propagates_and_all_workers_close() -> None:
    acquisition, sessions = _acquisition(fail_b_after=2)
    acquisition.start()
    with pytest.raises(LiveRigWorkerFailure, match="camera_b"):
        for _ in range(20):
            acquisition.next_frame_set()
    acquisition.stop()
    report = acquisition.report()
    assert report["workers_alive"] == []
    assert report["worker_errors"][0]["camera_name"] == "camera_b"
    assert all(values[0].closed for values in sessions.values())


def _synthetic_raw() -> dict:
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
            "mode": "nearest_host_timestamp",
            "maximum_skew_ms": 20.0,
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
            "seed": 17,
        },
    }


def test_live_frame_set_uses_the_same_processor_as_offline() -> None:
    config = parse_rig_config(_synthetic_raw())
    scene = create_synthetic_scene(("camera_a", "camera_b"), frame_count=3)
    offline = build_synthetic_rig(config, scene)
    sessions: dict[str, _Session] = {}

    def factory(name: str):
        def create(_):
            session = _Session(
                name,
                offset_ns=4_000_000 if name == "camera_b" else 0,
                frames=scene.frames[name],
            )
            sessions[name] = session
            return session

        return create

    acquisition = LiveRigAcquisition(
        {name: _camera_config(name) for name in scene.frames},
        timing=config.timing,
        live_config=RigLiveConfig(matcher_wait_timeout_s=0.1, join_timeout_s=2.0),
        session_factories={name: factory(name) for name in scene.frames},
        required_streams_by_camera={name: ("depth",) for name in scene.frames},
    )
    live = LiveRigPipeline(acquisition, offline.processor)
    with live:
        live_result = live.capture_next().result
    offline_result = offline.build(0)
    assert torch.equal(live_result.sampled.points, offline_result.sampled.points)
    assert live_result.processing_metadata == offline_result.processing_metadata
    assert live_result.frame_match.matching_policy == "nearest_host_timestamp"
    assert list(live_result.timing_report_ms["per_camera"]) == ["camera_a", "camera_b"]
    assert all(session.closed for session in sessions.values())
