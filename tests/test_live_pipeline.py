from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from camera_rig.api import CameraFrame, StreamFrame, load_camera_bundle
from pointcloud_builder.config import CropConfig, SamplingConfig
from pointcloud_builder.integrations.camera_rig import create_native_builder
from pointcloud_builder.live import CameraRigLiveSource, LiveSingleCameraWorkspacePipeline
from pointcloud_builder.workspace import SingleCameraWorkspacePipeline

ROOT = Path(__file__).parents[1]
BUNDLE_FIXTURE = ROOT / "third_party/CameraRig/tests/fixtures/consumer/fixed_camera_bundle_v1.json"


@dataclass
class FakeConfig:
    name: str = "synthetic"


class FakeSession:
    def __init__(self, frames: list[CameraFrame]) -> None:
        self.frames = frames
        self.open_calls = 0
        self.close_calls = 0
        self.index = 0

    def open(self) -> None:
        self.open_calls += 1

    def capture(self) -> CameraFrame:
        frame = self.frames[self.index]
        self.index += 1
        return frame

    def close(self) -> None:
        self.close_calls += 1


def _frame(number: int) -> CameraFrame:
    return CameraFrame(
        camera_name="synthetic_camera",
        serial="SYNTHETIC-CONSUMER-0001",
        streams={
            "depth": StreamFrame(
                "depth",
                np.full((3, 4), 1000, dtype=np.uint16),
                number,
                1_000_000_000 + number,
                "synthetic",
            )
        },
        host_receive_timestamp_ns=2_000_000_000 + number,
    )


def _workspace_pipeline() -> SingleCameraWorkspacePipeline:
    context = create_native_builder(
        load_camera_bundle(BUNDLE_FIXTURE),
        device="cpu",
        sampling=SamplingConfig(mode="stride", num_points=12),
    )
    return SingleCameraWorkspacePipeline(
        context,
        workspace_crop=CropConfig(
            enabled=False,
            x=(-10.0, 10.0),
            y=(-10.0, 10.0),
            z=(-10.0, 10.0),
            frame="workspace",
        ),
    )


def test_live_source_opens_once_captures_many_closes_once_and_can_reopen() -> None:
    sessions: list[FakeSession] = []

    def factory(config):
        session = FakeSession([_frame(1), _frame(2)])
        sessions.append(session)
        return session

    source = CameraRigLiveSource(FakeConfig(), session_factory=factory)  # type: ignore[arg-type]
    with source:
        assert source.capture().depth.frame_number == 1
        assert source.capture().depth.frame_number == 2
        with pytest.raises(RuntimeError, match="already open"):
            source.open()
    assert source.open_count == source.close_count == 1
    assert sessions[0].open_calls == sessions[0].close_calls == 1
    with source:
        assert source.capture().depth.frame_number == 1
    assert source.open_count == source.close_count == 2
    assert sessions[1].open_calls == sessions[1].close_calls == 1


def test_live_pipeline_is_synchronous_and_reports_required_timings() -> None:
    session = FakeSession([_frame(1)])
    source = CameraRigLiveSource(
        FakeConfig(),  # type: ignore[arg-type]
        session_factory=lambda config: session,
    )
    live = LiveSingleCameraWorkspacePipeline(source, _workspace_pipeline())
    with live:
        result = live.capture_next()
    assert result.stages.workspace_raw.frame == "workspace"
    assert result.stages.workspace_sampled.points.shape == (12, 3)
    required = {
        "capture",
        "frame_adapter",
        "depth_inference",
        "deprojection",
        "local_crop",
        "workspace_transform",
        "workspace_crop",
        "sampling",
        "total",
    }
    assert required <= result.timing_ms.keys()
    assert all(result.timing_ms[name] >= 0.0 for name in required)
