"""Synchronous live CameraRig-to-workspace processing."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from pointcloud_builder.live.source import CameraRigLiveSource
from pointcloud_builder.workspace.pipeline import (
    SingleCameraWorkspacePipeline,
    SingleCameraWorkspaceStages,
)


@dataclass(frozen=True)
class LiveWorkspaceFrame:
    stages: SingleCameraWorkspaceStages
    timing_ms: dict[str, float]


class LiveSingleCameraWorkspacePipeline:
    """Capture and process one live frame at a time without buffering."""

    def __init__(
        self,
        source: CameraRigLiveSource,
        workspace_pipeline: SingleCameraWorkspacePipeline,
    ) -> None:
        self.source = source
        self.workspace_pipeline = workspace_pipeline

    def capture_next(self) -> LiveWorkspaceFrame:
        total_start = time.perf_counter()
        capture_start = total_start
        frame = self.source.capture()
        capture_ms = (time.perf_counter() - capture_start) * 1000.0
        stages = self.workspace_pipeline.process(frame)
        timing = dict(stages.metadata["timing_ms"])
        timing["capture"] = capture_ms
        timing["total"] = (time.perf_counter() - total_start) * 1000.0
        return LiveWorkspaceFrame(stages=stages, timing_ms=timing)

    def __enter__(self) -> "LiveSingleCameraWorkspacePipeline":
        self.source.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.source.close()
