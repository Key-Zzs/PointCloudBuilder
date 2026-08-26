"""Bounded single-camera offline sources for rig orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from camera_rig.api import ReplayCameraSession

from pointcloud_builder.rig.types import CameraFrameEnvelope


class OfflineCameraSource(Protocol):
    camera_name: str

    @property
    def frame_count(self) -> int: ...

    def envelope(self, index: int) -> CameraFrameEnvelope: ...


class CameraRigReplaySource:
    """Eagerly validate and load one bounded CameraRig capture artifact."""

    def __init__(self, camera_name: str, artifact: str | Path) -> None:
        self.camera_name = camera_name
        frames: list[Any] = []
        with ReplayCameraSession.from_artifact(artifact) as session:
            for _ in range(session.frame_count):
                frames.append(session.capture())
        self._frames = tuple(frames)
        if not self._frames:
            raise ValueError(f"camera {camera_name!r} replay is empty")
        if any(frame.camera_name != camera_name for frame in self._frames):
            raise ValueError(f"camera {camera_name!r} replay frame identity mismatch")

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def envelope(self, index: int) -> CameraFrameEnvelope:
        return _envelope(self.camera_name, self._frames, index)


class SyntheticCameraSource:
    """Deterministic in-memory source with the same one-camera envelope contract."""

    def __init__(self, camera_name: str, frames: Sequence[Any]) -> None:
        self.camera_name = camera_name
        self._frames = tuple(frames)
        if not self._frames:
            raise ValueError(f"camera {camera_name!r} synthetic source is empty")
        if any(frame.camera_name != camera_name for frame in self._frames):
            raise ValueError(f"camera {camera_name!r} synthetic frame identity mismatch")

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def envelope(self, index: int) -> CameraFrameEnvelope:
        return _envelope(self.camera_name, self._frames, index)


def _envelope(camera_name: str, frames: tuple[Any, ...], index: int) -> CameraFrameEnvelope:
    if index < 0 or index >= len(frames):
        raise IndexError(f"camera {camera_name!r} has no frame at index {index}")
    frame = frames[index]
    return CameraFrameEnvelope(
        camera_name=camera_name,
        frame_index=index,
        host_receive_timestamp_ns=int(frame.host_receive_timestamp_ns),
        frame=frame,
    )
