"""Context-managed stable-API CameraRig live source."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pointcloud_builder.integrations.camera_rig.dependencies import (
    CameraConfig,
    CameraSession,
    load_camera_config,
)


class CameraRigLiveSource:
    """Open one CameraRig session, capture synchronously, and close exactly once."""

    def __init__(
        self,
        config: CameraConfig,
        *,
        session_factory: Callable[[CameraConfig], Any] | None = None,
    ) -> None:
        self.config = config
        self._session_factory = session_factory or CameraSession.from_config
        self._session: Any | None = None
        self.open_count = 0
        self.close_count = 0
        self.capture_count = 0

    @classmethod
    def from_config_path(cls, path: str | Path) -> "CameraRigLiveSource":
        return cls(load_camera_config(path))

    @property
    def is_open(self) -> bool:
        return self._session is not None

    def open(self) -> None:
        if self._session is not None:
            raise RuntimeError("CameraRigLiveSource is already open")
        session = self._session_factory(self.config)
        try:
            session.open()
        except Exception:
            try:
                session.close()
            finally:
                raise
        self._session = session
        self.open_count += 1

    def capture(self) -> Any:
        if self._session is None:
            raise RuntimeError("CameraRigLiveSource is not open")
        frame = self._session.capture()
        self.capture_count += 1
        return frame

    def close(self) -> None:
        session = self._session
        if session is None:
            return
        self._session = None
        session.close()
        self.close_count += 1

    def __enter__(self) -> "CameraRigLiveSource":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
