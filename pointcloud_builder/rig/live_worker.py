"""One-thread-per-camera capture workers for live rig acquisition."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import queue
import threading
import traceback
from typing import Any

from pointcloud_builder.live.source import CameraRigLiveSource
from pointcloud_builder.rig.live_telemetry import LiveRigTelemetry
from pointcloud_builder.rig.types import CameraFrameEnvelope


@dataclass(frozen=True)
class LiveWorkerError:
    camera_name: str
    phase: str
    error_type: str
    message: str
    traceback_text: str

    def summary(self) -> dict[str, str]:
        return {
            "camera_name": self.camera_name,
            "phase": self.phase,
            "error_type": self.error_type,
            "message": self.message,
        }


class LiveCameraWorker:
    """Own a CameraRig session entirely inside one non-daemon thread."""

    def __init__(
        self,
        camera_name: str,
        camera_config: Any,
        *,
        buffer: Any,
        stop_event: threading.Event,
        error_queue: queue.Queue[LiveWorkerError],
        telemetry: LiveRigTelemetry,
        required_streams: tuple[str, ...] = ("color", "depth", "ir_left", "ir_right"),
        session_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        if getattr(camera_config.camera, "name", None) != camera_name:
            raise ValueError("worker camera name must match the CameraRig config")
        self.camera_name = camera_name
        self.camera_config = camera_config
        self.buffer = buffer
        self.stop_event = stop_event
        self.error_queue = error_queue
        self.telemetry = telemetry
        self.required_streams = required_streams
        self.session_factory = session_factory
        self.ready_event = threading.Event()
        self.finished_event = threading.Event()
        self.thread = threading.Thread(
            name=f"live-camera-{camera_name}", target=self._run, daemon=False
        )

    @property
    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float | None = None) -> None:
        self.thread.join(timeout=timeout)

    def _publish_error(self, phase: str, exc: BaseException) -> None:
        timeout = isinstance(exc, TimeoutError) or "timeout" in str(exc).casefold()
        self.telemetry.record_error(self.camera_name, phase, timeout=timeout)
        self.error_queue.put(
            LiveWorkerError(
                camera_name=self.camera_name,
                phase=phase,
                error_type=type(exc).__name__,
                message=str(exc),
                traceback_text=traceback.format_exc(),
            )
        )
        self.stop_event.set()
        self.buffer.notify_all()

    def _run(self) -> None:
        source: CameraRigLiveSource | None = None
        phase = "create"
        capture_index = 0
        try:
            source = CameraRigLiveSource(
                self.camera_config, session_factory=self.session_factory
            )
            phase = "open"
            source.open()
            self.telemetry.record_lifecycle(self.camera_name, "open")
            self.ready_event.set()
            phase = "capture"
            while not self.stop_event.is_set():
                frame = source.capture()
                if getattr(frame, "camera_name", None) != self.camera_name:
                    raise ValueError("captured frame camera name does not match worker")
                missing = [name for name in self.required_streams if name not in frame.streams]
                monotonic = self.telemetry.record_frame(
                    self.camera_name, frame, self.required_streams
                )
                if missing:
                    raise ValueError(f"captured frame missing required streams: {missing}")
                if not monotonic:
                    raise ValueError("host_receive_timestamp_ns is not strictly monotonic")
                self.buffer.push(
                    CameraFrameEnvelope(
                        camera_name=self.camera_name,
                        frame_index=capture_index,
                        host_receive_timestamp_ns=int(frame.host_receive_timestamp_ns),
                        frame=frame,
                    )
                )
                capture_index += 1
        except BaseException as exc:  # thread boundary must preserve every failure
            self._publish_error(phase, exc)
        finally:
            self.ready_event.set()
            if source is not None:
                try:
                    source.close()
                    self.telemetry.record_lifecycle(self.camera_name, "close")
                except BaseException as exc:
                    self._publish_error("close", exc)
            self.finished_event.set()
            self.buffer.notify_all()
