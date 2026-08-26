"""Bounded scalar telemetry for live multi-camera acquisition."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import threading
from typing import Any


def _jump_bucket(delta: int) -> str:
    if -4 <= delta <= 6:
        return str(delta)
    return "lt_-4" if delta < -4 else "gt_6"


@dataclass
class CameraTelemetry:
    captured: int = 0
    first_host_timestamp_ns: int | None = None
    last_host_timestamp_ns: int | None = None
    host_timestamp_monotonic: bool = True
    host_timestamp_violations: int = 0
    required_stream_missing: Counter[str] = field(default_factory=Counter)
    frame_number_jump_histogram: dict[str, Counter[str]] = field(default_factory=dict)
    last_frame_number: dict[str, int] = field(default_factory=dict)
    timeout_count: int = 0
    capture_error_count: int = 0
    open_error_count: int = 0
    close_error_count: int = 0
    session_opened: bool = False
    session_closed: bool = False
    worker_thread_id: int | None = None
    lifecycle_thread_ids: dict[str, int] = field(default_factory=dict)

    def report(self) -> dict[str, Any]:
        elapsed_ns = (
            self.last_host_timestamp_ns - self.first_host_timestamp_ns
            if self.first_host_timestamp_ns is not None
            and self.last_host_timestamp_ns is not None
            else 0
        )
        fps = (
            (self.captured - 1) * 1_000_000_000.0 / elapsed_ns
            if self.captured > 1 and elapsed_ns > 0
            else 0.0
        )
        return {
            "captured": self.captured,
            "capture_fps": fps,
            "first_host_timestamp_ns": self.first_host_timestamp_ns,
            "last_host_timestamp_ns": self.last_host_timestamp_ns,
            "host_timestamp_monotonic": self.host_timestamp_monotonic,
            "host_timestamp_violations": self.host_timestamp_violations,
            "required_stream_missing": dict(self.required_stream_missing),
            "frame_number_jump_histogram": {
                name: dict(sorted(counts.items()))
                for name, counts in sorted(self.frame_number_jump_histogram.items())
            },
            "timeout_count": self.timeout_count,
            "capture_error_count": self.capture_error_count,
            "open_error_count": self.open_error_count,
            "close_error_count": self.close_error_count,
            "session_opened": self.session_opened,
            "session_closed": self.session_closed,
            "worker_thread_id": self.worker_thread_id,
            "lifecycle_thread_ids": dict(self.lifecycle_thread_ids),
        }


class LiveRigTelemetry:
    """Thread-safe counters that never retain frames or per-frame tensors."""

    def __init__(self, camera_names: tuple[str, ...] | list[str]) -> None:
        names = tuple(camera_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("camera_names must be non-empty and unique")
        self._lock = threading.Lock()
        self._cameras = {name: CameraTelemetry() for name in names}

    def record_lifecycle(self, camera_name: str, phase: str) -> None:
        thread_id = threading.get_ident()
        with self._lock:
            item = self._cameras[camera_name]
            item.worker_thread_id = thread_id
            item.lifecycle_thread_ids[phase] = thread_id
            if phase == "open":
                item.session_opened = True
            elif phase == "close":
                item.session_closed = True

    def record_frame(self, camera_name: str, frame: Any, required_streams: tuple[str, ...]) -> bool:
        timestamp = int(frame.host_receive_timestamp_ns)
        with self._lock:
            item = self._cameras[camera_name]
            previous_timestamp = item.last_host_timestamp_ns
            monotonic = previous_timestamp is None or timestamp > previous_timestamp
            if not monotonic:
                item.host_timestamp_monotonic = False
                item.host_timestamp_violations += 1
                return False
            item.captured += 1
            if item.first_host_timestamp_ns is None:
                item.first_host_timestamp_ns = timestamp
            item.last_host_timestamp_ns = timestamp
            streams = getattr(frame, "streams", {})
            for name in required_streams:
                stream = streams.get(name)
                if stream is None:
                    item.required_stream_missing[name] += 1
                    continue
                number = int(stream.frame_number)
                previous = item.last_frame_number.get(name)
                if previous is not None and number != previous + 1:
                    item.frame_number_jump_histogram.setdefault(name, Counter())[
                        _jump_bucket(number - previous)
                    ] += 1
                item.last_frame_number[name] = number
            return True

    def record_error(self, camera_name: str, phase: str, *, timeout: bool = False) -> None:
        with self._lock:
            item = self._cameras[camera_name]
            if timeout:
                item.timeout_count += 1
            if phase == "open":
                item.open_error_count += 1
            elif phase == "close":
                item.close_error_count += 1
            else:
                item.capture_error_count += 1

    def report(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {name: item.report() for name, item in sorted(self._cameras.items())}
