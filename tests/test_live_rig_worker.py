from __future__ import annotations

from dataclasses import dataclass
import queue
from types import SimpleNamespace
import threading

from pointcloud_builder.rig.live_telemetry import LiveRigTelemetry
from pointcloud_builder.rig.live_worker import LiveCameraWorker


@dataclass
class _Stream:
    frame_number: int


@dataclass
class _Frame:
    camera_name: str
    host_receive_timestamp_ns: int
    number: int

    @property
    def streams(self):
        return {
            name: _Stream(self.number)
            for name in ("color", "depth", "ir_left", "ir_right")
        }


class _Buffer:
    def __init__(self, stop_event: threading.Event, stop_after: int = 3) -> None:
        self.stop_event = stop_event
        self.stop_after = stop_after
        self.items = []
        self.notifications = 0

    def push(self, value) -> None:
        self.items.append(value)
        if len(self.items) >= self.stop_after:
            self.stop_event.set()

    def notify_all(self) -> None:
        self.notifications += 1


class _Session:
    def __init__(self, *, fail_phase: str | None = None, reverse: bool = False) -> None:
        self.fail_phase = fail_phase
        self.reverse = reverse
        self.thread_ids: dict[str, list[int]] = {}
        self.index = 0
        self.close_count = 0

    def open(self) -> None:
        self.thread_ids.setdefault("open", []).append(threading.get_ident())
        if self.fail_phase == "open":
            raise RuntimeError("open failed")

    def capture(self):
        self.thread_ids.setdefault("capture", []).append(threading.get_ident())
        if self.fail_phase == "capture" and self.index == 1:
            raise RuntimeError("capture failed")
        timestamp = 100 + self.index
        if self.reverse and self.index == 1:
            timestamp = 99
        frame = _Frame("camera_a", timestamp, self.index)
        self.index += 1
        return frame

    def close(self) -> None:
        self.thread_ids.setdefault("close", []).append(threading.get_ident())
        self.close_count += 1
        if self.fail_phase == "close":
            raise RuntimeError("close failed")


def _config():
    return SimpleNamespace(camera=SimpleNamespace(name="camera_a"))


def _run(session: _Session, *, stop_after: int = 3):
    stop = threading.Event()
    errors = queue.Queue()
    telemetry = LiveRigTelemetry(["camera_a"])
    buffer = _Buffer(stop, stop_after=stop_after)
    def factory(_):
        session.thread_ids.setdefault("factory", []).append(threading.get_ident())
        return session

    worker = LiveCameraWorker(
        "camera_a",
        _config(),
        buffer=buffer,
        stop_event=stop,
        error_queue=errors,
        telemetry=telemetry,
        session_factory=factory,
    )
    worker.start()
    worker.join(timeout=2.0)
    assert not worker.is_alive
    return worker, buffer, errors, telemetry.report()["camera_a"]


def test_worker_owns_create_open_capture_and_close_in_one_thread() -> None:
    session = _Session()
    _, buffer, errors, report = _run(session)
    assert errors.empty()
    assert len(buffer.items) == 3
    all_ids = {
        thread_id
        for values in session.thread_ids.values()
        for thread_id in values
    }
    assert len(all_ids) == 1
    assert report["captured"] == 3
    assert report["host_timestamp_monotonic"]
    assert report["session_opened"] and report["session_closed"]
    assert set(report["lifecycle_thread_ids"].values()) == all_ids


def test_capture_failure_reports_error_stops_and_closes() -> None:
    session = _Session(fail_phase="capture")
    _, buffer, errors, report = _run(session, stop_after=10)
    error = errors.get_nowait()
    assert error.phase == "capture"
    assert error.error_type == "RuntimeError"
    assert len(buffer.items) == 1
    assert session.close_count == 1
    assert report["capture_error_count"] == 1
    assert report["session_closed"]


def test_open_failure_is_closed_by_source_and_reported() -> None:
    session = _Session(fail_phase="open")
    _, buffer, errors, report = _run(session, stop_after=10)
    error = errors.get_nowait()
    assert error.phase == "open"
    assert not buffer.items
    assert session.close_count == 1
    assert report["open_error_count"] == 1
    assert not report["session_opened"]


def test_timestamp_reversal_is_never_enqueued() -> None:
    session = _Session(reverse=True)
    _, buffer, errors, report = _run(session, stop_after=10)
    error = errors.get_nowait()
    assert error.phase == "capture"
    assert len(buffer.items) == 1
    assert report["host_timestamp_monotonic"] is False
    assert report["host_timestamp_violations"] == 1


def test_close_failure_is_reported_without_leaving_thread() -> None:
    session = _Session(fail_phase="close")
    worker, _, errors, report = _run(session)
    error = errors.get_nowait()
    assert error.phase == "close"
    assert not worker.is_alive
    assert report["close_error_count"] == 1
    assert not report["session_closed"]
