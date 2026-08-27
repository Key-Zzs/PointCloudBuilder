"""Bounded latest-only process wrapper for Rerun logging."""

from __future__ import annotations

from dataclasses import dataclass, replace
import multiprocessing as mp
from pathlib import Path
import queue
import time
from typing import Any

from pointcloud_builder.visualization.rerun.packet import VisualizationPacket


@dataclass(frozen=True)
class RerunOutputConfig:
    application_id: str = "pointcloud-builder-mapping"
    spawn: bool = False
    connect_url: str | None = None
    record_path: str | None = None
    queue_capacity: int = 2

    def __post_init__(self) -> None:
        if not self.application_id.strip():
            raise ValueError("Rerun application_id must be non-empty")
        if self.queue_capacity not in {1, 2}:
            raise ValueError("Rerun queue capacity must be one or two")
        if not self.spawn and self.connect_url is None and self.record_path is None:
            raise ValueError("Rerun output requires spawn, connect, or record mode")
        if self.spawn and self.connect_url is not None:
            raise ValueError(
                "Rerun spawn and explicit connect modes are mutually exclusive"
            )
        if self.connect_url is not None and not self.connect_url.strip():
            raise ValueError("Rerun connect URL must be non-empty")
        if (
            self.record_path is not None
            and Path(self.record_path).suffix.lower() != ".rrd"
        ):
            raise ValueError("Rerun record path must end in .rrd")


@dataclass(frozen=True)
class ViewerTelemetry:
    produced_packets: int
    dropped_packets: int
    maximum_queue_depth: int
    child_logged_packets: int
    child_error: str | None
    child_rss_mb: float | None
    running: bool


def _offer_status(status_queue: Any, value: tuple[str, object]) -> None:
    try:
        status_queue.put_nowait(value)
    except queue.Full:
        try:
            status_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            status_queue.put_nowait(value)
        except queue.Full:
            pass


def _logger_main(
    packet_queue: Any, status_queue: Any, config: RerunOutputConfig
) -> None:
    logger = None
    logged = 0
    try:
        from pointcloud_builder.visualization.rerun.logger import RerunPacketLogger

        logger = RerunPacketLogger(
            application_id=config.application_id,
            spawn=config.spawn,
            connect_url=config.connect_url,
            record_path=config.record_path,
        )
        _offer_status(status_queue, ("ready", True))
        while True:
            packet = packet_queue.get()
            if packet is None:
                break
            if not isinstance(packet, VisualizationPacket):
                raise TypeError("viewer child received an invalid packet")
            logger.log(packet)
            logged += 1
            _offer_status(status_queue, ("logged", logged))
        logger.close()
        _offer_status(status_queue, ("closed", logged))
    except Exception as error:
        _offer_status(
            status_queue,
            ("error", f"{type(error).__name__}: {str(error)[:500]}"),
        )
        if logger is not None:
            try:
                logger.close()
            except Exception:
                pass


class RerunViewerProcess:
    """Never-blocking producer facade over a spawned Rerun logger process."""

    def __init__(self, config: RerunOutputConfig) -> None:
        self.config = config
        self._context = mp.get_context("spawn")
        self._packet_queue: Any | None = None
        self._status_queue: Any | None = None
        self._process: mp.Process | None = None
        self._produced = 0
        self._dropped = 0
        self._maximum_depth = 0
        self._logged = 0
        self._error: str | None = None
        self._ready = False
        self._peak_child_rss_mb: float | None = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def start(self, *, timeout_s: float = 30.0) -> None:
        if self._process is not None:
            raise RuntimeError("Rerun viewer process has already been started")
        self._packet_queue = self._context.Queue(maxsize=self.config.queue_capacity)
        self._status_queue = self._context.Queue(maxsize=16)
        self._process = self._context.Process(
            target=_logger_main,
            args=(self._packet_queue, self._status_queue, self.config),
            name="pointcloud-builder-rerun",
            daemon=False,
        )
        self._process.start()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._drain_status()
            if self._error is not None:
                break
            if self._ready:
                return
            if self.running and self._status_queue is not None:
                try:
                    kind, value = self._status_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                self._handle_status(kind, value)
                if self._ready:
                    return
            if self._process is not None and not self._process.is_alive():
                break
        self._drain_status()
        startup_error = self._error or "Rerun viewer did not become ready"
        self.close(timeout_s=5.0)
        raise RuntimeError(f"Rerun viewer failed to start: {startup_error}")

    def submit(self, packet: VisualizationPacket) -> bool:
        if not isinstance(packet, VisualizationPacket):
            raise TypeError("viewer submit requires a VisualizationPacket")
        if not self.running or self._packet_queue is None:
            self._drain_status()
            return False
        self._produced += 1
        try:
            self._packet_queue.put_nowait(packet)
        except queue.Full:
            try:
                dropped_packet = self._packet_queue.get_nowait()
                if isinstance(dropped_packet, VisualizationPacket):
                    packet = _carry_static_map(dropped_packet, packet)
                self._dropped += 1
            except queue.Empty:
                pass
            try:
                self._packet_queue.put_nowait(packet)
            except queue.Full:
                self._dropped += 1
                return False
        try:
            depth = self._packet_queue.qsize()
        except (NotImplementedError, AttributeError):
            depth = self.config.queue_capacity
        self._maximum_depth = max(
            self._maximum_depth, min(depth, self.config.queue_capacity)
        )
        self._drain_status()
        self._sample_child_rss()
        return True

    def telemetry(self) -> ViewerTelemetry:
        self._drain_status()
        self._sample_child_rss()
        return ViewerTelemetry(
            produced_packets=self._produced,
            dropped_packets=self._dropped,
            maximum_queue_depth=self._maximum_depth,
            child_logged_packets=self._logged,
            child_error=self._error,
            child_rss_mb=self._peak_child_rss_mb,
            running=self.running,
        )

    def close(self, *, timeout_s: float = 30.0) -> ViewerTelemetry:
        process = self._process
        packet_queue = self._packet_queue
        if process is None:
            return self.telemetry()
        self._sample_child_rss()
        if process.is_alive() and packet_queue is not None:
            sentinel_deadline = time.monotonic() + min(max(timeout_s, 0.0), 5.0)
            while process.is_alive() and time.monotonic() < sentinel_deadline:
                try:
                    packet_queue.put_nowait(None)
                    break
                except queue.Full:
                    try:
                        packet_queue.get_nowait()
                        self._dropped += 1
                    except queue.Empty:
                        # multiprocessing.Queue can hold its capacity semaphore
                        # while the feeder has not yet made an item readable.
                        time.sleep(0.005)
        deadline = time.monotonic() + max(0.0, timeout_s)
        while process.is_alive() and time.monotonic() < deadline:
            self._drain_status()
            process.join(timeout=0.05)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
            self._error = (
                self._error
                or "viewer process exceeded close timeout and was terminated"
            )
        self._drain_status()
        return self.telemetry()

    def _drain_status(self) -> None:
        if self._status_queue is None:
            return
        while True:
            try:
                kind, value = self._status_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_status(kind, value)

    def _handle_status(self, kind: str, value: object) -> None:
        if kind == "ready":
            self._ready = bool(value)
        elif kind in {"logged", "closed"}:
            self._logged = max(self._logged, int(value))
        elif kind == "error":
            self._error = str(value)

    def _child_rss_mb(self) -> float | None:
        process = self._process
        if process is None or process.pid is None:
            return None
        try:
            lines = (
                Path(f"/proc/{process.pid}/status")
                .read_text(encoding="utf-8")
                .splitlines()
            )
        except (FileNotFoundError, OSError):
            return None
        for line in lines:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
        return None

    def _sample_child_rss(self) -> None:
        current = self._child_rss_mb()
        if current is not None:
            self._peak_child_rss_mb = max(self._peak_child_rss_mb or 0.0, current)

    def __enter__(self) -> "RerunViewerProcess":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _carry_static_map(
    dropped: VisualizationPacket, replacement: VisualizationPacket
) -> VisualizationPacket:
    """Preserve a not-yet-consumed static revision across latest-only drops."""

    old_map = dropped.map
    new_map = replacement.map
    if (
        old_map is None
        or new_map is None
        or old_map.static_revision != new_map.static_revision
        or (
            new_map.tsdf_points is not None
            or new_map.tsdf_points_raw is not None
            or new_map.tsdf_points_cropped is not None
            or new_map.tsdf_points_sampled is not None
            or new_map.tsdf_mesh is not None
        )
        or (
            old_map.tsdf_points is None
            and old_map.tsdf_points_raw is None
            and old_map.tsdf_points_cropped is None
            and old_map.tsdf_points_sampled is None
            and old_map.tsdf_mesh is None
        )
    ):
        return replacement
    return replace(
        replacement,
        map=replace(
            new_map,
            tsdf_points=old_map.tsdf_points,
            tsdf_points_raw=old_map.tsdf_points_raw,
            tsdf_points_cropped=old_map.tsdf_points_cropped,
            tsdf_points_sampled=old_map.tsdf_points_sampled,
            tsdf_mesh=old_map.tsdf_mesh,
        ),
    )
