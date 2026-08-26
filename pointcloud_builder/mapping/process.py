"""Bounded latest-only asynchronous TSDF mapper process."""

from __future__ import annotations

from dataclasses import dataclass, replace
import multiprocessing as mp
from pathlib import Path
import queue
import threading
import time
from typing import Any

import numpy as np

from pointcloud_builder.mapping.config import TsdfMapConfig
from pointcloud_builder.mapping.types import (
    DynamicMaskReport,
    MapExtraction,
    RigDepthFrameSet,
    TsdfMapState,
)


@dataclass(frozen=True)
class MapperProcessConfig:
    tsdf: TsdfMapConfig
    workspace_frame: str
    initial_volume_path: str | None = None

    def __post_init__(self) -> None:
        if not self.workspace_frame.strip():
            raise ValueError("mapper workspace_frame must be non-empty")
        if (
            self.tsdf.dynamic.mode == "frozen_static"
            and self.initial_volume_path is None
        ):
            raise ValueError("frozen_static mapper requires an initial TSDF volume")
        if (
            self.initial_volume_path is not None
            and Path(self.initial_volume_path).suffix != ".npz"
        ):
            raise ValueError("initial TSDF volume path must end in .npz")


@dataclass(frozen=True)
class MapperSnapshot:
    matched_set_index: int
    map_state: TsdfMapState
    extraction: MapExtraction
    dynamic_reports: tuple[DynamicMaskReport, ...]
    dynamic_masks: tuple[tuple[str, np.ndarray], ...]
    raycast_depths: tuple[tuple[str, np.ndarray], ...]
    integration_ms: float
    active_voxel_count: int


@dataclass(frozen=True)
class MapperTelemetry:
    submitted_frame_sets: int
    producer_dropped_frame_sets: int
    maximum_queue_depth: int
    child_received_frame_sets: int
    child_rate_limited_frame_sets: int
    child_control_discarded_frame_sets: int
    child_integrated_frame_sets: int
    child_snapshots: int
    child_error: str | None
    child_rss_mb: float | None
    child_rss_samples_mb: tuple[tuple[int, float], ...]
    running: bool


def _offer_latest(target: Any, value: object) -> bool:
    try:
        target.put_nowait(value)
        return True
    except queue.Full:
        try:
            target.get_nowait()
        except queue.Empty:
            pass
        try:
            target.put_nowait(value)
            return True
        except queue.Full:
            return False


def _mapper_main(
    frame_queue: Any,
    control_queue: Any,
    output_queue: Any,
    status_queue: Any,
    config: MapperProcessConfig,
) -> None:
    mapper = None
    received = rate_limited = control_discarded = integrated = snapshots = 0
    try:
        from pointcloud_builder.mapping.open3d import Open3dTsdfMap

        mapper = Open3dTsdfMap(config.tsdf, workspace_frame=config.workspace_frame)
        if config.initial_volume_path is not None:
            mapper.load(config.initial_volume_path)
            if config.tsdf.dynamic.mode != "frozen_static":
                mapper.unfreeze()
        guarded = None
        if config.tsdf.dynamic.mode == "guarded_continuous":
            from pointcloud_builder.mapping.guarded import GuardedDepthFilter

            guarded = GuardedDepthFilter(config.tsdf.dynamic)
        _offer_latest(status_queue, ("ready", mapper.state))
        if config.initial_volume_path is not None:
            initial_statistics = mapper.volume_statistics()
            _offer_latest(
                output_queue,
                MapperSnapshot(
                    matched_set_index=0,
                    map_state=mapper.state,
                    extraction=mapper.extract(),
                    dynamic_reports=(),
                    dynamic_masks=(),
                    raycast_depths=(),
                    integration_ms=0.0,
                    active_voxel_count=int(
                        initial_statistics["attributes"]["weight"].get(
                            "nonzero_count", 0
                        )
                    ),
                ),
            )
            snapshots = 1
        last_update = 0.0
        last_extraction = 0.0
        pending_barriers: set[int] = set()
        while True:
            command = None
            try:
                command = control_queue.get_nowait()
            except queue.Empty:
                pass
            if command is not None:
                name, command_id, payload, barrier_id = command
                if name == "stop":
                    _offer_latest(status_queue, ("ack", (command_id, mapper.state)))
                    break
                if name == "freeze":
                    state = mapper.freeze()
                elif name == "unfreeze":
                    state = mapper.unfreeze()
                elif name == "reset":
                    state = mapper.reset()
                    if guarded is not None:
                        guarded.reset()
                elif name == "save":
                    mapper.save(payload)
                    state = mapper.state
                else:
                    raise ValueError(f"unsupported mapper command: {name}")
                if barrier_id is not None:
                    if int(barrier_id) in pending_barriers:
                        pending_barriers.remove(int(barrier_id))
                    else:
                        control_discarded += _drain_to_barrier(
                            frame_queue, int(barrier_id)
                        )
                _offer_latest(status_queue, ("ack", (command_id, state)))
                continue
            try:
                frame_set = frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if (
                isinstance(frame_set, tuple)
                and len(frame_set) == 2
                and frame_set[0] == "lifecycle_barrier"
            ):
                pending_barriers.add(int(frame_set[1]))
                continue
            if not isinstance(frame_set, RigDepthFrameSet):
                raise TypeError("mapper child received an invalid depth frame set")
            received += 1
            now = time.monotonic()
            if now - last_update < 1.0 / config.tsdf.integration.maximum_update_hz:
                rate_limited += 1
                _offer_latest(
                    status_queue,
                    (
                        "counts",
                        (
                            received,
                            rate_limited,
                            control_discarded,
                            integrated,
                            snapshots,
                        ),
                    ),
                )
                continue
            last_update = now
            reports: tuple[DynamicMaskReport, ...] = ()
            masks: tuple[tuple[str, np.ndarray], ...] = ()
            raycast_depths: tuple[tuple[str, np.ndarray], ...] = ()
            integration_input = frame_set
            if guarded is not None:
                predictions = tuple(
                    mapper.raycast_depth(observation)
                    for observation in frame_set.observations
                )
                decisions = tuple(
                    guarded.apply(observation, predicted)
                    for observation, predicted in zip(
                        frame_set.observations, predictions, strict=True
                    )
                )
                integration_input = replace(
                    frame_set,
                    observations=tuple(item.observation for item in decisions),
                )
                reports = tuple(item.report for item in decisions)
                masks = tuple(
                    (item.observation.camera_name, item.dynamic_mask)
                    for item in decisions
                )
                raycast_depths = tuple(
                    (observation.camera_name, predicted)
                    for observation, predicted in zip(
                        frame_set.observations, predictions, strict=True
                    )
                )
            result = mapper.integrate(integration_input)
            if not result.skipped:
                integrated += 1
            if (
                not result.skipped
                and now - last_extraction
                >= 1.0 / config.tsdf.integration.maximum_mesh_hz
            ):
                last_extraction = now
                snapshot = MapperSnapshot(
                    matched_set_index=frame_set.matched_set_index,
                    map_state=mapper.state,
                    extraction=mapper.extract(),
                    dynamic_reports=reports,
                    dynamic_masks=masks,
                    raycast_depths=raycast_depths,
                    integration_ms=result.integration_ms,
                    active_voxel_count=int(
                        mapper.volume_statistics()["attributes"]["weight"].get(
                            "nonzero_count", 0
                        )
                    ),
                )
                _offer_latest(output_queue, snapshot)
                snapshots += 1
            _offer_latest(
                status_queue,
                (
                    "counts",
                    (
                        received,
                        rate_limited,
                        control_discarded,
                        integrated,
                        snapshots,
                    ),
                ),
            )
    except Exception as error:
        _offer_latest(
            status_queue,
            ("error", f"{type(error).__name__}: {str(error)[:500]}"),
        )
    finally:
        if mapper is not None:
            mapper.close()
        _offer_latest(
            status_queue,
            (
                "closed",
                (
                    received,
                    rate_limited,
                    control_discarded,
                    integrated,
                    snapshots,
                ),
            ),
        )


def _drain_to_barrier(
    frame_queue: Any, barrier_id: int, *, timeout_s: float = 5.0
) -> int:
    """Discard FIFO predecessors until the parent's exact barrier token arrives."""

    discarded = 0
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            value = frame_queue.get(
                timeout=min(0.05, max(deadline - time.monotonic(), 0.0))
            )
        except queue.Empty:
            continue
        if isinstance(value, tuple) and value == ("lifecycle_barrier", barrier_id):
            return discarded
        if not isinstance(value, RigDepthFrameSet):
            raise TypeError(
                "mapper frame queue contained an invalid barrier predecessor"
            )
        discarded += 1
    raise RuntimeError("mapper lifecycle frame barrier timed out")


class AsyncTsdfMapper:
    """Non-blocking producer facade for fixed-camera TSDF integration."""

    def __init__(self, config: MapperProcessConfig) -> None:
        self.config = config
        self._context = mp.get_context("spawn")
        self._frame_queue: Any | None = None
        self._control_queue: Any | None = None
        self._output_queue: Any | None = None
        self._status_queue: Any | None = None
        self._process: mp.Process | None = None
        self._submitted = 0
        self._dropped = 0
        self._maximum_depth = 0
        self._received = self._rate_limited = self._control_discarded = 0
        self._integrated = self._snapshots = 0
        self._error: str | None = None
        self._ready = False
        self._command_id = 0
        self._acks: dict[int, TsdfMapState] = {}
        self._peak_rss_mb: float | None = None
        self._rss_samples_mb: list[tuple[int, float]] = []
        self._operation_lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def start(self, *, timeout_s: float = 30.0) -> None:
        if self._process is not None:
            raise RuntimeError("TSDF mapper has already been started")
        capacity = self.config.tsdf.integration.queue_capacity
        self._frame_queue = self._context.Queue(maxsize=capacity)
        self._control_queue = self._context.Queue(maxsize=16)
        self._output_queue = self._context.Queue(maxsize=1)
        self._status_queue = self._context.Queue(maxsize=16)
        self._process = self._context.Process(
            target=_mapper_main,
            args=(
                self._frame_queue,
                self._control_queue,
                self._output_queue,
                self._status_queue,
                self.config,
            ),
            name="pointcloud-builder-tsdf",
            daemon=False,
        )
        self._process.start()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._drain_status()
            if self._ready:
                return
            if self._error is not None or not self.running:
                break
            time.sleep(0.01)
        self._drain_status()
        startup_error = self._error or "TSDF mapper did not become ready"
        self.close(timeout_s=5.0)
        raise RuntimeError(startup_error)

    def submit(self, frame_set: RigDepthFrameSet) -> bool:
        if not isinstance(frame_set, RigDepthFrameSet):
            raise TypeError("mapper submit requires RigDepthFrameSet")
        if not self.running or self._frame_queue is None:
            self._drain_status()
            return False
        self._submitted += 1
        if not self._operation_lock.acquire(blocking=False):
            self._dropped += 1
            return False
        try:
            try:
                self._frame_queue.put_nowait(frame_set)
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                    self._dropped += 1
                except queue.Empty:
                    pass
                try:
                    self._frame_queue.put_nowait(frame_set)
                except queue.Full:
                    self._dropped += 1
                    return False
        finally:
            self._operation_lock.release()
        try:
            depth = self._frame_queue.qsize()
        except (AttributeError, NotImplementedError):
            depth = self.config.tsdf.integration.queue_capacity
        self._maximum_depth = max(self._maximum_depth, depth)
        self._drain_status()
        self._sample_rss()
        return True

    def poll_snapshot(self) -> MapperSnapshot | None:
        if self._output_queue is None:
            return None
        latest = None
        while True:
            try:
                value = self._output_queue.get_nowait()
            except queue.Empty:
                break
            if not isinstance(value, MapperSnapshot):
                raise TypeError("mapper output queue contained an invalid snapshot")
            latest = value
        return latest

    def wait_for_snapshot(self, *, timeout_s: float = 30.0) -> MapperSnapshot:
        """Wait for the first bounded mapper snapshot before live acquisition."""

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            snapshot = self.poll_snapshot()
            if snapshot is not None:
                return snapshot
            self._drain_status()
            if self._error is not None or not self.running:
                break
            time.sleep(0.01)
        raise RuntimeError(self._error or "TSDF mapper snapshot timed out")

    def sample_resources(self, frame_index: int) -> None:
        """Sample child RSS without submitting a depth frame to a frozen map."""

        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise TypeError("mapper resource sample index must be an integer")
        if frame_index < 0:
            raise ValueError("mapper resource sample index must be non-negative")
        if self._rss_samples_mb and frame_index <= self._rss_samples_mb[-1][0]:
            raise ValueError("mapper resource sample indices must increase")
        self._drain_status()
        if not self.running:
            raise RuntimeError(self._error or "TSDF mapper is not running")
        self._sample_rss(frame_index)

    def freeze(self, *, timeout_s: float = 30.0) -> TsdfMapState:
        return self._command("freeze", timeout_s=timeout_s)

    def unfreeze(self, *, timeout_s: float = 30.0) -> TsdfMapState:
        return self._command("unfreeze", timeout_s=timeout_s)

    def reset(self, *, timeout_s: float = 30.0) -> TsdfMapState:
        return self._command("reset", timeout_s=timeout_s)

    def save_volume(self, path: str | Path, *, timeout_s: float = 30.0) -> TsdfMapState:
        output = Path(path)
        if output.suffix.lower() != ".npz":
            raise ValueError("TSDF volume path must end in .npz")
        return self._command("save", payload=str(output), timeout_s=timeout_s)

    def telemetry(self) -> MapperTelemetry:
        self._drain_status()
        self._sample_rss()
        return MapperTelemetry(
            submitted_frame_sets=self._submitted,
            producer_dropped_frame_sets=self._dropped,
            maximum_queue_depth=min(
                self._maximum_depth, self.config.tsdf.integration.queue_capacity
            ),
            child_received_frame_sets=self._received,
            child_rate_limited_frame_sets=self._rate_limited,
            child_control_discarded_frame_sets=self._control_discarded,
            child_integrated_frame_sets=self._integrated,
            child_snapshots=self._snapshots,
            child_error=self._error,
            child_rss_mb=self._peak_rss_mb,
            child_rss_samples_mb=tuple(self._rss_samples_mb),
            running=self.running,
        )

    def close(self, *, timeout_s: float = 30.0) -> MapperTelemetry:
        if self._process is None:
            return self.telemetry()
        if self.running:
            try:
                self._command("stop", timeout_s=min(timeout_s, 5.0))
            except RuntimeError as error:
                self._error = self._error or str(error)
        deadline = time.monotonic() + max(timeout_s, 0.0)
        while self._process.is_alive() and time.monotonic() < deadline:
            # A large extraction can still be in the multiprocessing feeder
            # pipe. Drain it while joining so the child can exit cleanly.
            self.poll_snapshot()
            self._drain_status()
            self._process.join(timeout=0.05)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)
            self._error = (
                self._error or "TSDF mapper exceeded close timeout and was terminated"
            )
        self._drain_status()
        return self.telemetry()

    def _command(
        self, name: str, *, payload: object = None, timeout_s: float
    ) -> TsdfMapState:
        if not self.running or self._control_queue is None:
            raise RuntimeError("TSDF mapper is not running")
        if not self._operation_lock.acquire(timeout=timeout_s):
            raise RuntimeError(f"TSDF mapper command lock timed out: {name}")
        try:
            self._command_id += 1
            command_id = self._command_id
            barrier_id = command_id if name in {"freeze", "unfreeze", "reset"} else None
            if barrier_id is not None:
                assert self._frame_queue is not None
                self._frame_queue.put(
                    ("lifecycle_barrier", barrier_id), timeout=timeout_s
                )
            self._control_queue.put(
                (name, command_id, payload, barrier_id), timeout=timeout_s
            )
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                self._drain_status()
                state = self._acks.pop(command_id, None)
                if state is not None:
                    return state
                if self._error is not None or not self.running:
                    break
                time.sleep(0.01)
            raise RuntimeError(self._error or f"TSDF mapper command timed out: {name}")
        finally:
            self._operation_lock.release()

    def _drain_status(self) -> None:
        if self._status_queue is None:
            return
        while True:
            try:
                kind, value = self._status_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "ready":
                self._ready = True
            elif kind in {"counts", "closed"}:
                (
                    self._received,
                    self._rate_limited,
                    self._control_discarded,
                    self._integrated,
                    self._snapshots,
                ) = (int(item) for item in value)
            elif kind == "ack":
                command_id, state = value
                self._acks[int(command_id)] = state
            elif kind == "error":
                self._error = str(value)

    def _sample_rss(self, sample_index: int | None = None) -> None:
        process = self._process
        if process is None or process.pid is None:
            return
        try:
            lines = (
                Path(f"/proc/{process.pid}/status")
                .read_text(encoding="utf-8")
                .splitlines()
            )
        except (FileNotFoundError, OSError):
            return
        for line in lines:
            if line.startswith("VmRSS:"):
                value = float(line.split()[1]) / 1024.0
                self._peak_rss_mb = max(self._peak_rss_mb or 0.0, value)
                sample = (
                    self._submitted if sample_index is None else sample_index,
                    value,
                )
                if self._rss_samples_mb and self._rss_samples_mb[-1][0] == sample[0]:
                    self._rss_samples_mb[-1] = sample
                elif self._rss_samples_mb and self._rss_samples_mb[-1][0] > sample[0]:
                    return
                else:
                    self._rss_samples_mb.append(sample)
                    if len(self._rss_samples_mb) > 4096:
                        del self._rss_samples_mb[: len(self._rss_samples_mb) - 4096]
                return

    def __enter__(self) -> "AsyncTsdfMapper":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
