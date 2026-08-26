"""Bounded host-monotonic-clock matching for concurrent live cameras."""

from __future__ import annotations

from collections.abc import Mapping
from collections import deque
from dataclasses import dataclass
import math
import statistics
import time

from pointcloud_builder.rig.live_buffer import LatestFrameBuffer
from pointcloud_builder.rig.types import CameraFrameEnvelope, RigFrameSet


@dataclass(frozen=True)
class LiveMatcherStats:
    """Constant-memory counters for one live matcher run."""

    reference_frames_considered: int
    matched_sets: int
    impossible_drops: int
    impossible_drops_by_camera: dict[str, int]
    lookahead_drops: int
    wait_count: int
    wait_time_ms: float
    wait_timeouts: int
    closed_before_match: int
    frame_reuse_violations: int
    last_per_camera_delta_ms: dict[str, float]
    maximum_absolute_skew_ms: float
    match_ratio: float
    signed_skew_ms: dict[str, dict[str, float | int | None]]
    absolute_skew_ms: dict[str, dict[str, float | int | None]]
    matcher_wait_ms: dict[str, float | int | None]
    telemetry_sample_evictions: int


class LiveRigFrameMatcher:
    """Consume complete, nearest-host-timestamp frame sets at most once."""

    def __init__(
        self,
        buffers: Mapping[str, LatestFrameBuffer],
        *,
        reference_camera: str,
        maximum_skew_ms: float,
        telemetry_history_capacity: int = 8192,
    ) -> None:
        if not buffers:
            raise ValueError("live matcher requires at least one camera buffer")
        if not math.isfinite(maximum_skew_ms) or maximum_skew_ms < 0:
            raise ValueError("maximum_skew_ms must be finite and non-negative")
        if (
            isinstance(telemetry_history_capacity, bool)
            or not isinstance(telemetry_history_capacity, int)
            or not 1 <= telemetry_history_capacity <= 100_000
        ):
            raise ValueError("telemetry_history_capacity must be an integer from 1 to 100000")
        self.buffers = dict(buffers)
        if set(self.buffers) != {buffer.camera_name for buffer in self.buffers.values()}:
            raise ValueError("live buffer mapping keys must match camera names")
        if reference_camera not in self.buffers:
            raise ValueError("reference_camera must name a live buffer")
        conditions = {id(buffer.condition) for buffer in self.buffers.values()}
        if len(conditions) != 1:
            raise ValueError("all live buffers must share one threading.Condition")
        self.reference_camera = reference_camera
        self.maximum_skew_ms = float(maximum_skew_ms)
        self._maximum_skew_ns = self.maximum_skew_ms * 1_000_000.0
        self._camera_order = tuple(sorted(self.buffers))
        self._condition = next(iter(self.buffers.values())).condition
        self._reference_frames_considered = 0
        self._matched_sets = 0
        self._impossible_drops = 0
        self._impossible_drops_by_camera = {name: 0 for name in self._camera_order}
        self._lookahead_drops = 0
        self._wait_count = 0
        self._wait_time_ns = 0
        self._wait_timeouts = 0
        self._closed_before_match = 0
        self._frame_reuse_violations = 0
        self._last_considered_reference_index: int | None = None
        self._last_emitted_index = {name: -1 for name in self._camera_order}
        self._last_per_camera_delta_ms = {name: 0.0 for name in self._camera_order}
        self._maximum_absolute_skew_ms = 0.0
        self._history_capacity = telemetry_history_capacity
        self._signed_skew_samples = {
            name: deque(maxlen=telemetry_history_capacity) for name in self._camera_order
        }
        self._absolute_skew_samples = {
            name: deque(maxlen=telemetry_history_capacity) for name in self._camera_order
        }
        self._wait_samples_ms: deque[float] = deque(maxlen=telemetry_history_capacity)
        self._telemetry_sample_evictions = 0

    def match_next(self, timeout_s: float = 1.0) -> RigFrameSet | None:
        """Return one complete match, or ``None`` on timeout/exhausted close.

        The bounded timeout is deliberate: callers must regularly observe stop
        and worker-error state instead of entering an unbounded matcher wait.
        """

        if not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("timeout_s must be finite and non-negative")
        deadline_ns = time.monotonic_ns() + round(timeout_s * 1_000_000_000.0)
        with self._condition:
            while True:
                snapshots = {
                    name: self.buffers[name]._snapshot_locked() for name in self._camera_order
                }
                if all(snapshots.values()):
                    self._record_reference_candidate_locked(snapshots)
                    if self._discard_reference_if_next_is_closer_locked(snapshots):
                        continue
                    if not (
                        timeout_s > 0.0
                        and self._needs_newer_candidate_locked(snapshots)
                    ):
                        frame_set = self._try_match_locked(snapshots)
                        if frame_set is not None:
                            return frame_set
                    if self._discard_globally_oldest_impossible_locked(snapshots):
                        continue
                if any(
                    self.buffers[name]._closed and not snapshots[name]
                    for name in self._camera_order
                ):
                    self._closed_before_match += 1
                    return None
                remaining_ns = deadline_ns - time.monotonic_ns()
                if remaining_ns <= 0:
                    self._wait_timeouts += 1
                    return None
                wait_started_ns = time.monotonic_ns()
                self._wait_count += 1
                self._condition.wait(timeout=remaining_ns / 1_000_000_000.0)
                waited_ns = time.monotonic_ns() - wait_started_ns
                self._wait_time_ns += waited_ns
                self._append_bounded(self._wait_samples_ms, waited_ns / 1_000_000.0)

    def _needs_newer_candidate_locked(
        self, snapshots: Mapping[str, tuple[CameraFrameEnvelope, ...]]
    ) -> bool:
        """Wait until each non-reference stream brackets the reference.

        A latest candidate that is still older than the reference cannot yet be
        called the nearest candidate: the next frame may be much closer.  The
        bounded outer wait and latest-biased buffers prevent this look-ahead
        from becoming an unbounded backlog.
        """

        reference = snapshots[self.reference_camera][0]
        reference_frames = snapshots[self.reference_camera]
        non_reference_is_only_ahead = any(
            name != self.reference_camera
            and frames[0].host_receive_timestamp_ns
            > reference.host_receive_timestamp_ns
            for name, frames in snapshots.items()
        )
        return (
            non_reference_is_only_ahead
            and len(reference_frames) == 1
            and not self.buffers[self.reference_camera]._closed
        ) or any(
            name != self.reference_camera
            and not self.buffers[name]._closed
            and frames[-1].host_receive_timestamp_ns
            < reference.host_receive_timestamp_ns
            for name, frames in snapshots.items()
        )

    def _discard_reference_if_next_is_closer_locked(
        self, snapshots: Mapping[str, tuple[CameraFrameEnvelope, ...]]
    ) -> bool:
        reference_frames = snapshots[self.reference_camera]
        if len(reference_frames) < 2:
            return False

        def score(candidate: CameraFrameEnvelope) -> int:
            return max(
                min(
                    abs(
                        other.host_receive_timestamp_ns
                        - candidate.host_receive_timestamp_ns
                    )
                    for other in frames
                )
                for name, frames in snapshots.items()
                if name != self.reference_camera
            )

        if score(reference_frames[1]) >= score(reference_frames[0]):
            return False
        removed = self.buffers[self.reference_camera]._discard_oldest_locked()
        if removed is None:  # pragma: no cover - protected by the shared condition
            return False
        self._lookahead_drops += 1
        return True

    def _record_reference_candidate_locked(
        self, snapshots: Mapping[str, tuple[CameraFrameEnvelope, ...]]
    ) -> None:
        reference = snapshots[self.reference_camera][0]
        if reference.frame_index != self._last_considered_reference_index:
            self._reference_frames_considered += 1
            self._last_considered_reference_index = reference.frame_index

    def stats(self) -> LiveMatcherStats:
        with self._condition:
            return LiveMatcherStats(
                reference_frames_considered=self._reference_frames_considered,
                matched_sets=self._matched_sets,
                impossible_drops=self._impossible_drops,
                impossible_drops_by_camera=dict(self._impossible_drops_by_camera),
                lookahead_drops=self._lookahead_drops,
                wait_count=self._wait_count,
                wait_time_ms=self._wait_time_ns / 1_000_000.0,
                wait_timeouts=self._wait_timeouts,
                closed_before_match=self._closed_before_match,
                frame_reuse_violations=self._frame_reuse_violations,
                last_per_camera_delta_ms=dict(self._last_per_camera_delta_ms),
                maximum_absolute_skew_ms=self._maximum_absolute_skew_ms,
                match_ratio=(
                    self._matched_sets / self._reference_frames_considered
                    if self._reference_frames_considered
                    else 0.0
                ),
                signed_skew_ms={
                    name: _summary(list(values))
                    for name, values in self._signed_skew_samples.items()
                },
                absolute_skew_ms={
                    name: _summary(list(values))
                    for name, values in self._absolute_skew_samples.items()
                },
                matcher_wait_ms=_summary(list(self._wait_samples_ms)),
                telemetry_sample_evictions=self._telemetry_sample_evictions,
            )

    def _try_match_locked(
        self, snapshots: Mapping[str, tuple[CameraFrameEnvelope, ...]]
    ) -> RigFrameSet | None:
        reference = snapshots[self.reference_camera][0]
        selected = {self.reference_camera: reference}
        deltas_ms = {self.reference_camera: 0.0}
        for name in self._camera_order:
            if name == self.reference_camera:
                continue
            closest = min(
                snapshots[name],
                key=lambda candidate: (
                    abs(
                        candidate.host_receive_timestamp_ns
                        - reference.host_receive_timestamp_ns
                    ),
                    candidate.frame_index,
                ),
            )
            delta_ms = (
                closest.host_receive_timestamp_ns - reference.host_receive_timestamp_ns
            ) / 1_000_000.0
            if abs(delta_ms) > self.maximum_skew_ms:
                return None
            selected[name] = closest
            deltas_ms[name] = delta_ms
        for name, envelope in selected.items():
            if envelope.frame_index <= self._last_emitted_index[name]:
                self._frame_reuse_violations += 1
                raise RuntimeError(
                    f"live matcher attempted to reuse camera {name!r} frame {envelope.frame_index}"
                )
        for name in self._camera_order:
            self.buffers[name]._consume_through_locked(selected[name])
            self._last_emitted_index[name] = selected[name].frame_index
        observed_skew_ms = max(abs(delta) for delta in deltas_ms.values())
        self._matched_sets += 1
        self._last_per_camera_delta_ms = dict(deltas_ms)
        self._maximum_absolute_skew_ms = max(
            self._maximum_absolute_skew_ms, observed_skew_ms
        )
        for name, delta in deltas_ms.items():
            self._append_bounded(self._signed_skew_samples[name], delta)
            self._append_bounded(self._absolute_skew_samples[name], abs(delta))
        return RigFrameSet(
            envelopes=selected,
            reference_camera=self.reference_camera,
            per_camera_delta_ms=deltas_ms,
            maximum_skew_ms=observed_skew_ms,
            match_sequence_index=self._matched_sets - 1,
            match_timestamp_ns=reference.host_receive_timestamp_ns,
            per_camera_absolute_delta_ms={
                name: abs(delta) for name, delta in deltas_ms.items()
            },
            matching_policy="nearest_host_timestamp",
        )

    def _discard_globally_oldest_impossible_locked(
        self, snapshots: Mapping[str, tuple[CameraFrameEnvelope, ...]]
    ) -> bool:
        oldest_camera, oldest = min(
            ((name, frames[0]) for name, frames in snapshots.items()),
            key=lambda item: (
                item[1].host_receive_timestamp_ns,
                item[0],
                item[1].frame_index,
            ),
        )
        impossible = any(
            name != oldest_camera
            and frames[-1].host_receive_timestamp_ns
            > oldest.host_receive_timestamp_ns + self._maximum_skew_ns
            for name, frames in snapshots.items()
        )
        if not impossible:
            return False
        removed = self.buffers[oldest_camera]._discard_oldest_locked()
        if removed is None:  # pragma: no cover - protected by the shared condition
            return False
        self._impossible_drops += 1
        self._impossible_drops_by_camera[oldest_camera] += 1
        return True

    def _append_bounded(self, target: deque[float], value: float) -> None:
        if len(target) == target.maxlen:
            self._telemetry_sample_evictions += 1
        target.append(float(value))


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "mean": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "p50": statistics.median(values),
        "p95": _quantile(ordered, 0.95),
        "mean": statistics.mean(values),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _quantile(ordered: list[float], q: float) -> float:
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
