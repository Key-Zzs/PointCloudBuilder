"""Thread-safe bounded buffers for latest-biased live rig capture."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading

from pointcloud_builder.rig.types import CameraFrameEnvelope


@dataclass(frozen=True)
class LiveBufferStats:
    """Bounded-buffer counters that never retain captured frame payloads."""

    camera_name: str
    capacity: int
    captured: int
    enqueued: int
    dropped_stale: int
    discarded_by_matcher: int
    consumed: int
    maximum_depth: int
    current_depth: int
    last_capture_timestamp_ns: int | None
    generation: int
    closed: bool


class LatestFrameBuffer:
    """A capacity-limited FIFO that explicitly evicts the oldest frame.

    All buffers participating in one live rig must receive the same
    :class:`threading.Condition`.  The matcher then holds that shared condition
    while it snapshots and consumes multiple buffers, making a match atomic
    with respect to producer pushes.
    """

    def __init__(
        self,
        camera_name: str,
        *,
        capacity: int = 2,
        condition: threading.Condition | None = None,
    ) -> None:
        if not isinstance(camera_name, str) or not camera_name.strip():
            raise ValueError("camera_name must be a non-empty string")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or not 1 <= capacity <= 4:
            raise ValueError("live buffer capacity must be an integer from 1 to 4")
        self.camera_name = camera_name
        self.capacity = capacity
        self.condition = condition or threading.Condition(threading.RLock())
        self._items: deque[CameraFrameEnvelope] = deque()
        self._captured = 0
        self._enqueued = 0
        self._dropped_stale = 0
        self._discarded_by_matcher = 0
        self._consumed = 0
        self._maximum_depth = 0
        self._last_capture_timestamp_ns: int | None = None
        self._last_capture_index: int | None = None
        self._generation = 0
        self._closed = False

    @property
    def closed(self) -> bool:
        with self.condition:
            return self._closed

    @property
    def generation(self) -> int:
        with self.condition:
            return self._generation

    def push(self, envelope: CameraFrameEnvelope) -> CameraFrameEnvelope | None:
        """Append one frame and return the stale frame evicted, if any."""

        if envelope.camera_name != self.camera_name:
            raise ValueError(
                f"buffer {self.camera_name!r} cannot accept frame for {envelope.camera_name!r}"
            )
        if (
            isinstance(envelope.frame_index, bool)
            or not isinstance(envelope.frame_index, int)
            or envelope.frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative integer")
        if (
            isinstance(envelope.host_receive_timestamp_ns, bool)
            or not isinstance(envelope.host_receive_timestamp_ns, int)
            or envelope.host_receive_timestamp_ns < 0
        ):
            raise ValueError("host_receive_timestamp_ns must be a non-negative integer")
        with self.condition:
            if self._closed:
                raise RuntimeError(f"live buffer {self.camera_name!r} is closed")
            if self._last_capture_index is not None and envelope.frame_index <= self._last_capture_index:
                raise ValueError(
                    f"buffer {self.camera_name!r} frame_index must increase strictly"
                )
            if (
                self._last_capture_timestamp_ns is not None
                and envelope.host_receive_timestamp_ns < self._last_capture_timestamp_ns
            ):
                raise ValueError(
                    f"buffer {self.camera_name!r} host timestamp must be monotonic"
                )
            self._captured += 1
            evicted = None
            if len(self._items) == self.capacity:
                evicted = self._items.popleft()
                self._dropped_stale += 1
            self._items.append(envelope)
            self._enqueued += 1
            self._maximum_depth = max(self._maximum_depth, len(self._items))
            self._last_capture_index = envelope.frame_index
            self._last_capture_timestamp_ns = envelope.host_receive_timestamp_ns
            self._changed_locked()
            return evicted

    def snapshot(self) -> tuple[CameraFrameEnvelope, ...]:
        with self.condition:
            return self._snapshot_locked()

    def consume_through(
        self, selected: CameraFrameEnvelope
    ) -> tuple[CameraFrameEnvelope, ...]:
        """Consume ``selected`` and every older buffered frame atomically."""

        with self.condition:
            return self._consume_through_locked(selected)

    def discard_oldest(self) -> CameraFrameEnvelope | None:
        """Discard the oldest matcher-impossible candidate, if present."""

        with self.condition:
            return self._discard_oldest_locked()

    def wait_for_update(self, generation: int, timeout_s: float) -> int:
        """Wait boundedly for a push, consume, discard, or close notification."""

        if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("timeout_s must be finite and non-negative")
        with self.condition:
            self.condition.wait_for(
                lambda: self._generation != generation or self._closed,
                timeout=timeout_s,
            )
            return self._generation

    def close(self) -> None:
        """Close the buffer idempotently and wake every producer/consumer waiter."""

        with self.condition:
            if not self._closed:
                self._closed = True
                self._changed_locked()
            else:
                self.condition.notify_all()

    def notify_all(self) -> None:
        """Wake shared-condition waiters after an external stop or worker error."""

        with self.condition:
            self._changed_locked()

    def stats(self) -> LiveBufferStats:
        with self.condition:
            return LiveBufferStats(
                camera_name=self.camera_name,
                capacity=self.capacity,
                captured=self._captured,
                enqueued=self._enqueued,
                dropped_stale=self._dropped_stale,
                discarded_by_matcher=self._discarded_by_matcher,
                consumed=self._consumed,
                maximum_depth=self._maximum_depth,
                current_depth=len(self._items),
                last_capture_timestamp_ns=self._last_capture_timestamp_ns,
                generation=self._generation,
                closed=self._closed,
            )

    # The matcher calls these only while holding the shared condition.  Keeping
    # them separate from the public locking methods avoids releasing the lock
    # between a multi-camera snapshot and its corresponding consumption.
    def _snapshot_locked(self) -> tuple[CameraFrameEnvelope, ...]:
        return tuple(self._items)

    def _consume_through_locked(
        self, selected: CameraFrameEnvelope
    ) -> tuple[CameraFrameEnvelope, ...]:
        if selected.camera_name != self.camera_name:
            raise ValueError("selected frame camera does not match buffer")
        selected_position = None
        for position, candidate in enumerate(self._items):
            if (
                candidate.frame_index == selected.frame_index
                and candidate.host_receive_timestamp_ns == selected.host_receive_timestamp_ns
            ):
                selected_position = position
                break
        if selected_position is None:
            raise ValueError("selected frame is not present in the live buffer")
        removed = tuple(self._items.popleft() for _ in range(selected_position + 1))
        self._consumed += len(removed)
        self._changed_locked()
        return removed

    def _discard_oldest_locked(self) -> CameraFrameEnvelope | None:
        if not self._items:
            return None
        removed = self._items.popleft()
        self._discarded_by_matcher += 1
        self._changed_locked()
        return removed

    def _changed_locked(self) -> None:
        self._generation += 1
        self.condition.notify_all()


def create_live_frame_buffers(
    camera_names: tuple[str, ...] | list[str], *, capacity: int = 2
) -> dict[str, LatestFrameBuffer]:
    """Create canonically keyed buffers sharing one condition and lock."""

    names = tuple(camera_names)
    if not names:
        raise ValueError("at least one live camera name is required")
    if len(names) != len(set(names)):
        raise ValueError("live camera names must be unique")
    condition = threading.Condition(threading.RLock())
    return {
        name: LatestFrameBuffer(name, capacity=capacity, condition=condition)
        for name in sorted(names)
    }
