from __future__ import annotations

import threading

import pytest

from pointcloud_builder.rig.live_buffer import LatestFrameBuffer, create_live_frame_buffers
from pointcloud_builder.rig.types import CameraFrameEnvelope


def _envelope(name: str, index: int, timestamp_ns: int | None = None) -> CameraFrameEnvelope:
    return CameraFrameEnvelope(
        camera_name=name,
        frame_index=index,
        host_receive_timestamp_ns=index if timestamp_ns is None else timestamp_ns,
        frame=object(),
    )


@pytest.mark.parametrize("capacity", [0, 5, -1, True, 2.5])
def test_live_buffer_capacity_is_strictly_bounded(capacity) -> None:
    with pytest.raises(ValueError, match="1 to 4"):
        LatestFrameBuffer("camera_a", capacity=capacity)


def test_live_buffer_explicitly_drops_oldest_and_tracks_bounded_depth() -> None:
    buffer = LatestFrameBuffer("camera_a", capacity=2)
    assert buffer.push(_envelope("camera_a", 0, 100)) is None
    assert buffer.push(_envelope("camera_a", 1, 200)) is None
    evicted = buffer.push(_envelope("camera_a", 2, 300))
    assert evicted is not None and evicted.frame_index == 0
    assert [item.frame_index for item in buffer.snapshot()] == [1, 2]
    stats = buffer.stats()
    assert stats.captured == stats.enqueued == 3
    assert stats.dropped_stale == 1
    assert stats.maximum_depth == stats.current_depth == 2
    assert stats.last_capture_timestamp_ns == 300


def test_live_buffer_consume_discard_and_identity_validation() -> None:
    buffer = LatestFrameBuffer("camera_a", capacity=4)
    frames = [_envelope("camera_a", index, 100 + index) for index in range(3)]
    for frame in frames:
        buffer.push(frame)
    removed = buffer.consume_through(frames[1])
    assert [item.frame_index for item in removed] == [0, 1]
    assert [item.frame_index for item in buffer.snapshot()] == [2]
    assert buffer.discard_oldest() == frames[2]
    assert buffer.discard_oldest() is None
    assert buffer.stats().consumed == 2
    assert buffer.stats().discarded_by_matcher == 1
    with pytest.raises(ValueError, match="cannot accept"):
        buffer.push(_envelope("camera_b", 3, 103))


def test_live_buffer_rejects_nonmonotonic_capture_order() -> None:
    buffer = LatestFrameBuffer("camera_a")
    buffer.push(_envelope("camera_a", 1, 200))
    with pytest.raises(ValueError, match="frame_index must increase"):
        buffer.push(_envelope("camera_a", 1, 201))
    with pytest.raises(ValueError, match="host timestamp must be monotonic"):
        buffer.push(_envelope("camera_a", 2, 199))


def test_wait_for_update_and_close_wake_condition_without_polling() -> None:
    buffers = create_live_frame_buffers(["camera_b", "camera_a"], capacity=2)
    assert list(buffers) == ["camera_a", "camera_b"]
    assert buffers["camera_a"].condition is buffers["camera_b"].condition
    buffer = buffers["camera_a"]
    generation = buffer.generation
    started = threading.Event()
    observed: list[int] = []

    def waiter() -> None:
        started.set()
        observed.append(buffer.wait_for_update(generation, 1.0))

    thread = threading.Thread(target=waiter)
    thread.start()
    assert started.wait(1.0)
    buffer.push(_envelope("camera_a", 0, 100))
    thread.join(1.0)
    assert not thread.is_alive()
    assert observed and observed[0] > generation

    notified_generation = buffer.generation
    buffer.notify_all()
    assert buffer.generation > notified_generation
    buffer.close()
    buffer.close()
    assert buffer.closed
    with pytest.raises(RuntimeError, match="closed"):
        buffer.push(_envelope("camera_a", 1, 200))
