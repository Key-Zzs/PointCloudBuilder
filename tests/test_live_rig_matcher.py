from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import pytest

from pointcloud_builder.rig.live_buffer import create_live_frame_buffers
from pointcloud_builder.rig.live_matcher import LiveRigFrameMatcher
from pointcloud_builder.rig.types import CameraFrameEnvelope


@dataclass(frozen=True)
class DeviceClockFrame:
    sensor_timestamp_ns: int
    frame_number: int


def _envelope(
    name: str,
    index: int,
    host_ns: int,
    *,
    sensor_ns: int = 0,
    frame_number: int = 0,
) -> CameraFrameEnvelope:
    return CameraFrameEnvelope(
        camera_name=name,
        frame_index=index,
        host_receive_timestamp_ns=host_ns,
        frame=DeviceClockFrame(sensor_ns, frame_number),
    )


def _matcher(*, reverse: bool = False, gate_ms: float = 5.0):
    names = ["camera_b", "camera_a"] if reverse else ["camera_a", "camera_b"]
    buffers = create_live_frame_buffers(names, capacity=4)
    matcher = LiveRigFrameMatcher(
        dict(reversed(list(buffers.items()))) if reverse else buffers,
        reference_camera="camera_a",
        maximum_skew_ms=gate_ms,
    )
    return buffers, matcher


def test_matcher_selects_nearest_host_timestamp_and_consumes_selected_and_older() -> None:
    buffers, matcher = _matcher(gate_ms=20.0)
    buffers["camera_a"].push(_envelope("camera_a", 0, 100_000_000))
    buffers["camera_b"].push(_envelope("camera_b", 0, 70_000_000))
    buffers["camera_b"].push(_envelope("camera_b", 1, 103_000_000))
    matched = matcher.match_next(0.0)
    assert matched is not None
    assert matched.reference_camera == "camera_a"
    assert matched.envelopes["camera_b"].frame_index == 1
    assert matched.per_camera_delta_ms == {"camera_a": 0.0, "camera_b": 3.0}
    assert matched.per_camera_absolute_delta_ms == {"camera_a": 0.0, "camera_b": 3.0}
    assert matched.maximum_skew_ms == 3.0
    assert matched.match_sequence_index == 0
    assert matched.match_timestamp_ns == 100_000_000
    assert matched.matching_policy == "nearest_host_timestamp"
    assert buffers["camera_a"].snapshot() == ()
    assert buffers["camera_b"].snapshot() == ()
    assert buffers["camera_b"].stats().consumed == 2


def test_matcher_ignores_sensor_timestamps_and_device_frame_numbers() -> None:
    buffers, matcher = _matcher(gate_ms=5.0)
    buffers["camera_a"].push(
        _envelope("camera_a", 0, 1_000_000_000, sensor_ns=900_000_000_000, frame_number=1)
    )
    buffers["camera_b"].push(
        _envelope("camera_b", 0, 1_004_000_000, sensor_ns=1, frame_number=999_999)
    )
    matched = matcher.match_next(0.0)
    assert matched is not None
    assert matched.per_camera_delta_ms["camera_b"] == 4.0


def test_host_mismatch_is_rejected_even_when_device_clocks_match() -> None:
    buffers, matcher = _matcher(gate_ms=5.0)
    buffers["camera_a"].push(
        _envelope("camera_a", 0, 100_000_000, sensor_ns=55, frame_number=7)
    )
    buffers["camera_a"].push(
        _envelope("camera_a", 1, 131_000_000, sensor_ns=99, frame_number=8)
    )
    buffers["camera_b"].push(
        _envelope("camera_b", 0, 130_000_000, sensor_ns=55, frame_number=7)
    )
    buffers["camera_b"].close()
    matched = matcher.match_next(0.0)
    assert matched is not None
    assert matched.envelopes["camera_a"].frame_index == 1
    assert matched.per_camera_delta_ms["camera_b"] == -1.0
    stats = matcher.stats()
    assert stats.impossible_drops == 0
    assert stats.lookahead_drops == 1
    assert stats.impossible_drops_by_camera == {"camera_a": 0, "camera_b": 0}
    assert stats.reference_frames_considered == 2


def test_matcher_never_reuses_frames_across_emitted_sets() -> None:
    buffers, matcher = _matcher()
    used: set[tuple[str, int]] = set()
    for index, base_ns in enumerate((100_000_000, 200_000_000)):
        buffers["camera_a"].push(_envelope("camera_a", index, base_ns))
        buffers["camera_b"].push(_envelope("camera_b", index, base_ns + 2_000_000))
        matched = matcher.match_next(0.0)
        assert matched is not None
        keys = {(name, envelope.frame_index) for name, envelope in matched.envelopes.items()}
        assert used.isdisjoint(keys)
        used.update(keys)
    assert matcher.stats().matched_sets == 2
    assert matcher.stats().frame_reuse_violations == 0


def test_camera_mapping_order_does_not_change_matching_or_tie_break() -> None:
    results = []
    for reverse in (False, True):
        buffers, matcher = _matcher(reverse=reverse, gate_ms=20.0)
        buffers["camera_a"].push(_envelope("camera_a", 0, 100_000_000))
        buffers["camera_b"].push(_envelope("camera_b", 0, 90_000_000))
        buffers["camera_b"].push(_envelope("camera_b", 1, 110_000_000))
        matched = matcher.match_next(0.0)
        assert matched is not None
        results.append(
            (
                tuple(matched.envelopes),
                matched.envelopes["camera_b"].frame_index,
                matched.per_camera_delta_ms,
            )
        )
    assert results[0] == results[1]
    assert results[0][1] == 0


def test_matcher_wait_is_bounded_and_closed_empty_buffer_ends_promptly() -> None:
    buffers, matcher = _matcher()
    assert matcher.match_next(0.01) is None
    assert matcher.stats().wait_timeouts == 1
    buffers["camera_a"].close()
    assert matcher.match_next(1.0) is None
    assert matcher.stats().closed_before_match == 1


def test_matcher_waits_for_a_newer_candidate_before_selecting_nearest() -> None:
    buffers, matcher = _matcher(gate_ms=40.0)
    buffers["camera_a"].push(_envelope("camera_a", 0, 100_000_000))
    buffers["camera_b"].push(_envelope("camera_b", 0, 70_000_000))

    def publish_bracketing_frame() -> None:
        time.sleep(0.01)
        buffers["camera_b"].push(_envelope("camera_b", 1, 103_000_000))

    producer = threading.Thread(target=publish_bracketing_frame)
    producer.start()
    matched = matcher.match_next(0.1)
    producer.join()
    assert matched is not None
    assert matched.envelopes["camera_b"].frame_index == 1
    assert matched.per_camera_delta_ms["camera_b"] == 3.0


@pytest.mark.parametrize("gate", [-1.0, float("nan"), float("inf")])
def test_matcher_rejects_invalid_skew_gate(gate: float) -> None:
    buffers = create_live_frame_buffers(["camera_a", "camera_b"])
    with pytest.raises(ValueError, match="finite and non-negative"):
        LiveRigFrameMatcher(
            buffers,
            reference_camera="camera_a",
            maximum_skew_ms=gate,
        )


def test_matcher_requires_one_shared_condition_and_valid_reference() -> None:
    left = create_live_frame_buffers(["camera_a"])["camera_a"]
    right = create_live_frame_buffers(["camera_b"])["camera_b"]
    with pytest.raises(ValueError, match="share one"):
        LiveRigFrameMatcher(
            {"camera_a": left, "camera_b": right},
            reference_camera="camera_a",
            maximum_skew_ms=5.0,
        )
    buffers = create_live_frame_buffers(["camera_a"])
    with pytest.raises(ValueError, match="reference_camera"):
        LiveRigFrameMatcher(
            buffers,
            reference_camera="camera_b",
            maximum_skew_ms=5.0,
        )
