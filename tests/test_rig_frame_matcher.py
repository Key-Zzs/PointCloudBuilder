from __future__ import annotations

import pytest

from pointcloud_builder.rig import SyntheticCameraSource, create_synthetic_scene
from pointcloud_builder.rig.frame_matcher import (
    match_exact_index,
    match_nearest_host_timestamp,
)


def test_exact_index_rejects_frame_count_mismatch() -> None:
    scene = create_synthetic_scene(("camera_a", "camera_b"), frame_count=2)
    sources = {
        "camera_a": SyntheticCameraSource("camera_a", scene.frames["camera_a"]),
        "camera_b": SyntheticCameraSource("camera_b", scene.frames["camera_b"][:1]),
    }
    with pytest.raises(ValueError, match="frame mismatch"):
        match_exact_index(sources, 1, reference_camera="camera_a")


def test_nearest_host_timestamp_uses_host_clock_and_reports_skew() -> None:
    scene = create_synthetic_scene(
        ("camera_a", "camera_b", "camera_c"),
        timestamp_offsets_ns={"camera_b": 4_000_000, "camera_c": -3_000_000},
    )
    sources = {
        name: SyntheticCameraSource(name, scene.frames[name]) for name in scene.frames
    }
    matched = match_nearest_host_timestamp(
        sources,
        1,
        reference_camera="camera_a",
        maximum_skew_ms=5.0,
    )
    assert matched.reference_camera == "camera_a"
    assert matched.per_camera_delta_ms == {
        "camera_a": 0.0,
        "camera_b": 4.0,
        "camera_c": -3.0,
    }
    assert matched.maximum_skew_ms == 4.0
    assert matched.unmatched_cameras == ()


def test_nearest_host_timestamp_marks_out_of_gate_camera_unmatched() -> None:
    scene = create_synthetic_scene(
        ("camera_a", "camera_b"), timestamp_offsets_ns={"camera_b": 12_000_000}
    )
    sources = {
        name: SyntheticCameraSource(name, scene.frames[name]) for name in scene.frames
    }
    matched = match_nearest_host_timestamp(
        sources, 0, reference_camera="camera_a", maximum_skew_ms=5.0
    )
    assert matched.unmatched_cameras == ("camera_b",)
    assert set(matched.envelopes) == {"camera_a"}
