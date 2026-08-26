"""Host-clock based offline rig frame matching."""

from __future__ import annotations

from collections.abc import Mapping

from pointcloud_builder.rig.sources import OfflineCameraSource
from pointcloud_builder.rig.types import RigFrameSet


def match_exact_index(
    sources: Mapping[str, OfflineCameraSource], index: int, *, reference_camera: str
) -> RigFrameSet:
    envelopes = {}
    missing = []
    for name in sorted(sources):
        try:
            envelopes[name] = sources[name].envelope(index)
        except IndexError:
            missing.append(name)
    if missing:
        raise ValueError(f"exact_index frame mismatch at {index}: {missing}")
    reference = envelopes[reference_camera].host_receive_timestamp_ns
    deltas = {
        name: (envelope.host_receive_timestamp_ns - reference) / 1_000_000.0
        for name, envelope in envelopes.items()
    }
    return RigFrameSet(
        envelopes=envelopes,
        reference_camera=reference_camera,
        per_camera_delta_ms=deltas,
        maximum_skew_ms=max(abs(value) for value in deltas.values()),
    )


def match_nearest_host_timestamp(
    sources: Mapping[str, OfflineCameraSource],
    reference_index: int,
    *,
    reference_camera: str,
    maximum_skew_ms: float,
) -> RigFrameSet:
    reference = sources[reference_camera].envelope(reference_index)
    envelopes = {reference_camera: reference}
    deltas = {reference_camera: 0.0}
    unmatched: list[str] = []
    for name in sorted(sources):
        if name == reference_camera:
            continue
        candidates = (sources[name].envelope(index) for index in range(sources[name].frame_count))
        closest = min(
            candidates,
            key=lambda item: (abs(item.host_receive_timestamp_ns - reference.host_receive_timestamp_ns), item.frame_index),
        )
        delta = (closest.host_receive_timestamp_ns - reference.host_receive_timestamp_ns) / 1_000_000.0
        deltas[name] = delta
        if abs(delta) > maximum_skew_ms:
            unmatched.append(name)
        else:
            envelopes[name] = closest
    return RigFrameSet(
        envelopes=envelopes,
        reference_camera=reference_camera,
        per_camera_delta_ms=deltas,
        maximum_skew_ms=max(abs(value) for value in deltas.values()),
        unmatched_cameras=tuple(unmatched),
    )
