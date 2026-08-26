"""Adapt one stable CameraRig CameraFrame to PCB's frame mapping contract."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from pointcloud_builder.integrations.camera_rig.dependencies import CameraBundle, CameraFrame
from pointcloud_builder.integrations.camera_rig.validation import (
    require_streams,
    validate_frame_identity,
    validate_passed_fixed_bundle,
)

_STREAM_TO_OUTPUT = {
    "color": "rgb",
    "depth": "depth",
    "ir_left": "left_ir",
    "ir_right": "right_ir",
}


class CameraRigFrameAdapter:
    """Preserve CameraRig pixels and timing metadata without geometric processing."""

    def __init__(
        self,
        bundle: CameraBundle,
        *,
        required_streams: Iterable[str],
        timestamp_stream: str | None = None,
    ) -> None:
        validate_passed_fixed_bundle(bundle)
        self.expected_camera_name = str(bundle.device.camera_name)
        self.expected_serial = str(bundle.device.serial)
        self.required_streams = tuple(required_streams)
        if not self.required_streams:
            raise ValueError("required_streams must not be empty")
        unknown = sorted(set(self.required_streams) - set(_STREAM_TO_OUTPUT))
        if unknown:
            raise ValueError(f"unsupported CameraRig streams: {unknown}")
        self.timestamp_stream = timestamp_stream or self.required_streams[0]
        if self.timestamp_stream not in _STREAM_TO_OUTPUT:
            raise ValueError(f"unsupported timestamp stream: {self.timestamp_stream!r}")

    def adapt(self, frame: CameraFrame) -> dict[str, Any]:
        """Return a mapping with raw arrays and explicit per-stream provenance."""

        validate_frame_identity(
            frame,
            expected_camera_name=self.expected_camera_name,
            expected_serial=self.expected_serial,
        )
        require_streams(frame, self.required_streams)
        output: dict[str, Any] = {name: None for name in _STREAM_TO_OUTPUT.values()}
        frame_numbers: dict[str, int] = {}
        timestamps: dict[str, int | None] = {}
        domains: dict[str, str | None] = {}
        originals: dict[str, float | None] = {}
        for stream_name, output_name in _STREAM_TO_OUTPUT.items():
            stream = frame.streams.get(stream_name)
            if stream is None:
                continue
            _validate_stream_array(stream_name, stream.data)
            output[output_name] = stream.data
            frame_numbers[stream_name] = int(stream.frame_number)
            timestamps[stream_name] = stream.sensor_timestamp_ns
            domains[stream_name] = stream.timestamp_domain
            originals[stream_name] = stream.original_timestamp

        reference = frame.streams.get(self.timestamp_stream)
        timestamp_ns = None if reference is None else reference.sensor_timestamp_ns
        if timestamp_ns is None:
            timestamp_ns = int(frame.host_receive_timestamp_ns)
        output.update(
            {
                "timestamp": float(timestamp_ns) / 1_000_000_000.0,
                "timestamp_ns": int(timestamp_ns),
                "host_receive_timestamp_ns": int(frame.host_receive_timestamp_ns),
                "camera_name": frame.camera_name,
                "frame_numbers": frame_numbers,
                "stream_timestamps_ns": timestamps,
                "stream_timestamp_domains": domains,
                "stream_original_timestamps": originals,
            }
        )
        return output


def _validate_stream_array(stream_name: str, data: np.ndarray) -> None:
    if not isinstance(data, np.ndarray):
        raise TypeError(f"CameraFrame stream {stream_name!r} is not a NumPy array")
    if stream_name == "color":
        if data.ndim != 3 or data.shape[2] != 3 or data.dtype != np.uint8:
            raise ValueError("CameraRig color must be HxWx3 uint8 RGB")
    elif stream_name == "depth":
        if data.ndim != 2 or data.dtype != np.uint16:
            raise ValueError("CameraRig depth must be HxW uint16 raw units")
    elif data.ndim != 2 or data.dtype != np.uint8:
        raise ValueError(f"CameraRig {stream_name} must be HxW uint8")
