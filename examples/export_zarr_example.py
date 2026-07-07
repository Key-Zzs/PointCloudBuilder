"""Placeholder for future zarr export integration."""

from __future__ import annotations

from pointcloud_builder import PointCloudBuilder


def convert_frame(builder: PointCloudBuilder, frame: dict[str, object]) -> tuple[object, dict[str, object]]:
    """Convert one recorded frame before writing it into a zarr dataset."""

    return builder.from_recorded_frame(frame)
