"""Bounded conversion from rig results to privacy-minimal visualization packets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from pointcloud_builder.visualization.rerun.packet import (
    CameraVisualization,
    MapVisualization,
    PinholeVisualization,
    TriangleMeshVisualization,
    VisualizationPacket,
)


def bounded_cloud(points: torch.Tensor, budget: int) -> np.ndarray:
    """Select on the source device, then transfer no more than ``budget`` points."""

    if (
        not isinstance(points, torch.Tensor)
        or points.ndim != 2
        or points.shape[1] not in {3, 6}
    ):
        raise TypeError("visualization cloud must be an Nx3 or Nx6 torch.Tensor")
    if budget <= 0:
        raise ValueError("viewer point budget must be positive")
    count = int(points.shape[0])
    selected = points.detach()
    if count > budget:
        indices = (
            torch.linspace(0, count - 1, budget, device=points.device).round().long()
        )
        selected = selected.index_select(0, indices)
    return np.ascontiguousarray(selected.to(device="cpu").numpy())


def bounded_rgb_preview(
    image: object, maximum_width: int = 320
) -> tuple[np.ndarray, int]:
    """Create a bounded RGB preview and return its integer pixel stride."""

    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3:
        raise TypeError("color preview source must be an HxWx3 uint8 NumPy array")
    if image.shape[2] not in {3, 4}:
        raise ValueError("color preview source must have three or four channels")
    if maximum_width <= 0:
        raise ValueError("maximum preview width must be positive")
    stride = max(1, int(np.ceil(image.shape[1] / maximum_width)))
    return np.ascontiguousarray(image[::stride, ::stride, :3]), stride


def packet_from_rig_result(
    result: Any,
    runtimes: Mapping[str, Any],
    *,
    point_budget: int = 30_000,
    metrics: Mapping[str, float] | None = None,
) -> VisualizationPacket:
    """Extract only bounded CPU visualization data from one processed matched set."""

    frame_set = result.frame_match
    if frame_set.match_sequence_index is None or frame_set.match_timestamp_ns is None:
        raise ValueError(
            "visualization requires matched-set sequence and host timestamp"
        )
    camera_clouds = {
        item.camera_name: item.cloud.points for item in result.per_camera_workspace
    }
    expected = tuple(sorted(frame_set.envelopes))
    if tuple(sorted(runtimes)) != expected or tuple(sorted(camera_clouds)) != expected:
        raise ValueError(
            "visualization runtimes, frames, and clouds must have identical cameras"
        )
    cameras = []
    for name in expected:
        runtime = runtimes[name]
        context = runtime.pipeline.context
        color_frame = context.calibration.intrinsic_frames["color"]
        T_color_from_source = context.calibration.transform(
            context.source_frame, color_frame
        ).matrix
        T_workspace_from_color = (
            context.T_workspace_from_source.matrix @ np.linalg.inv(T_color_from_source)
        )
        intrinsics = context.calibration.intrinsics["color"]
        frame = frame_set.envelopes[name].frame
        color = frame.streams.get("color")
        if color is None:
            raise ValueError(f"matched camera {name!r} has no color preview stream")
        preview, preview_stride = bounded_rgb_preview(color.data)
        cameras.append(
            CameraVisualization(
                camera_name=name,
                rgb_preview=preview,
                workspace_cloud=bounded_cloud(camera_clouds[name], point_budget),
                T_workspace_from_color=T_workspace_from_color,
                color_intrinsics=PinholeVisualization(
                    width=int(preview.shape[1]),
                    height=int(preview.shape[0]),
                    fx=intrinsics.fx / preview_stride,
                    fy=intrinsics.fy / preview_stride,
                    cx=intrinsics.cx / preview_stride,
                    cy=intrinsics.cy / preview_stride,
                ),
            )
        )
    return VisualizationPacket(
        matched_set_index=frame_set.match_sequence_index,
        host_time_seconds=frame_set.match_timestamp_ns / 1_000_000_000.0,
        cameras=tuple(cameras),
        concatenated_cloud=bounded_cloud(result.concatenated.points, point_budget),
        fused_cloud=bounded_cloud(result.fused.points, point_budget),
        sampled_cloud=bounded_cloud(result.sampled.points, point_budget),
        metrics=dict(metrics or {}),
    )


def map_visualization_from_snapshot(
    snapshot: Any,
    dynamic_overlay: torch.Tensor,
    *,
    point_budget: int = 30_000,
    reset: bool = False,
    include_static: bool = True,
) -> MapVisualization:
    """Bound mapper geometry and the current cloud for one Rerun packet."""

    extraction = snapshot.extraction
    points = None
    mesh = None
    if include_static:
        points = _bounded_numpy_cloud(extraction.points, point_budget)
        mesh = _bounded_mesh(
            extraction.vertices,
            extraction.triangles,
            triangle_budget=point_budget,
        )
    raycast_depth = snapshot.raycast_depths[0][1] if snapshot.raycast_depths else None
    dynamic_mask = snapshot.dynamic_masks[0][1] if snapshot.dynamic_masks else None
    return MapVisualization(
        tsdf_points=None if points is None else np.ascontiguousarray(points),
        tsdf_points_raw=(
            None
            if not include_static
            else _bounded_numpy_cloud(extraction.raw_points, point_budget)
        ),
        tsdf_points_cropped=(
            None
            if not include_static
            else _bounded_numpy_cloud(extraction.cropped_points, point_budget)
        ),
        tsdf_points_sampled=(
            None
            if not include_static
            else _bounded_numpy_cloud(extraction.sampled_points, point_budget)
        ),
        tsdf_mesh=mesh,
        dynamic_overlay=bounded_cloud(dynamic_overlay, point_budget),
        raycast_depth=raycast_depth,
        dynamic_mask=dynamic_mask,
        static_revision=snapshot.map_state.map_revision,
        frozen=snapshot.map_state.lifecycle == "frozen",
        reset=reset,
    )


def _bounded_numpy_cloud(points: np.ndarray, point_budget: int) -> np.ndarray:
    selected = points
    if len(selected) > point_budget:
        indices = np.linspace(0, len(selected) - 1, point_budget).round().astype(int)
        selected = selected[indices]
    return np.ascontiguousarray(selected)


def _bounded_mesh(
    vertices: np.ndarray, triangles: np.ndarray, *, triangle_budget: int
) -> TriangleMeshVisualization:
    """Bound mesh packet size while retaining valid triangle indices."""

    if triangle_budget <= 0:
        raise ValueError("mesh triangle budget must be positive")
    selected = triangles
    if len(selected) > triangle_budget:
        indices = np.linspace(0, len(selected) - 1, triangle_budget).round().astype(int)
        selected = selected[indices]
    if len(selected):
        used = np.unique(selected.reshape(-1))
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[used] = np.arange(len(used), dtype=np.int64)
        return TriangleMeshVisualization(
            vertices=np.ascontiguousarray(vertices[used]),
            triangles=np.ascontiguousarray(remap[selected]),
        )
    bounded_vertices = vertices[:triangle_budget]
    return TriangleMeshVisualization(
        vertices=np.ascontiguousarray(bounded_vertices),
        triangles=np.empty((0, 3), dtype=np.int64),
    )
