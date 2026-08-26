"""Convert Open3D tensor geometry to backend-neutral map contracts."""

from __future__ import annotations

import time

import numpy as np

from pointcloud_builder.mapping.types import MapExtraction


def extract_geometry(volume: object, *, weight_threshold: float) -> MapExtraction:
    started = time.perf_counter()
    if int(volume.hashmap().size()) == 0:  # type: ignore[attr-defined]
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_triangles = np.empty((0, 3), dtype=np.int64)
        return MapExtraction(
            points=empty_points,
            vertices=empty_points.copy(),
            triangles=empty_triangles,
            point_count=0,
            vertex_count=0,
            triangle_count=0,
            extraction_ms=(time.perf_counter() - started) * 1000.0,
        )
    point_cloud = volume.extract_point_cloud(weight_threshold=weight_threshold)  # type: ignore[attr-defined]
    mesh = volume.extract_triangle_mesh(weight_threshold=weight_threshold)  # type: ignore[attr-defined]
    points = _tensor_map_array(point_cloud.point, "positions")
    vertices = _tensor_map_array(mesh.vertex, "positions")
    triangles = _tensor_map_array(mesh.triangle, "indices").astype(np.int64, copy=False)
    return MapExtraction(
        points=np.ascontiguousarray(points),
        vertices=np.ascontiguousarray(vertices),
        triangles=np.ascontiguousarray(triangles),
        point_count=int(len(points)),
        vertex_count=int(len(vertices)),
        triangle_count=int(len(triangles)),
        extraction_ms=(time.perf_counter() - started) * 1000.0,
    )


def _tensor_map_array(tensor_map: object, name: str) -> np.ndarray:
    if name not in tensor_map:  # type: ignore[operator]
        dtype = np.int64 if name == "indices" else np.float32
        return np.empty((0, 3), dtype=dtype)
    tensor = tensor_map[name]  # type: ignore[index]
    return np.asarray(tensor.cpu().numpy())
