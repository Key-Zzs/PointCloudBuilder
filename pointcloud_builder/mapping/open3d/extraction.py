"""Convert Open3D tensor geometry to backend-neutral map contracts."""

from __future__ import annotations

import time

import numpy as np

from pointcloud_builder.mapping.config import TsdfPostprocessConfig
from pointcloud_builder.mapping.postprocess import cloud_numpy, postprocess_extracted_cloud
from pointcloud_builder.mapping.types import MapExtraction


def extract_geometry(
    volume: object,
    *,
    weight_threshold: float,
    workspace_frame: str = "workspace",
    postprocess: TsdfPostprocessConfig | None = None,
    device: str = "CPU:0",
    synchronize: object | None = None,
) -> MapExtraction:
    started = time.perf_counter()
    if int(volume.hashmap().size()) == 0:  # type: ignore[attr-defined]
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_triangles = np.empty((0, 3), dtype=np.int64)
        processed = postprocess_extracted_cloud(
            empty_points,
            workspace_frame=workspace_frame,
            config=postprocess or TsdfPostprocessConfig(),
            device=device,
        )
        return MapExtraction(
            points=empty_points,
            vertices=empty_points.copy(),
            triangles=empty_triangles,
            point_count=0,
            vertex_count=0,
            triangle_count=0,
            extraction_ms=(time.perf_counter() - started) * 1000.0,
            raw_point_cloud=cloud_numpy(processed.raw),
            cropped_point_cloud=cloud_numpy(processed.cropped),
            sampled_point_cloud=cloud_numpy(processed.sampled),
            post_crop_ms=processed.crop_ms,
            post_sampling_ms=processed.sampling_ms,
        )
    _sync(synchronize)
    point_started = time.perf_counter()
    point_cloud = volume.extract_point_cloud(weight_threshold=weight_threshold)  # type: ignore[attr-defined]
    _sync(synchronize)
    points = _canonical_point_order(
        _tensor_map_array(point_cloud.point, "positions")
    )
    point_ms = (time.perf_counter() - point_started) * 1000.0
    mesh_started = time.perf_counter()
    mesh = volume.extract_triangle_mesh(weight_threshold=weight_threshold)  # type: ignore[attr-defined]
    _sync(synchronize)
    vertices = _tensor_map_array(mesh.vertex, "positions")
    triangles = _tensor_map_array(mesh.triangle, "indices").astype(
        np.int64, copy=False
    )
    mesh_ms = (time.perf_counter() - mesh_started) * 1000.0
    processed = postprocess_extracted_cloud(
        points,
        workspace_frame=workspace_frame,
        config=postprocess or TsdfPostprocessConfig(),
        device=device,
    )
    cropped_started = time.perf_counter()
    cropped_points = cloud_numpy(processed.cropped)
    cropped_materialization_ms = (time.perf_counter() - cropped_started) * 1000.0
    sampled_started = time.perf_counter()
    sampled_points = cloud_numpy(processed.sampled)
    sampled_materialization_ms = (time.perf_counter() - sampled_started) * 1000.0
    return MapExtraction(
        points=np.ascontiguousarray(points),
        vertices=np.ascontiguousarray(vertices),
        triangles=np.ascontiguousarray(triangles),
        point_count=int(len(points)),
        vertex_count=int(len(vertices)),
        triangle_count=int(len(triangles)),
        extraction_ms=(time.perf_counter() - started) * 1000.0,
        raw_point_cloud=np.ascontiguousarray(points),
        cropped_point_cloud=cropped_points,
        sampled_point_cloud=sampled_points,
        extract_point_cloud_ms=point_ms,
        extract_mesh_ms=mesh_ms,
        post_crop_ms=processed.crop_ms + cropped_materialization_ms,
        post_sampling_ms=processed.sampling_ms + sampled_materialization_ms,
    )


def _sync(callback: object | None) -> None:
    if callable(callback):
        callback()


def _tensor_map_array(tensor_map: object, name: str) -> np.ndarray:
    if name not in tensor_map:  # type: ignore[operator]
        dtype = np.int64 if name == "indices" else np.float32
        return np.empty((0, 3), dtype=dtype)
    tensor = tensor_map[name]  # type: ignore[index]
    return np.asarray(tensor.cpu().numpy())


def _canonical_point_order(points: np.ndarray) -> np.ndarray:
    """Remove backend hash-table iteration order from deterministic postprocessing."""

    if not len(points):
        return np.ascontiguousarray(points)
    indices = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    return np.ascontiguousarray(points[indices])
