"""Strict CPU-only packet contracts crossing the visualization process boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
import numbers

import numpy as np


def _array(
    value: object,
    name: str,
    *,
    ndim: int,
    last_dimensions: tuple[int, ...] | None = None,
    allow_bool: bool = False,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    result = np.ascontiguousarray(value).copy()
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if last_dimensions is not None and result.shape[-1] not in last_dimensions:
        raise ValueError(f"{name} has an unsupported final dimension")
    if result.dtype == np.bool_:
        if not allow_bool:
            raise ValueError(f"{name} must be numeric")
    elif not np.issubdtype(result.dtype, np.number) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite numeric values")
    result.setflags(write=False)
    return result


def _rigid_transform(value: object, name: str) -> np.ndarray:
    result = _array(value, name, ndim=2)
    if result.shape != (4, 4) or not np.allclose(
        result[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-9
    ):
        raise ValueError(f"{name} must be a homogeneous 4x4 matrix")
    rotation = result[:3, :3].astype(np.float64, copy=False)
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"{name} rotation must have determinant +1")
    return result


@dataclass(frozen=True)
class PinholeVisualization:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("pinhole resolution must be positive")
        values = np.asarray((self.fx, self.fy, self.cx, self.cy), dtype=np.float64)
        if not np.isfinite(values).all() or self.fx <= 0 or self.fy <= 0:
            raise ValueError(
                "pinhole intrinsics must be finite with positive focal lengths"
            )


@dataclass(frozen=True)
class CameraVisualization:
    camera_name: str
    rgb_preview: np.ndarray
    workspace_cloud: np.ndarray
    T_workspace_from_color: np.ndarray
    color_intrinsics: PinholeVisualization

    def __post_init__(self) -> None:
        if (
            not self.camera_name.strip()
            or "/" in self.camera_name
            or "\\" in self.camera_name
        ):
            raise ValueError("camera_name must be a portable entity segment")
        image = _array(self.rgb_preview, "rgb_preview", ndim=3, last_dimensions=(3, 4))
        if image.dtype != np.uint8:
            raise ValueError("rgb_preview must be uint8")
        cloud = _array(
            self.workspace_cloud,
            "workspace_cloud",
            ndim=2,
            last_dimensions=(3, 6),
        )
        transform = _rigid_transform(
            self.T_workspace_from_color, "T_workspace_from_color"
        )
        object.__setattr__(self, "rgb_preview", image)
        object.__setattr__(self, "workspace_cloud", cloud)
        object.__setattr__(self, "T_workspace_from_color", transform)


@dataclass(frozen=True)
class TriangleMeshVisualization:
    vertices: np.ndarray
    triangles: np.ndarray
    vertex_colors: np.ndarray | None = None

    def __post_init__(self) -> None:
        vertices = _array(self.vertices, "mesh.vertices", ndim=2, last_dimensions=(3,))
        triangles = _array(
            self.triangles, "mesh.triangles", ndim=2, last_dimensions=(3,)
        )
        if not np.issubdtype(triangles.dtype, np.integer):
            raise ValueError("mesh.triangles must be integer indices")
        if triangles.size and (triangles.min() < 0 or triangles.max() >= len(vertices)):
            raise ValueError("mesh.triangles contain out-of-range vertex indices")
        colors = self.vertex_colors
        if colors is not None:
            colors = _array(
                colors, "mesh.vertex_colors", ndim=2, last_dimensions=(3, 4)
            )
            if len(colors) != len(vertices):
                raise ValueError("mesh vertex colors must match vertex count")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "triangles", triangles)
        object.__setattr__(self, "vertex_colors", colors)


@dataclass(frozen=True)
class MapVisualization:
    tsdf_points: np.ndarray | None = None
    tsdf_mesh: TriangleMeshVisualization | None = None
    dynamic_overlay: np.ndarray | None = None
    raycast_depth: np.ndarray | None = None
    dynamic_mask: np.ndarray | None = None
    static_revision: int | None = None
    frozen: bool = False
    reset: bool = False

    def __post_init__(self) -> None:
        if self.static_revision is not None and (
            isinstance(self.static_revision, bool)
            or not isinstance(self.static_revision, numbers.Integral)
            or self.static_revision < 0
        ):
            raise ValueError("static_revision must be a non-negative integer")
        if (
            self.tsdf_points is not None or self.tsdf_mesh is not None
        ) and self.static_revision is None:
            raise ValueError("static TSDF geometry requires a map revision")
        if self.tsdf_points is not None:
            object.__setattr__(
                self,
                "tsdf_points",
                _array(self.tsdf_points, "tsdf_points", ndim=2, last_dimensions=(3, 6)),
            )
        if self.dynamic_overlay is not None:
            object.__setattr__(
                self,
                "dynamic_overlay",
                _array(
                    self.dynamic_overlay,
                    "dynamic_overlay",
                    ndim=2,
                    last_dimensions=(3, 6),
                ),
            )
        if self.raycast_depth is not None:
            object.__setattr__(
                self,
                "raycast_depth",
                _array(self.raycast_depth, "raycast_depth", ndim=2),
            )
        if self.dynamic_mask is not None:
            if (
                not isinstance(self.dynamic_mask, np.ndarray)
                or self.dynamic_mask.dtype != np.bool_
            ):
                raise ValueError("dynamic_mask must be a bool NumPy array")
            object.__setattr__(
                self,
                "dynamic_mask",
                _array(self.dynamic_mask, "dynamic_mask", ndim=2, allow_bool=True),
            )


@dataclass(frozen=True)
class VisualizationPacket:
    matched_set_index: int
    host_time_seconds: float
    cameras: tuple[CameraVisualization, ...]
    concatenated_cloud: np.ndarray
    fused_cloud: np.ndarray
    sampled_cloud: np.ndarray
    metrics: dict[str, float] = field(default_factory=dict)
    map: MapVisualization | None = None
    schema_version: str = "pointcloud-builder.visualization-packet.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "pointcloud-builder.visualization-packet.v1":
            raise ValueError("unsupported visualization packet schema")
        if (
            isinstance(self.matched_set_index, bool)
            or not isinstance(self.matched_set_index, numbers.Integral)
            or self.matched_set_index < 0
        ):
            raise ValueError("matched_set_index must be a non-negative integer")
        if not np.isfinite(self.host_time_seconds) or self.host_time_seconds < 0:
            raise ValueError("host_time_seconds must be finite and non-negative")
        cameras = tuple(self.cameras)
        names = [camera.camera_name for camera in cameras]
        if not cameras or names != sorted(names) or len(set(names)) != len(names):
            raise ValueError(
                "cameras must be non-empty, unique, and canonically ordered"
            )
        metrics: dict[str, float] = {}
        for name, value in self.metrics.items():
            if not name.strip() or "/" in name or "\\" in name:
                raise ValueError("metric names must be portable entity segments")
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError("metrics must contain finite scalars")
            metrics[name] = numeric
        object.__setattr__(self, "cameras", cameras)
        object.__setattr__(
            self,
            "concatenated_cloud",
            _array(
                self.concatenated_cloud,
                "concatenated_cloud",
                ndim=2,
                last_dimensions=(3, 6),
            ),
        )
        object.__setattr__(
            self,
            "fused_cloud",
            _array(self.fused_cloud, "fused_cloud", ndim=2, last_dimensions=(3, 6)),
        )
        object.__setattr__(
            self,
            "sampled_cloud",
            _array(self.sampled_cloud, "sampled_cloud", ndim=2, last_dimensions=(3, 6)),
        )
        object.__setattr__(self, "metrics", metrics)
