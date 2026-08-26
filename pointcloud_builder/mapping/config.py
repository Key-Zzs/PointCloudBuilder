"""Strict parser for the versioned fixed-camera TSDF configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numbers
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(frozen=True)
class TsdfBackendConfig:
    type: Literal["open3d_tensor"] = "open3d_tensor"
    device: str = "CPU:0"


@dataclass(frozen=True)
class TsdfVolumeConfig:
    voxel_size_m: float = 0.005
    block_resolution: int = 16
    block_count: int = 50_000
    trunc_voxel_multiplier: float = 4.0


@dataclass(frozen=True)
class TsdfDepthConfig:
    minimum_m: float = 0.10
    maximum_m: float = 2.00


@dataclass(frozen=True)
class TsdfIntegrationConfig:
    source: Literal["native", "ffs_stereo"] = "ffs_stereo"
    frame_stride: int = 5
    maximum_weight: None = None
    queue_capacity: int = 2
    maximum_update_hz: float = 5.0
    maximum_mesh_hz: float = 1.0


@dataclass(frozen=True)
class TsdfExtractionConfig:
    weight_threshold: float = 3.0
    point_cloud: bool = True
    triangle_mesh: bool = True


@dataclass(frozen=True)
class TsdfDynamicConfig:
    mode: Literal["build_static", "frozen_static", "guarded_continuous"] = (
        "frozen_static"
    )
    residual_threshold_m: float = 0.020
    persistence_frames: int = 10
    consistency_tolerance_m: float = 0.010
    integrate_background_consistent: bool = True
    integrate_persistent_new_surface: bool = True


@dataclass(frozen=True)
class TsdfMapConfig:
    schema_version: str
    backend: TsdfBackendConfig
    volume: TsdfVolumeConfig
    depth: TsdfDepthConfig
    integration: TsdfIntegrationConfig
    extraction: TsdfExtractionConfig
    dynamic: TsdfDynamicConfig

    def __post_init__(self) -> None:
        if self.schema_version != "pointcloud-builder.tsdf.v1":
            raise ValueError("unsupported TSDF schema_version")
        if (
            self.backend.type != "open3d_tensor"
            or not isinstance(self.backend.device, str)
            or not self.backend.device.strip()
        ):
            raise ValueError("TSDF backend must be open3d_tensor with a device")
        _positive_real(self.volume.voxel_size_m, "TSDF voxel_size_m")
        _positive_integer(self.volume.block_resolution, "TSDF block_resolution")
        _positive_integer(self.volume.block_count, "TSDF block_count")
        _positive_real(
            self.volume.trunc_voxel_multiplier, "TSDF trunc_voxel_multiplier"
        )
        _positive_real(self.depth.minimum_m, "TSDF minimum_m")
        _positive_real(self.depth.maximum_m, "TSDF maximum_m")
        if self.depth.minimum_m >= self.depth.maximum_m:
            raise ValueError("TSDF depth range must be positive and ordered")
        if self.integration.source not in {"native", "ffs_stereo"}:
            raise ValueError("unsupported TSDF integration source")
        _positive_integer(self.integration.frame_stride, "TSDF frame_stride")
        if self.integration.maximum_weight is not None:
            raise ValueError("Open3D tensor backend does not support maximum_weight")
        if (
            isinstance(self.integration.queue_capacity, bool)
            or not isinstance(self.integration.queue_capacity, numbers.Integral)
            or self.integration.queue_capacity not in {1, 2}
        ):
            raise ValueError("TSDF mapper queue capacity must be one or two")
        _positive_real(self.integration.maximum_update_hz, "TSDF maximum_update_hz")
        _positive_real(self.integration.maximum_mesh_hz, "TSDF maximum_mesh_hz")
        if self.integration.maximum_mesh_hz > self.integration.maximum_update_hz:
            raise ValueError(
                "TSDF maximum_mesh_hz must be positive and no faster than updates"
            )
        _positive_real(self.extraction.weight_threshold, "TSDF weight_threshold")
        if not isinstance(self.extraction.point_cloud, bool) or not isinstance(
            self.extraction.triangle_mesh, bool
        ):
            raise ValueError("TSDF extraction switches must be booleans")
        if not (self.extraction.point_cloud or self.extraction.triangle_mesh):
            raise ValueError("TSDF extraction must enable point cloud or triangle mesh")
        if self.dynamic.mode not in {
            "build_static",
            "frozen_static",
            "guarded_continuous",
        }:
            raise ValueError("unsupported TSDF dynamic mode")
        _positive_real(
            self.dynamic.residual_threshold_m, "dynamic residual_threshold_m"
        )
        _positive_integer(self.dynamic.persistence_frames, "dynamic persistence_frames")
        _positive_real(
            self.dynamic.consistency_tolerance_m,
            "dynamic consistency_tolerance_m",
        )
        if not isinstance(
            self.dynamic.integrate_background_consistent, bool
        ) or not isinstance(self.dynamic.integrate_persistent_new_surface, bool):
            raise ValueError("dynamic integration switches must be booleans")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def estimated_attribute_bytes(self) -> int:
        # Open3D geometry-only integration still requires a dummy UInt16x3 color
        # attribute: Float32 TSDF + UInt16 weight + UInt16x3 color = 12 B/voxel.
        voxels = self.volume.block_resolution**3 * self.volume.block_count
        return voxels * 12


def load_tsdf_config(path: str | Path) -> TsdfMapConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("TSDF config must be a YAML mapping")
    return parse_tsdf_config(raw)


def parse_tsdf_config(raw: dict[str, Any]) -> TsdfMapConfig:
    _exact_keys(
        raw,
        {
            "schema_version",
            "backend",
            "volume",
            "depth",
            "integration",
            "extraction",
            "dynamic",
        },
        "TSDF config",
    )
    return TsdfMapConfig(
        schema_version=str(raw.get("schema_version", "")),
        backend=_section(TsdfBackendConfig, raw, "backend"),
        volume=_section(TsdfVolumeConfig, raw, "volume"),
        depth=_section(TsdfDepthConfig, raw, "depth"),
        integration=_section(TsdfIntegrationConfig, raw, "integration"),
        extraction=_section(TsdfExtractionConfig, raw, "extraction"),
        dynamic=_section(TsdfDynamicConfig, raw, "dynamic"),
    )


def _section(cls: type[Any], raw: dict[str, Any], name: str) -> Any:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"TSDF {name} must be a mapping")
    fields = set(cls.__dataclass_fields__)
    _exact_keys(value, fields, f"TSDF {name}")
    return cls(**value)


def _exact_keys(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} has unknown fields: {unknown}")


def _positive_real(value: object, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{label} must be a finite positive number")


def _positive_integer(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
