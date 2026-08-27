"""Strict parser for the versioned fixed-camera TSDF configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import numbers
from pathlib import Path
from typing import Any, Literal

import yaml

from pointcloud_builder.config import CropConfig, SamplingConfig


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


def _disabled_workspace_crop() -> CropConfig:
    return CropConfig(
        enabled=False,
        x=(-float("inf"), float("inf")),
        y=(-float("inf"), float("inf")),
        z=(-float("inf"), float("inf")),
        frame="workspace",
    )


def _disabled_sampling() -> SamplingConfig:
    return SamplingConfig(
        enabled=False,
        mode="voxel_fps",
        num_points=4096,
        voxel_size=0.005,
        seed=42,
        deterministic=True,
        pad_mode="repeat",
    )


@dataclass(frozen=True)
class TsdfPostprocessConfig:
    """Optional point-cloud-only processing after TSDF extraction."""

    crop: CropConfig = field(default_factory=_disabled_workspace_crop)
    sampling: SamplingConfig = field(default_factory=_disabled_sampling)


@dataclass(frozen=True)
class TsdfMapConfig:
    schema_version: str
    backend: TsdfBackendConfig
    volume: TsdfVolumeConfig
    depth: TsdfDepthConfig
    integration: TsdfIntegrationConfig
    extraction: TsdfExtractionConfig
    dynamic: TsdfDynamicConfig
    postprocess: TsdfPostprocessConfig = field(default_factory=TsdfPostprocessConfig)

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
        if self.postprocess.crop.frame != "workspace":
            raise ValueError("TSDF postprocess crop frame must be workspace")
        if not isinstance(self.postprocess.crop.enabled, bool):
            raise ValueError("TSDF postprocess crop enabled must be boolean")
        for axis in ("x", "y", "z"):
            bounds = getattr(self.postprocess.crop, axis)
            if (
                not isinstance(bounds, tuple)
                or len(bounds) != 2
                or any(isinstance(value, bool) for value in bounds)
                or not all(isinstance(value, numbers.Real) for value in bounds)
                or math.isnan(float(bounds[0]))
                or math.isnan(float(bounds[1]))
                or bounds[0] > bounds[1]
            ):
                raise ValueError(
                    f"TSDF postprocess crop {axis} must be an ordered numeric range"
                )
        sampling = self.postprocess.sampling
        if not isinstance(sampling.enabled, bool):
            raise ValueError("TSDF postprocess sampling enabled must be boolean")
        if sampling.mode not in {
            "fps",
            "stride",
            "random",
            "voxel",
            "voxel_random",
            "voxel_fps",
        }:
            raise ValueError("unsupported TSDF postprocess sampling mode")
        _positive_integer(sampling.num_points, "TSDF postprocess num_points")
        _positive_integer(sampling.stride, "TSDF postprocess stride")
        _positive_real(sampling.voxel_size, "TSDF postprocess voxel_size")
        if sampling.seed is not None and (
            isinstance(sampling.seed, bool)
            or not isinstance(sampling.seed, numbers.Integral)
        ):
            raise ValueError("TSDF postprocess seed must be an integer or null")
        if not isinstance(sampling.deterministic, bool):
            raise ValueError("TSDF postprocess deterministic must be boolean")
        if sampling.pad_mode not in {"repeat", "zero"}:
            raise ValueError("TSDF postprocess pad_mode must be repeat or zero")

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
            "postprocess",
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
        postprocess=_parse_postprocess(raw.get("postprocess")),
    )


def _parse_postprocess(value: Any) -> TsdfPostprocessConfig:
    if value is None:
        return TsdfPostprocessConfig()
    if not isinstance(value, dict):
        raise ValueError("TSDF postprocess must be a mapping")
    _exact_keys(value, {"crop", "sampling"}, "TSDF postprocess")
    crop_value = value.get("crop", {})
    sampling_value = value.get("sampling", {})
    if not isinstance(crop_value, dict) or not isinstance(sampling_value, dict):
        raise ValueError("TSDF postprocess crop and sampling must be mappings")
    _exact_keys(crop_value, {"enabled", "frame", "x", "y", "z"}, "TSDF postprocess crop")
    _exact_keys(
        sampling_value,
        {
            "enabled",
            "mode",
            "num_points",
            "stride",
            "voxel_size",
            "seed",
            "deterministic",
            "pad_mode",
        },
        "TSDF postprocess sampling",
    )
    crop_default = _disabled_workspace_crop()
    sampling_default = _disabled_sampling()
    crop = CropConfig(
        enabled=_strict_bool(crop_value.get("enabled", False), "crop.enabled"),
        frame=str(crop_value.get("frame", crop_default.frame)),
        x=_range(crop_value.get("x", crop_default.x), "crop.x"),
        y=_range(crop_value.get("y", crop_default.y), "crop.y"),
        z=_range(crop_value.get("z", crop_default.z), "crop.z"),
    )
    mode = str(sampling_value.get("mode", sampling_default.mode)).lower()
    pad_mode = str(
        sampling_value.get("pad_mode", sampling_default.pad_mode)
    ).lower()
    seed = sampling_value.get("seed", sampling_default.seed)
    sampling = SamplingConfig(
        enabled=_strict_bool(
            sampling_value.get("enabled", False), "sampling.enabled"
        ),
        mode=mode,  # type: ignore[arg-type]
        num_points=sampling_value.get("num_points", sampling_default.num_points),
        stride=sampling_value.get("stride", sampling_default.stride),
        voxel_size=sampling_value.get("voxel_size", sampling_default.voxel_size),
        seed=seed,
        deterministic=_strict_bool(
            sampling_value.get("deterministic", sampling_default.deterministic),
            "sampling.deterministic",
        ),
        pad_mode=pad_mode,  # type: ignore[arg-type]
    )
    return TsdfPostprocessConfig(crop=crop, sampling=sampling)


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"TSDF postprocess {label} must be boolean")
    return value


def _range(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"TSDF postprocess {label} must be a two-element range")
    if any(isinstance(item, bool) or not isinstance(item, numbers.Real) for item in value):
        raise ValueError(f"TSDF postprocess {label} must be numeric")
    return float(value[0]), float(value[1])


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
