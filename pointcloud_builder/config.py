"""YAML configuration parsing for PointCloudBuilder."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from pointcloud_builder.camera_model import CameraIntrinsics

SamplingMode = Literal[
    "fps",
    "stride",
    "random",
    "voxel",
    "voxel_random",
    "voxel_fps",
]
PadMode = Literal["repeat", "zero"]


@dataclass(frozen=True)
class CameraConfig:
    """Camera model and depth interpretation settings."""

    name: str
    depth_scale: float
    aligned_depth_to_color: bool
    color_intrinsics: CameraIntrinsics
    depth_intrinsics: CameraIntrinsics


@dataclass(frozen=True)
class PointCloudConfig:
    """Raw point-cloud output settings."""

    use_rgb: bool
    output_format: str


@dataclass(frozen=True)
class CropConfig:
    """Axis-aligned crop bounds in camera coordinates."""

    enabled: bool
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]
    frame: str = "camera"


@dataclass(frozen=True)
class SamplingConfig:
    """Fixed-size point sampling settings."""

    mode: SamplingMode
    num_points: int
    enabled: bool = True
    stride: int = 1
    voxel_size: float = 0.01
    seed: int | None = None
    deterministic: bool = False
    pad_mode: PadMode = "repeat"


@dataclass(frozen=True)
class PointCloudBuilderConfig:
    """Top-level runtime configuration."""

    camera: CameraConfig
    pointcloud: PointCloudConfig
    device: str = "auto"
    crop: CropConfig = field(
        default_factory=lambda: CropConfig(
            enabled=False,
            x=(-float("inf"), float("inf")),
            y=(-float("inf"), float("inf")),
            z=(0.0, float("inf")),
            frame="camera",
        )
    )
    sampling: SamplingConfig = field(
        default_factory=lambda: SamplingConfig(mode="voxel_random", num_points=1024)
    )


def load_config(path: str | Path) -> PointCloudBuilderConfig:
    """Load a builder configuration from YAML."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> PointCloudBuilderConfig:
    """Parse a raw YAML mapping into typed dataclasses."""

    camera_raw = _require_mapping(raw, "camera")
    pointcloud_raw = _require_mapping(raw, "pointcloud")

    camera = CameraConfig(
        name=str(camera_raw.get("name", "camera")),
        depth_scale=float(camera_raw.get("depth_scale", 0.001)),
        aligned_depth_to_color=bool(camera_raw.get("aligned_depth_to_color", False)),
        color_intrinsics=_parse_intrinsics(
            _require_mapping(camera_raw, "color_intrinsics"),
            "camera.color_intrinsics",
        ),
        depth_intrinsics=_parse_intrinsics(
            _require_mapping(camera_raw, "depth_intrinsics"),
            "camera.depth_intrinsics",
        ),
    )
    pointcloud = PointCloudConfig(
        use_rgb=bool(pointcloud_raw.get("use_rgb", False)),
        output_format=str(pointcloud_raw.get("output_format", "xyz")).lower(),
    )
    if pointcloud.output_format not in {"xyz", "xyzrgb"}:
        raise ValueError("pointcloud.output_format must be 'xyz' or 'xyzrgb'")
    crop = _parse_crop(raw.get("crop"))
    sampling = _parse_sampling(raw.get("sampling"))
    return PointCloudBuilderConfig(
        camera=camera,
        pointcloud=pointcloud,
        device=str(raw.get("device", "auto")),
        crop=crop,
        sampling=sampling,
    )


def _parse_intrinsics(raw: dict[str, Any], name: str) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=int(_require_value(raw, "width")),
        height=int(_require_value(raw, "height")),
        fx=float(_require_value(raw, "fx")),
        fy=float(_require_value(raw, "fy")),
        cx=float(_require_value(raw, "cx")),
        cy=float(_require_value(raw, "cy")),
    )


def _parse_crop(value: Any) -> CropConfig:
    if value is None:
        return CropConfig(
            enabled=False,
            x=(-float("inf"), float("inf")),
            y=(-float("inf"), float("inf")),
            z=(0.0, float("inf")),
            frame="camera",
        )
    if not isinstance(value, dict):
        raise ValueError("crop must be a mapping when provided")
    return CropConfig(
        enabled=bool(value.get("enabled", True)),
        x=_parse_range(value.get("x", [-float("inf"), float("inf")]), "crop.x"),
        y=_parse_range(value.get("y", [-float("inf"), float("inf")]), "crop.y"),
        z=_parse_range(value.get("z", [0.0, float("inf")]), "crop.z"),
        frame=str(value.get("frame", "camera")),
    )


def _parse_sampling(value: Any) -> SamplingConfig:
    if value is None:
        return SamplingConfig(mode="voxel_random", num_points=1024)
    if not isinstance(value, dict):
        raise ValueError("sampling must be a mapping when provided")
    mode = str(value.get("mode", "voxel_random")).lower()
    if mode not in {"fps", "stride", "random", "voxel", "voxel_random", "voxel_fps"}:
        raise ValueError(f"Unsupported sampling mode: {mode}")
    pad_mode = str(value.get("pad_mode", "repeat")).lower()
    if pad_mode not in {"repeat", "zero"}:
        raise ValueError("sampling.pad_mode must be 'repeat' or 'zero'")
    seed_value = value.get("seed")
    sampling = SamplingConfig(
        mode=mode,  # type: ignore[arg-type]
        enabled=bool(value.get("enabled", True)),
        num_points=int(value.get("num_points", 1024)),
        stride=max(1, int(value.get("stride", 1))),
        voxel_size=float(value.get("voxel_size", 0.01)),
        seed=int(seed_value) if seed_value is not None else None,
        deterministic=bool(value.get("deterministic", False)),
        pad_mode=pad_mode,  # type: ignore[arg-type]
    )
    if sampling.num_points <= 0:
        raise ValueError("sampling.num_points must be positive")
    if sampling.voxel_size <= 0.0:
        raise ValueError("sampling.voxel_size must be positive")
    return sampling


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid mapping: {key}")
    return value


def _require_value(raw: dict[str, Any], key: str) -> Any:
    if key not in raw:
        raise ValueError(f"Missing required config value: {key}")
    return raw[key]


def _parse_range(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element range")
    lower = float(value[0])
    upper = float(value[1])
    if lower > upper:
        raise ValueError(f"{name} lower bound is greater than upper bound")
    return lower, upper
