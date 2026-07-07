"""YAML configuration parsing for PointCloudBuilder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

SamplingMode = Literal[
    "fps",
    "stride",
    "random",
    "voxel",
    "voxel_random",
    "voxel_fps",
]


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics."""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class CameraConfig:
    """Camera model and depth interpretation settings."""

    width: int
    height: int
    depth_scale: float
    aligned_depth_to_color: bool
    intrinsics: CameraIntrinsics


@dataclass(frozen=True)
class CropConfig:
    """Axis-aligned crop bounds in camera coordinates."""

    enabled: bool
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]


@dataclass(frozen=True)
class SamplingConfig:
    """Fixed-size point sampling settings."""

    mode: SamplingMode
    num_points: int
    stride: int = 1
    voxel_size: float = 0.01


@dataclass(frozen=True)
class PointCloudBuilderConfig:
    """Top-level runtime configuration."""

    camera: CameraConfig
    crop: CropConfig
    sampling: SamplingConfig
    device: str = "auto"


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
    crop_raw = _require_mapping(raw, "crop")
    sampling_raw = _require_mapping(raw, "sampling")
    intrinsics_raw = _require_mapping(camera_raw, "intrinsics")

    intrinsics = CameraIntrinsics(
        fx=float(_require_value(intrinsics_raw, "fx")),
        fy=float(_require_value(intrinsics_raw, "fy")),
        cx=float(_require_value(intrinsics_raw, "cx")),
        cy=float(_require_value(intrinsics_raw, "cy")),
    )
    camera = CameraConfig(
        width=int(_require_value(camera_raw, "width")),
        height=int(_require_value(camera_raw, "height")),
        depth_scale=float(camera_raw.get("depth_scale", 0.001)),
        aligned_depth_to_color=bool(camera_raw.get("aligned_depth_to_color", False)),
        intrinsics=intrinsics,
    )
    crop = CropConfig(
        enabled=bool(crop_raw.get("enabled", True)),
        x=_parse_range(crop_raw.get("x", [-float("inf"), float("inf")]), "crop.x"),
        y=_parse_range(crop_raw.get("y", [-float("inf"), float("inf")]), "crop.y"),
        z=_parse_range(crop_raw.get("z", [0.0, float("inf")]), "crop.z"),
    )
    mode = str(sampling_raw.get("mode", "voxel_random")).lower()
    if mode not in {"fps", "stride", "random", "voxel", "voxel_random", "voxel_fps"}:
        raise ValueError(f"Unsupported sampling mode: {mode}")
    sampling = SamplingConfig(
        mode=mode,  # type: ignore[arg-type]
        num_points=int(sampling_raw.get("num_points", 1024)),
        stride=max(1, int(sampling_raw.get("stride", 1))),
        voxel_size=float(sampling_raw.get("voxel_size", 0.01)),
    )
    if sampling.num_points <= 0:
        raise ValueError("sampling.num_points must be positive")
    if sampling.voxel_size <= 0.0:
        raise ValueError("sampling.voxel_size must be positive")
    return PointCloudBuilderConfig(
        camera=camera,
        crop=crop,
        sampling=sampling,
        device=str(raw.get("device", "auto")),
    )


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
