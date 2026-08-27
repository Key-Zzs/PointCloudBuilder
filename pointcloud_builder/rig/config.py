"""Strict versioned YAML contract owned by the PCB rig layer."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Literal

import yaml

from pointcloud_builder.config import CropConfig, SamplingConfig
from pointcloud_builder.fusion.config import VoxelFusionConfig

RIG_SCHEMA_VERSION = "pointcloud-builder.rig.v1"


@dataclass(frozen=True)
class CameraRigReplaySourceConfig:
    type: Literal["camera_rig_replay"]
    capture_artifact: str
    provision_artifact: str


@dataclass(frozen=True)
class CameraRigLiveSourceConfig:
    type: Literal["camera_rig_live"]
    camera_config: str
    provision_artifact: str


@dataclass(frozen=True)
class SyntheticRigSourceConfig:
    type: Literal["synthetic"]
    capture_artifact: str
    provision_artifact: str


RigSourceConfig = (
    CameraRigReplaySourceConfig | CameraRigLiveSourceConfig | SyntheticRigSourceConfig
)


@dataclass(frozen=True)
class RigDepthConfig:
    mode: Literal["native", "ffs_stereo"]


@dataclass(frozen=True)
class RigPointCloudConfig:
    """Per-camera point features produced before workspace fusion."""

    use_rgb: bool = False


@dataclass(frozen=True)
class RigCameraConfig:
    name: str
    enabled: bool
    source: RigSourceConfig
    depth: RigDepthConfig
    pointcloud: RigPointCloudConfig
    pipeline_config: str | None
    local_crop: CropConfig


@dataclass(frozen=True)
class RigTimingConfig:
    mode: Literal["exact_index", "nearest_host_timestamp"]
    maximum_skew_ms: float
    reference_camera: str | None = None


@dataclass(frozen=True)
class RigLiveConfig:
    buffer_capacity: int = 2
    matcher_wait_timeout_s: float = 0.1
    join_timeout_s: float = 5.0
    telemetry_history_capacity: int = 10_000


@dataclass(frozen=True)
class RigConfig:
    schema_version: str
    output_frame: str
    cameras: tuple[RigCameraConfig, ...]
    timing: RigTimingConfig
    workspace_crop: CropConfig
    fusion: VoxelFusionConfig
    sampling: SamplingConfig
    live: RigLiveConfig | None = None

    @property
    def fusion_enabled(self) -> bool:
        return self.fusion.enabled

    @property
    def enabled_cameras(self) -> tuple[RigCameraConfig, ...]:
        return tuple(camera for camera in self.cameras if camera.enabled)


def load_rig_config(path: str | Path) -> RigConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("rig config must be a YAML mapping")
    return parse_rig_config(raw, base_dir=config_path.parent)


def parse_rig_config(raw: dict[str, Any], *, base_dir: Path | None = None) -> RigConfig:
    _keys(
        raw,
        {
            "schema_version",
            "output_frame",
            "cameras",
            "timing",
            "workspace_crop",
            "fusion",
            "sampling",
            "live",
        },
        "rig",
    )
    if raw.get("schema_version") != RIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {RIG_SCHEMA_VERSION!r}")
    output_frame = _nonempty(raw.get("output_frame"), "output_frame")
    cameras_raw = raw.get("cameras")
    if not isinstance(cameras_raw, list) or not cameras_raw:
        raise ValueError("cameras must be a non-empty list")
    cameras = tuple(_camera(value, output_frame, base_dir) for value in cameras_raw)
    names = [camera.name for camera in cameras]
    if len(names) != len(set(names)):
        raise ValueError("duplicate camera names are not allowed")
    enabled_names = [camera.name for camera in cameras if camera.enabled]
    if not enabled_names:
        raise ValueError("at least one camera must be enabled")
    timing = _timing(raw.get("timing"), enabled_names)
    has_enabled_live_source = any(
        camera.enabled and camera.source.type == "camera_rig_live" for camera in cameras
    )
    if has_enabled_live_source and timing.mode != "nearest_host_timestamp":
        raise ValueError("live rigs require timing.mode=nearest_host_timestamp")
    live = None if raw.get("live") is None else _live(raw["live"])
    if live is not None and not has_enabled_live_source:
        raise ValueError("live section requires an enabled camera_rig_live source")
    workspace_crop = _crop(raw.get("workspace_crop"), output_frame, "workspace_crop")
    fusion = _mapping(raw.get("fusion"), "fusion")
    _keys(fusion, {"enabled", "voxel_size_m", "origin", "deterministic"}, "fusion")
    origin_raw = fusion.get("origin", [0.0, 0.0, 0.0])
    if not isinstance(origin_raw, list) or len(origin_raw) != 3:
        raise ValueError("fusion.origin must be a three-element list")
    fusion_config = VoxelFusionConfig(
        enabled=_boolean(fusion.get("enabled", False), "fusion.enabled"),
        voxel_size_m=float(fusion.get("voxel_size_m", 0.01)),
        origin=tuple(float(value) for value in origin_raw),  # type: ignore[arg-type]
        deterministic=_boolean(
            fusion.get("deterministic", True), "fusion.deterministic"
        ),
    )
    return RigConfig(
        schema_version=RIG_SCHEMA_VERSION,
        output_frame=output_frame,
        cameras=cameras,
        timing=timing,
        workspace_crop=workspace_crop,
        fusion=fusion_config,
        sampling=_sampling(raw.get("sampling")),
        live=live,
    )


def _camera(value: Any, output_frame: str, base_dir: Path | None) -> RigCameraConfig:
    raw = _mapping(value, "cameras[]")
    _keys(
        raw,
        {
            "name",
            "enabled",
            "source",
            "depth",
            "pointcloud",
            "pipeline_config",
            "local_crop",
        },
        "cameras[]",
    )
    name = _nonempty(raw.get("name"), "cameras[].name")
    source = _source(raw.get("source"), name, base_dir)
    depth_raw = _mapping(raw.get("depth"), f"camera {name}.depth")
    _keys(depth_raw, {"mode"}, f"camera {name}.depth")
    depth_mode = str(depth_raw.get("mode", ""))
    if depth_mode not in {"native", "ffs_stereo"}:
        raise ValueError(f"camera {name}.depth.mode is unsupported")
    pipeline_value = raw.get("pipeline_config")
    pipeline_config = (
        None
        if pipeline_value is None
        else _path(pipeline_value, f"camera {name}.pipeline_config", base_dir)
    )
    return RigCameraConfig(
        name=name,
        enabled=_boolean(raw.get("enabled", True), f"camera {name}.enabled"),
        source=source,
        depth=RigDepthConfig(mode=depth_mode),  # type: ignore[arg-type]
        pointcloud=_pointcloud(raw.get("pointcloud"), name),
        pipeline_config=pipeline_config,
        local_crop=_crop(
            raw.get("local_crop"),
            f"{name}/{'depth' if depth_mode == 'native' else 'ir_left'}_optical",
            f"camera {name}.local_crop",
        ),
    )


def _pointcloud(value: Any, camera_name: str) -> RigPointCloudConfig:
    label = f"camera {camera_name}.pointcloud"
    raw = {} if value is None else _mapping(value, label)
    _keys(raw, {"use_rgb"}, label)
    return RigPointCloudConfig(
        use_rgb=_boolean(raw.get("use_rgb", False), f"{label}.use_rgb")
    )


def _source(value: Any, camera_name: str, base_dir: Path | None) -> RigSourceConfig:
    label = f"camera {camera_name}.source"
    raw = _mapping(value, label)
    source_type = str(raw.get("type", ""))
    if source_type == "camera_rig_replay":
        _keys(raw, {"type", "capture_artifact", "provision_artifact"}, label)
        return CameraRigReplaySourceConfig(
            type="camera_rig_replay",
            capture_artifact=_path(
                raw.get("capture_artifact"), f"{label}.capture_artifact", base_dir
            ),
            provision_artifact=_path(
                raw.get("provision_artifact"), f"{label}.provision_artifact", base_dir
            ),
        )
    if source_type == "camera_rig_live":
        _keys(raw, {"type", "camera_config", "provision_artifact"}, label)
        return CameraRigLiveSourceConfig(
            type="camera_rig_live",
            camera_config=_path(
                raw.get("camera_config"), f"{label}.camera_config", base_dir
            ),
            provision_artifact=_path(
                raw.get("provision_artifact"), f"{label}.provision_artifact", base_dir
            ),
        )
    if source_type == "synthetic":
        _keys(raw, {"type", "capture_artifact", "provision_artifact"}, label)
        return SyntheticRigSourceConfig(
            type="synthetic",
            capture_artifact=_path(
                raw.get("capture_artifact"), f"{label}.capture_artifact", base_dir
            ),
            provision_artifact=_path(
                raw.get("provision_artifact"), f"{label}.provision_artifact", base_dir
            ),
        )
    raise ValueError(f"camera {camera_name}.source.type is unsupported")


def _timing(value: Any, enabled_camera_names: list[str]) -> RigTimingConfig:
    raw = _mapping(value, "timing")
    _keys(raw, {"mode", "maximum_skew_ms", "reference_camera"}, "timing")
    mode = str(raw.get("mode", ""))
    if mode not in {"exact_index", "nearest_host_timestamp"}:
        raise ValueError("timing.mode must be exact_index or nearest_host_timestamp")
    maximum_skew_ms = _finite_float(
        raw.get("maximum_skew_ms", 33.4), "timing.maximum_skew_ms"
    )
    if maximum_skew_ms < 0:
        raise ValueError("timing.maximum_skew_ms must be non-negative")
    reference = raw.get("reference_camera")
    if reference is not None and str(reference) not in enabled_camera_names:
        raise ValueError("timing.reference_camera must name an enabled camera")
    return RigTimingConfig(
        mode=mode,
        maximum_skew_ms=maximum_skew_ms,
        reference_camera=None if reference is None else str(reference),
    )  # type: ignore[arg-type]


def _crop(value: Any, frame: str, label: str) -> CropConfig:
    raw = {} if value is None else _mapping(value, label)
    _keys(raw, {"enabled", "x", "y", "z"}, label)
    return CropConfig(
        enabled=_boolean(raw.get("enabled", False), f"{label}.enabled"),
        x=_range(raw.get("x", [-float("inf"), float("inf")]), f"{label}.x"),
        y=_range(raw.get("y", [-float("inf"), float("inf")]), f"{label}.y"),
        z=_range(raw.get("z", [-float("inf"), float("inf")]), f"{label}.z"),
        frame=frame,
    )


def _sampling(value: Any) -> SamplingConfig:
    raw = _mapping(value, "sampling")
    _keys(
        raw,
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
        "sampling",
    )
    return SamplingConfig(
        enabled=_boolean(raw.get("enabled", True), "sampling.enabled"),
        mode=str(raw.get("mode", "voxel_fps")),  # type: ignore[arg-type]
        num_points=_integer(raw.get("num_points", 1024), "sampling.num_points"),
        stride=_integer(raw.get("stride", 1), "sampling.stride"),
        voxel_size=float(raw.get("voxel_size", 0.01)),
        seed=None
        if raw.get("seed") is None
        else _integer(raw["seed"], "sampling.seed"),
        deterministic=_boolean(
            raw.get("deterministic", True), "sampling.deterministic"
        ),
        pad_mode=str(raw.get("pad_mode", "repeat")),  # type: ignore[arg-type]
    )


def _live(value: Any) -> RigLiveConfig:
    raw = _mapping(value, "live")
    _keys(
        raw,
        {
            "buffer_capacity",
            "matcher_wait_timeout_s",
            "join_timeout_s",
            "telemetry_history_capacity",
        },
        "live",
    )
    buffer_capacity = _integer(raw.get("buffer_capacity", 2), "live.buffer_capacity")
    if not 1 <= buffer_capacity <= 4:
        raise ValueError("live.buffer_capacity must be between 1 and 4")
    matcher_wait_timeout_s = _finite_float(
        raw.get("matcher_wait_timeout_s", 0.1), "live.matcher_wait_timeout_s"
    )
    if matcher_wait_timeout_s <= 0.0:
        raise ValueError("live.matcher_wait_timeout_s must be positive")
    join_timeout_s = _finite_float(
        raw.get("join_timeout_s", 5.0), "live.join_timeout_s"
    )
    if join_timeout_s <= 0.0:
        raise ValueError("live.join_timeout_s must be positive")
    telemetry_history_capacity = _integer(
        raw.get("telemetry_history_capacity", 10_000),
        "live.telemetry_history_capacity",
    )
    if not 1 <= telemetry_history_capacity <= 100_000:
        raise ValueError("live.telemetry_history_capacity must be between 1 and 100000")
    return RigLiveConfig(
        buffer_capacity=buffer_capacity,
        matcher_wait_timeout_s=matcher_wait_timeout_s,
        join_timeout_s=join_timeout_s,
        telemetry_history_capacity=telemetry_history_capacity,
    )


def _keys(raw: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {unknown}")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def _integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _path(value: Any, label: str, base_dir: Path | None) -> str:
    text = _nonempty(value, label)
    if text.startswith("synthetic://") or base_dir is None:
        return text
    path = Path(text).expanduser()
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def _range(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must be a two-element list")
    result = (float(value[0]), float(value[1]))
    if result[0] > result[1]:
        raise ValueError(f"{label} must be ordered")
    return result
