"""YAML configuration parsing for PointCloudBuilder."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from pointcloud_builder.camera_model import CameraExtrinsics, CameraIntrinsics

SamplingMode = Literal[
    "fps",
    "stride",
    "random",
    "voxel",
    "voxel_random",
    "voxel_fps",
]
PadMode = Literal["repeat", "zero"]
DepthSourceMode = Literal["frame", "ffs_stereo"]
FFSBackendName = Literal[
    "pytorch",
    "tensorrt_single",
    "tensorrt_two_stage",
    "tensorrt_plugin",
]


@dataclass(frozen=True)
class CameraConfig:
    """Camera model and depth interpretation settings."""

    name: str
    depth_scale: float
    aligned_depth_to_color: bool
    color_intrinsics: CameraIntrinsics
    depth_intrinsics: CameraIntrinsics
    depth_to_color_extrinsics: CameraExtrinsics | None = None


@dataclass(frozen=True)
class PointCloudConfig:
    """Raw point-cloud output settings."""

    use_rgb: bool
    output_format: str
    rgb_mapping: str = "aligned"
    rgb_sampling: str = "nearest"
    xyz_frame: str = "depth"


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
class FFSConfig:
    """Configuration shared by the four optional FFS backends."""

    backend: FFSBackendName
    left_key: str = "left_ir"
    right_key: str = "right_ir"
    checkpoint_path: str | None = None
    model_config_path: str | None = None
    engine_path: str | None = None
    feature_engine_path: str | None = None
    post_engine_path: str | None = None
    plugin_library_path: str | None = None
    manifest_path: str | None = None
    calibration_path: str | None = None
    calibration_camera: str = "head"
    width: int = 640
    height: int = 480
    max_disp: int = 416
    valid_iters: int = 8
    precision: str = "fp16"
    cv_group: int = 8
    builder_optimization_level: int = 3
    workspace_gib: float = 8.0
    config_path: str | None = None
    artifact_id: str | None = None
    baseline_m: float = 0.0
    rectification_mode: str = "auto"
    remove_invisible: bool = True
    min_disparity_px: float = 0.001
    min_depth_m: float = 0.0
    max_depth_m: float | None = None
    right_intrinsics: CameraIntrinsics | None = None
    left_distortion: tuple[float, ...] = ()
    right_distortion: tuple[float, ...] = ()


@dataclass(frozen=True)
class DepthSourceConfig:
    """Select the original frame depth or the shared FFS stereo resolver."""

    mode: DepthSourceMode = "frame"
    ffs: FFSConfig | None = None


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
    depth_source: DepthSourceConfig = field(default_factory=DepthSourceConfig)


def load_config(path: str | Path) -> PointCloudBuilderConfig:
    """Load a builder configuration from YAML."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    _resolve_relative_ffs_paths(raw, config_path.parent)
    return parse_config(raw)


def _resolve_relative_ffs_paths(raw: dict[str, Any], config_dir: Path) -> None:
    """Resolve optional FFS assets relative to the YAML that declares them."""

    depth_source = raw.get("depth_source")
    if not isinstance(depth_source, dict):
        return
    ffs = depth_source.get("ffs")
    if not isinstance(ffs, dict):
        return
    for key in (
        "checkpoint_path",
        "model_config_path",
        "engine_path",
        "feature_engine_path",
        "post_engine_path",
        "plugin_library_path",
        "manifest_path",
        "config_path",
        "calibration_path",
    ):
        value = ffs.get(key)
        if not isinstance(value, str) or not value:
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            ffs[key] = str((config_dir / candidate).resolve())


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
        depth_to_color_extrinsics=_parse_extrinsics(camera_raw.get("depth_to_color_extrinsics")),
    )
    depth_source = _parse_depth_source(raw.get("depth_source"))
    if depth_source.mode == "ffs_stereo" and camera.aligned_depth_to_color:
        raise ValueError("depth_source.mode=ffs_stereo requires camera.aligned_depth_to_color=false")
    default_rgb_mapping = "aligned" if camera.aligned_depth_to_color else "project_depth_to_color"
    pointcloud = PointCloudConfig(
        use_rgb=bool(pointcloud_raw.get("use_rgb", False)),
        output_format=str(pointcloud_raw.get("output_format", "xyz")).lower(),
        rgb_mapping=str(pointcloud_raw.get("rgb_mapping", default_rgb_mapping)).lower(),
        rgb_sampling=str(pointcloud_raw.get("rgb_sampling", "nearest")).lower(),
        xyz_frame=str(pointcloud_raw.get("xyz_frame", "depth")).lower(),
    )
    if pointcloud.output_format not in {"xyz", "xyzrgb"}:
        raise ValueError("pointcloud.output_format must be 'xyz' or 'xyzrgb'")
    if pointcloud.rgb_mapping not in {"aligned", "project_depth_to_color"}:
        raise ValueError("pointcloud.rgb_mapping must be 'aligned' or 'project_depth_to_color'")
    if pointcloud.rgb_sampling != "nearest":
        raise ValueError("Only pointcloud.rgb_sampling='nearest' is currently supported")
    if pointcloud.xyz_frame != "depth":
        raise ValueError("Only pointcloud.xyz_frame='depth' is currently supported")
    if (
        pointcloud.use_rgb
        and pointcloud.output_format == "xyzrgb"
        and not camera.aligned_depth_to_color
        and pointcloud.rgb_mapping == "project_depth_to_color"
        and camera.depth_to_color_extrinsics is None
        and depth_source.mode == "frame"
    ):
        raise ValueError(
            "camera.depth_to_color_extrinsics is required for raw depth RGB mapping"
        )
    crop = _parse_crop(raw.get("crop"))
    sampling = _parse_sampling(raw.get("sampling"))
    return PointCloudBuilderConfig(
        camera=camera,
        pointcloud=pointcloud,
        device=str(raw.get("device", "auto")),
        crop=crop,
        sampling=sampling,
        depth_source=depth_source,
    )


def _parse_depth_source(value: Any) -> DepthSourceConfig:
    """Parse the optional depth source without importing any FFS dependency."""

    if value is None:
        return DepthSourceConfig()
    if not isinstance(value, dict):
        raise ValueError("depth_source must be a mapping")
    mode = str(value.get("mode", "frame")).lower()
    if mode == "frame":
        return DepthSourceConfig(mode="frame")
    if mode != "ffs_stereo":
        raise ValueError("depth_source.mode must be 'frame' or 'ffs_stereo'")
    ffs_raw = value.get("ffs")
    if not isinstance(ffs_raw, dict):
        raise ValueError("depth_source.ffs is required for mode=ffs_stereo")
    backend = str(ffs_raw.get("backend", "")).lower()
    valid_backends = {"pytorch", "tensorrt_single", "tensorrt_two_stage", "tensorrt_plugin"}
    if backend not in valid_backends:
        raise ValueError(f"depth_source.ffs.backend must be one of {sorted(valid_backends)}")
    width = int(ffs_raw.get("width", 640))
    height = int(ffs_raw.get("height", 480))
    if (height, width) != (480, 640):
        raise ValueError(
            "Current FFS mode is fixed to input height=480,width=640; "
            f"got height={height}, width={width}"
        )
    max_disp = int(ffs_raw.get("max_disp", 416))
    valid_iters = int(ffs_raw.get("valid_iters", 8))
    if max_disp <= 0 or max_disp % 4 != 0:
        raise ValueError("depth_source.ffs.max_disp must be positive and divisible by 4")
    if valid_iters <= 0:
        raise ValueError("depth_source.ffs.valid_iters must be positive")
    precision = str(ffs_raw.get("precision", "fp16")).lower()
    if precision not in {"fp16", "fp32"}:
        raise ValueError("depth_source.ffs.precision must be fp16 or fp32")
    builder_optimization_level = int(ffs_raw.get("builder_optimization_level", 3))
    if not 0 <= builder_optimization_level <= 5:
        raise ValueError("depth_source.ffs.builder_optimization_level must be between 0 and 5")
    workspace_gib = float(ffs_raw.get("workspace_gib", 8.0))
    if workspace_gib <= 0.0:
        raise ValueError("depth_source.ffs.workspace_gib must be positive")
    rectification_mode = str(ffs_raw.get("rectification_mode", "auto")).lower()
    if rectification_mode not in {"auto", "require_rectified", "opencv"}:
        raise ValueError("depth_source.ffs.rectification_mode must be auto, require_rectified, or opencv")
    max_depth = ffs_raw.get("max_depth_m")
    max_depth_m = float(max_depth) if max_depth is not None else None
    right_intrinsics_raw = ffs_raw.get("right_intrinsics")
    right_intrinsics = (
        _parse_intrinsics(_require_mapping(right_intrinsics_raw, "depth_source.ffs.right_intrinsics"), "depth_source.ffs.right_intrinsics")
        if right_intrinsics_raw is not None
        else None
    )
    ffs = FFSConfig(
        backend=backend,  # type: ignore[arg-type]
        left_key=str(ffs_raw.get("left_key", "left_ir")),
        right_key=str(ffs_raw.get("right_key", "right_ir")),
        checkpoint_path=_optional_string(ffs_raw.get("checkpoint_path")),
        model_config_path=_optional_string(ffs_raw.get("model_config_path")),
        engine_path=_optional_string(ffs_raw.get("engine_path")),
        feature_engine_path=_optional_string(ffs_raw.get("feature_engine_path")),
        post_engine_path=_optional_string(ffs_raw.get("post_engine_path")),
        plugin_library_path=_optional_string(ffs_raw.get("plugin_library_path")),
        manifest_path=_optional_string(ffs_raw.get("manifest_path")),
        calibration_path=_optional_string(ffs_raw.get("calibration_path")),
        calibration_camera=str(ffs_raw.get("calibration_camera", "head")),
        width=width,
        height=height,
        max_disp=max_disp,
        valid_iters=valid_iters,
        precision=precision,
        cv_group=int(ffs_raw.get("cv_group", 8)),
        builder_optimization_level=builder_optimization_level,
        workspace_gib=workspace_gib,
        config_path=_optional_string(ffs_raw.get("config_path")),
        artifact_id=_optional_string(ffs_raw.get("artifact_id")),
        baseline_m=float(ffs_raw.get("baseline_m", 0.0)),
        rectification_mode=rectification_mode,
        remove_invisible=bool(ffs_raw.get("remove_invisible", True)),
        min_disparity_px=float(ffs_raw.get("min_disparity_px", 0.001)),
        min_depth_m=float(ffs_raw.get("min_depth_m", 0.0)),
        max_depth_m=max_depth_m,
        right_intrinsics=right_intrinsics,
        left_distortion=tuple(float(x) for x in ffs_raw.get("left_distortion", ())),
        right_distortion=tuple(float(x) for x in ffs_raw.get("right_distortion", ())),
    )
    if ffs.min_disparity_px <= 0.0:
        raise ValueError("depth_source.ffs.min_disparity_px must be positive")
    if ffs.cv_group <= 0:
        raise ValueError("depth_source.ffs.cv_group must be positive")
    if ffs.min_depth_m < 0.0:
        raise ValueError("depth_source.ffs.min_depth_m must be non-negative")
    if ffs.max_depth_m is not None and ffs.max_depth_m <= 0.0:
        raise ValueError("depth_source.ffs.max_depth_m must be positive")
    return DepthSourceConfig(mode="ffs_stereo", ffs=ffs)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_intrinsics(raw: dict[str, Any], name: str) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=int(_require_value(raw, "width")),
        height=int(_require_value(raw, "height")),
        fx=float(_require_value(raw, "fx")),
        fy=float(_require_value(raw, "fy")),
        cx=float(_require_value(raw, "cx")),
        cy=float(_require_value(raw, "cy")),
    )


def _parse_extrinsics(value: Any) -> CameraExtrinsics | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("camera.depth_to_color_extrinsics must be a mapping")

    rotation_value = value.get("rotation_matrix_row_major", value.get("rotation"))
    if rotation_value is None:
        raise ValueError("camera.depth_to_color_extrinsics.rotation is required")
    if isinstance(rotation_value, list | tuple) and len(rotation_value) == 9:
        rotation_rows = [
            rotation_value[0:3],
            rotation_value[3:6],
            rotation_value[6:9],
        ]
    else:
        rotation_rows = rotation_value
    if not isinstance(rotation_rows, list | tuple) or len(rotation_rows) != 3:
        raise ValueError("camera.depth_to_color_extrinsics.rotation must be 3x3 or flat length 9")
    rotation = []
    for row in rotation_rows:
        if not isinstance(row, list | tuple) or len(row) != 3:
            raise ValueError("camera.depth_to_color_extrinsics.rotation rows must have length 3")
        rotation.append(tuple(float(x) for x in row))

    translation_value = value.get("translation", value.get("translation_m"))
    if translation_value is None:
        raise ValueError("camera.depth_to_color_extrinsics.translation is required")
    if not isinstance(translation_value, list | tuple) or len(translation_value) != 3:
        raise ValueError("camera.depth_to_color_extrinsics.translation must have length 3")
    translation = tuple(float(x) for x in translation_value)
    return CameraExtrinsics(
        rotation=(rotation[0], rotation[1], rotation[2]),  # type: ignore[arg-type]
        translation=translation,  # type: ignore[arg-type]
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
