"""FFS stereo calibration and rectification gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pointcloud_builder.camera_model import CameraExtrinsics, CameraIntrinsics


@dataclass(frozen=True)
class FFSCalibration:
    """Calibration in the left-IR optical frame."""

    left_intrinsics: CameraIntrinsics
    right_intrinsics: CameraIntrinsics
    left_distortion: tuple[float, ...]
    right_distortion: tuple[float, ...]
    left_to_right: CameraExtrinsics
    left_to_color: CameraExtrinsics | None
    baseline_m: float
    rectification_mode: str
    rectification_identity: bool

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "rectification_mode": self.rectification_mode,
            "rectification_identity": self.rectification_identity,
            "left_intrinsics": _intrinsics_dict(self.left_intrinsics),
            "right_intrinsics": _intrinsics_dict(self.right_intrinsics),
            "left_distortion": list(self.left_distortion),
            "right_distortion": list(self.right_distortion),
            "baseline_m": self.baseline_m,
            "left_to_right_translation_m": list(self.left_to_right.translation),
        }


def load_realsense_calibration(path: str | Path, camera: str, *, rectification_mode: str = "auto") -> FFSCalibration:
    """Load the IR1/IR2 and IR1-to-color entries from v05-style JSON."""

    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("cameras"), Mapping):
        raise ValueError(f"Invalid RealSense calibration mapping: {path}")
    aliases = {"head": "head_rgb", "left_wrist": "left_wrist_rgb", "right_wrist": "right_wrist_rgb"}
    camera_key = aliases.get(camera, camera)
    camera_raw = raw["cameras"].get(camera_key)
    if not isinstance(camera_raw, Mapping):
        raise KeyError(f"Calibration has no camera {camera!r} (looked for {camera_key!r})")
    streams = camera_raw.get("streams")
    extrinsics = camera_raw.get("extrinsics")
    baseline = camera_raw.get("baseline")
    if not isinstance(streams, Mapping) or not isinstance(extrinsics, Mapping) or not isinstance(baseline, Mapping):
        raise ValueError(f"Calibration camera {camera_key!r} lacks streams/extrinsics/baseline")
    left_stream = _require_mapping(streams, "infrared1")
    right_stream = _require_mapping(streams, "infrared2")
    left_i, left_d = _stream_intrinsics(left_stream, f"{camera_key}.infrared1")
    right_i, right_d = _stream_intrinsics(right_stream, f"{camera_key}.infrared2")
    left_to_right = _parse_extrinsics(_require_mapping(extrinsics, "infrared1_to_infrared2"), "infrared1_to_infrared2")
    left_to_color_value = extrinsics.get("infrared1_to_color")
    left_to_color = _parse_extrinsics(left_to_color_value, "infrared1_to_color") if isinstance(left_to_color_value, Mapping) else None
    baseline_m = float(baseline.get("recommended_baseline_m", baseline.get("baseline_m_abs_x", 0.0)))
    return make_calibration(
        left_i,
        right_i,
        left_d,
        right_d,
        left_to_right,
        left_to_color,
        baseline_m,
        rectification_mode=rectification_mode,
    )


def make_calibration(
    left_intrinsics: CameraIntrinsics,
    right_intrinsics: CameraIntrinsics,
    left_distortion: tuple[float, ...] | list[float],
    right_distortion: tuple[float, ...] | list[float],
    left_to_right: CameraExtrinsics,
    left_to_color: CameraExtrinsics | None,
    baseline_m: float,
    *,
    rectification_mode: str = "auto",
) -> FFSCalibration:
    """Validate the strict identity-rectified contract used by v05."""

    mode = str(rectification_mode).lower()
    if mode not in {"auto", "require_rectified", "opencv"}:
        raise ValueError("rectification_mode must be auto, require_rectified, or opencv")
    if baseline_m <= 0.0:
        raise ValueError(f"baseline_m must be positive, got {baseline_m}")
    if left_intrinsics.width != right_intrinsics.width or left_intrinsics.height != right_intrinsics.height:
        raise ValueError("Left and right IR image sizes must match")
    identity_rotation = _is_identity_rotation(left_to_right.rotation)
    horizontal = abs(left_to_right.translation[0]) >= 0.999 * max(abs(baseline_m), 1e-9)
    no_vertical_or_forward = max(abs(left_to_right.translation[1]), abs(left_to_right.translation[2])) <= 1e-4
    same_k = _intrinsics_close(left_intrinsics, right_intrinsics)
    zero_distortion = all(abs(x) <= 1e-9 for x in (*left_distortion, *right_distortion))
    expected_negative_x = left_to_right.translation[0] < 0.0
    identity = identity_rotation and horizontal and no_vertical_or_forward and same_k and zero_distortion and expected_negative_x
    if mode == "opencv":
        raise ValueError(
            "opencv rectification is intentionally not enabled for the realtime contract; "
            "v05 requires identity/no-op rectification"
        )
    if not identity:
        raise ValueError(
            "FFS requires rectified IR input: expected equal K, zero distortion, identity rotation, "
            "and left_to_right translation (-baseline,0,0); use an explicit offline rectifier before inference"
        )
    return FFSCalibration(
        left_intrinsics=left_intrinsics,
        right_intrinsics=right_intrinsics,
        left_distortion=tuple(float(x) for x in left_distortion),
        right_distortion=tuple(float(x) for x in right_distortion),
        left_to_right=left_to_right,
        left_to_color=left_to_color,
        baseline_m=float(baseline_m),
        rectification_mode="identity/no-op",
        rectification_identity=True,
    )


def calibration_from_builder_config(camera: Any, ffs: Any) -> FFSCalibration:
    """Build calibration from a dataset JSON or explicit builder YAML values."""

    if getattr(ffs, "calibration_path", None):
        loaded = load_realsense_calibration(
            ffs.calibration_path,
            getattr(ffs, "calibration_camera", "head"),
            rectification_mode=ffs.rectification_mode,
        )
        configured_baseline = float(getattr(ffs, "baseline_m", 0.0) or 0.0)
        if configured_baseline > 0.0 and abs(configured_baseline - loaded.baseline_m) > 1e-5:
            raise ValueError(
                f"Configured baseline_m={configured_baseline} disagrees with calibration {loaded.baseline_m}"
            )
        return loaded

    baseline_m = float(getattr(ffs, "baseline_m", 0.0) or 0.0)
    right_intrinsics = getattr(ffs, "right_intrinsics", None) or camera.depth_intrinsics
    left_to_right = CameraExtrinsics(
        rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        translation=(-baseline_m, 0.0, 0.0),
    )
    return make_calibration(
        camera.depth_intrinsics,
        right_intrinsics,
        tuple(getattr(ffs, "left_distortion", ()) or ()),
        tuple(getattr(ffs, "right_distortion", ()) or ()),
        left_to_right,
        camera.depth_to_color_extrinsics,
        baseline_m,
        rectification_mode=ffs.rectification_mode,
    )


def _stream_intrinsics(stream: Mapping[str, Any], name: str) -> tuple[CameraIntrinsics, tuple[float, ...]]:
    intrinsics = _require_mapping(stream, "intrinsics")
    return (
        CameraIntrinsics(
            width=int(intrinsics["width"]),
            height=int(intrinsics["height"]),
            fx=float(intrinsics["fx"]),
            fy=float(intrinsics["fy"]),
            cx=float(intrinsics["cx"]),
            cy=float(intrinsics["cy"]),
        ),
        tuple(float(x) for x in intrinsics.get("coeffs", ())),
    )


def _parse_extrinsics(value: Mapping[str, Any], name: str) -> CameraExtrinsics:
    rotation_value = value.get("rotation_matrix_row_major", value.get("rotation"))
    translation_value = value.get("translation_m", value.get("translation"))
    if rotation_value is None or translation_value is None:
        raise ValueError(f"Missing rotation/translation in {name}")
    if len(rotation_value) == 9:
        rows = [rotation_value[0:3], rotation_value[3:6], rotation_value[6:9]]
    else:
        rows = rotation_value
    if len(rows) != 3 or any(len(row) != 3 for row in rows) or len(translation_value) != 3:
        raise ValueError(f"Invalid transform shape in {name}")
    return CameraExtrinsics(
        rotation=tuple(tuple(float(x) for x in row) for row in rows),  # type: ignore[arg-type]
        translation=tuple(float(x) for x in translation_value),  # type: ignore[arg-type]
    )


def _require_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise KeyError(f"Missing calibration mapping {key}")
    return result


def _is_identity_rotation(rotation: tuple[tuple[float, float, float], ...]) -> bool:
    return all(abs(rotation[r][c] - (1.0 if r == c else 0.0)) <= 1e-6 for r in range(3) for c in range(3))


def _intrinsics_close(left: CameraIntrinsics, right: CameraIntrinsics) -> bool:
    return left.width == right.width and left.height == right.height and all(
        abs(a - b) <= 1e-5 for a, b in ((left.fx, right.fx), (left.fy, right.fy), (left.cx, right.cx), (left.cy, right.cy))
    )


def _intrinsics_dict(value: CameraIntrinsics) -> dict[str, Any]:
    return {"width": value.width, "height": value.height, "fx": value.fx, "fy": value.fy, "cx": value.cx, "cy": value.cy}
