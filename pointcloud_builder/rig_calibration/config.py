"""Configuration and frozen preflight thresholds for rig calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RigCalibrationConfig:
    """Joint-solver, robust-loss, diversity, and acceptance configuration."""

    anchor_pose_id: str = "pose_0"
    robust_loss: str = "huber"
    loss_scale_px: float = 1.0
    max_nfev: int = 1500
    min_corners_per_observation: int = 6
    min_observations_per_camera: int = 3
    min_pose_count: int = 6
    min_image_coverage_fraction: float = 0.08
    min_translation_span_m: float = 0.08
    min_depth_span_m: float = 0.08
    min_normal_span_deg: float = 12.0
    min_yaw_span_deg: float = 6.0
    min_pitch_span_deg: float = 6.0
    final_reprojection_p95_px: float = 1.0
    max_condition_number: float = 1.0e8
    optimizer_xtol: float = 1e-10
    optimizer_ftol: float = 1e-10
    optimizer_gtol: float = 1e-10

    def __post_init__(self) -> None:
        if not self.anchor_pose_id.strip():
            raise ValueError("anchor_pose_id must be non-empty")
        if self.robust_loss not in {"huber", "cauchy"}:
            raise ValueError("robust_loss must be 'huber' or 'cauchy'")
        if self.loss_scale_px <= 0.0 or self.max_nfev <= 0:
            raise ValueError("loss_scale_px and max_nfev must be positive")
        if self.min_corners_per_observation < 4:
            raise ValueError("min_corners_per_observation must be at least four")
        if self.min_observations_per_camera < 1 or self.min_pose_count < 2:
            raise ValueError("camera observation and pose minima must be positive")
        for name in (
            "min_image_coverage_fraction",
            "min_translation_span_m",
            "min_depth_span_m",
            "min_normal_span_deg",
            "min_yaw_span_deg",
            "min_pitch_span_deg",
            "final_reprojection_p95_px",
            "max_condition_number",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_rig_calibration_config(path: str | Path) -> RigCalibrationConfig:
    raw = yaml.safe_load(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("rig calibration config must be a YAML mapping")
    return RigCalibrationConfig(**raw)
