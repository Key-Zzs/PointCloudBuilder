"""Typed, serialisable support-plane model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SupportPlane:
    """Plane ``normal dot x + offset = 0`` in the input camera frame."""

    normal: tuple[float, float, float]
    offset: float
    distance_threshold_m: float
    source_frame_indices: tuple[int, ...]
    inlier_ratio: float
    residual_median: float
    residual_p95: float
    config_hash: str | None = None
    support_polygon: tuple[tuple[float, float, float], ...] | None = None

    def normal_array(self) -> np.ndarray:
        value = np.asarray(self.normal, dtype=np.float64)
        norm = float(np.linalg.norm(value))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("support-plane normal must be finite and non-zero")
        return value / norm

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SupportPlane":
        polygon = value.get("support_polygon")
        return cls(
            normal=tuple(float(item) for item in value["normal"]),  # type: ignore[arg-type]
            offset=float(value["offset"]),
            distance_threshold_m=float(value["distance_threshold_m"]),
            source_frame_indices=tuple(int(item) for item in value.get("source_frame_indices", ())),
            inlier_ratio=float(value["inlier_ratio"]),
            residual_median=float(value["residual_median"]),
            residual_p95=float(value["residual_p95"]),
            config_hash=str(value["config_hash"]) if value.get("config_hash") is not None else None,
            support_polygon=(
                tuple(tuple(float(item) for item in point) for point in polygon)  # type: ignore[arg-type]
                if polygon is not None
                else None
            ),
        )
