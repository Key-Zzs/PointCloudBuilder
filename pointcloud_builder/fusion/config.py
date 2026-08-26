"""Configuration for deterministic snapshot voxel fusion."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class VoxelFusionConfig:
    enabled: bool = False
    voxel_size_m: float = 0.01
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    deterministic: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.voxel_size_m) or self.voxel_size_m <= 0:
            raise ValueError("fusion.voxel_size_m must be finite and positive")
        if len(self.origin) != 3:
            raise ValueError("fusion.origin must have three values")
        if not all(math.isfinite(value) for value in self.origin):
            raise ValueError("fusion.origin values must be finite")
