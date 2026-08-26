"""Versioned fixed-camera TSDF mapping contracts with lazy optional backends."""

from pointcloud_builder.mapping.config import TsdfMapConfig, load_tsdf_config
from pointcloud_builder.mapping.types import (
    DynamicMaskReport,
    MapExtraction,
    RigDepthFrameSet,
    RigDepthObservation,
    TsdfIntegrationResult,
    TsdfMapArtifact,
    TsdfMapState,
)

__all__ = [
    "DynamicMaskReport",
    "MapExtraction",
    "RigDepthFrameSet",
    "RigDepthObservation",
    "TsdfIntegrationResult",
    "TsdfMapArtifact",
    "TsdfMapConfig",
    "TsdfMapState",
    "load_tsdf_config",
]
