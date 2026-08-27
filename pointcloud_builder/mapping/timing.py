"""Adapters from TSDF runtime results into the unified timing schema."""

from __future__ import annotations

from pointcloud_builder.mapping.types import MapExtraction, TsdfIntegrationResult
from pointcloud_builder.reconstruction_timing import ReconstructionTiming


def tsdf_update_timing(
    result: TsdfIntegrationResult,
    *,
    raw_to_depth_frame_set_ms: float = 0.0,
) -> ReconstructionTiming:
    return ReconstructionTiming(
        path="persistent_tsdf_update",
        stages_ms={
            "block_activation_plus_coordinate_generation_ms": result.block_activation_ms,
            "volume_integrate_ms": result.volume_integrate_ms,
            "map_update_total_ms": result.map_update_total_ms,
            "raw_to_tsdf_update_ms": raw_to_depth_frame_set_ms
            + result.map_update_total_ms,
        },
    )


def tsdf_extraction_timing(extraction: MapExtraction) -> ReconstructionTiming:
    return ReconstructionTiming(
        path="tsdf_extraction",
        stages_ms={
            "extract_point_cloud_ms": extraction.extract_point_cloud_ms,
            "extract_mesh_ms": extraction.extract_mesh_ms,
            "post_crop_ms": extraction.post_crop_ms,
            "post_sampling_ms": extraction.post_sampling_ms,
            "map_to_raw_cloud_ms": extraction.extract_raw_world_cloud_ms,
            "map_to_cropped_cloud_ms": extraction.extract_cropped_world_cloud_ms,
            "map_to_sampled_cloud_ms": extraction.extract_sampled_world_cloud_ms,
        },
    )
