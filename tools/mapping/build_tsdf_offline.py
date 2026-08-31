#!/usr/bin/env python3
"""Build a fixed-workspace TSDF directly from per-camera recorded depth rays."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from pointcloud_builder.mapping.artifact import write_tsdf_map_artifact
from pointcloud_builder.mapping.config import load_tsdf_config
from pointcloud_builder.mapping.open3d import Open3dTsdfMap
from pointcloud_builder.mapping.recording import (
    iter_rig_depth_recording,
    validate_rig_depth_recording,
)
from pointcloud_builder.mapping.validation import sha256_file
from pointcloud_builder.reconstruction_timing import summarize_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    recording = Path(args.recording)
    output = _private_output(args.output)
    manifest = validate_rig_depth_recording(recording)
    config = load_tsdf_config(args.config)
    if config.integration.source != manifest["depth_source"]:
        raise ValueError("TSDF config source differs from rig-depth recording")
    mapper = Open3dTsdfMap(config, workspace_frame=manifest["workspace_frame"])
    integrations = []
    raw_to_depth_frame_set_ms = []
    started = time.perf_counter()
    try:
        for frame_set in iter_rig_depth_recording(recording):
            result = mapper.integrate(frame_set)
            if not result.skipped:
                integrations.append(result)
                raw_to_depth_frame_set_ms.append(
                    frame_set.raw_to_depth_frame_set_ms
                )
        if not integrations:
            raise RuntimeError("TSDF frame stride selected no recording frames")
        mapper.freeze()
        latencies = [item.integration_ms for item in integrations]
        timing_stages = {
            "block_activation_plus_coordinate_generation_ms": [
                item.block_activation_ms for item in integrations
            ],
            "volume_integrate_ms": [
                item.volume_integrate_ms for item in integrations
            ],
            "map_update_total_ms": [
                item.map_update_total_ms for item in integrations
            ],
        }
        if all(value > 0.0 for value in raw_to_depth_frame_set_ms):
            timing_stages["raw_to_tsdf_update_ms"] = [
                item.map_update_total_ms + raw_to_depth_ms
                for item, raw_to_depth_ms in zip(
                    integrations, raw_to_depth_frame_set_ms, strict=True
                )
            ]
        artifact = write_tsdf_map_artifact(
            output,
            mapper=mapper,
            source_recording_sha256=sha256_file(recording / "manifest.json"),
            integration_metrics={
                "integrated_frame_sets": len(integrations),
                "integrated_observations": sum(
                    len(item.integrated_cameras) for item in integrations
                ),
                "elapsed_s": time.perf_counter() - started,
                "latency_ms": {
                    "p50": statistics.median(latencies),
                    "p95": float(np.quantile(latencies, 0.95)),
                    "mean": statistics.mean(latencies),
                },
                "timing_ms": {
                    name: summarize_ms(values)
                    for name, values in timing_stages.items()
                },
                "estimated_attribute_bytes": config.estimated_attribute_bytes,
            },
            rig_calibration_provenance=manifest.get("rig_calibration"),
        )
    finally:
        mapper.close()
    print(
        json.dumps(
            {
                "schema_version": "pointcloud-builder.tsdf-offline-cli.v1",
                "published": True,
                "active_blocks": mapper.state.active_block_count,
                "integrated_frame_sets": len(integrations),
                "artifact_schema": artifact.manifest["schema_version"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _private_output(value: str) -> Path:
    output = Path(value).resolve()
    local = (Path.cwd() / ".local").resolve()
    if not output.is_relative_to(local):
        raise ValueError("real TSDF map artifacts must be written under .local/")
    return output


if __name__ == "__main__":
    main()
