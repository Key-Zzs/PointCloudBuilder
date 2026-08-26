#!/usr/bin/env python3
"""Build a fixed-workspace TSDF directly from per-camera recorded depth rays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from pointcloud_builder.mapping.artifact import write_tsdf_map_artifact
from pointcloud_builder.mapping.config import load_tsdf_config
from pointcloud_builder.mapping.open3d import Open3dTsdfMap
from pointcloud_builder.mapping.recording import (
    iter_rig_depth_recording,
    validate_rig_depth_recording,
)
from pointcloud_builder.mapping.validation import sha256_file


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
    started = time.perf_counter()
    try:
        for frame_set in iter_rig_depth_recording(recording):
            result = mapper.integrate(frame_set)
            if not result.skipped:
                integrations.append(result)
        if not integrations:
            raise RuntimeError("TSDF frame stride selected no recording frames")
        mapper.freeze()
        latencies = [item.integration_ms for item in integrations]
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
                "estimated_attribute_bytes": config.estimated_attribute_bytes,
            },
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
