#!/usr/bin/env python3
"""Record matched same-pass rig depth observations into an atomic local artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from pointcloud_builder.mapping.recording import RigDepthRecordingWriter
from pointcloud_builder.rig import build_live_rig, load_rig_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--matched-sets", type=int, default=300)
    parser.add_argument(
        "--depth-source", choices=("native", "ffs_stereo"), required=True
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.matched_sets <= 0:
        raise ValueError("--matched-sets must be positive")
    output = _private_output(args.output)
    config = load_rig_config(args.config)
    actual_sources = {camera.depth.mode for camera in config.enabled_cameras}
    if actual_sources != {args.depth_source}:
        raise ValueError(
            f"rig depth modes {sorted(actual_sources)} differ from --depth-source"
        )
    pipeline = build_live_rig(config, device="cuda")
    writer = RigDepthRecordingWriter(output, depth_source=args.depth_source)
    started = time.perf_counter()
    try:
        pipeline.acquisition.start()
        for _ in range(args.matched_sets):
            writer.append(pipeline.capture_next().result.depth_frame_set)
        pipeline.acquisition.stop()
        acquisition = pipeline.acquisition.report()
        duration_s = time.perf_counter() - started
        writer.finalize(
            report={
                "schema_version": "pointcloud-builder.rig-depth-recording-report.v1",
                "matched_sets": args.matched_sets,
                "depth_source": args.depth_source,
                "duration_s": duration_s,
                "recording_fps": args.matched_sets / duration_s,
                "matcher": acquisition["matcher"],
                "workers_clean": not acquisition["workers_alive"]
                and not acquisition["worker_errors"],
            }
        )
    except BaseException:
        try:
            pipeline.acquisition.stop()
        finally:
            writer.abort()
        raise
    print(
        json.dumps(
            {
                "schema_version": "pointcloud-builder.rig-depth-recording-cli.v1",
                "matched_sets": args.matched_sets,
                "depth_source": args.depth_source,
                "recording_fps": args.matched_sets / duration_s,
                "published": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _private_output(value: str) -> Path:
    output = Path(value).resolve()
    local = (Path.cwd() / ".local").resolve()
    if not output.is_relative_to(local):
        raise ValueError("real rig-depth recordings must be written under .local/")
    return output


if __name__ == "__main__":
    main()
