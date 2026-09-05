#!/usr/bin/env python3
"""Compare world reconstruction, crop, and sampling on frozen replay inputs."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from pointcloud_builder.mapping.provenance import (
    rig_backend_provenance,
    validate_production_ffs_provenance,
)
from pointcloud_builder.reconstruction_benchmark import (
    benchmark_reconstruction_scenarios,
    benchmark_timing_overhead,
)
from pointcloud_builder.rig import build_live_rig, build_replay_rig, load_rig_config
from pointcloud_builder.rig.pipeline import RigCameraRuntime
from pointcloud_builder.rig.processor import RigFrameProcessor
from pointcloud_builder.rig_calibration.deployment import (
    configured_rig_calibration_provenance,
)
from pointcloud_builder.workspace import SingleCameraWorkspacePipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rig-config", required=True)
    parser.add_argument("--input-mode", choices=("replay", "live"), default="replay")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if args.start_frame < 0 or args.frames <= 0 or args.warmup < 0:
        raise ValueError("frame bounds and warmup must be non-negative")
    config = load_rig_config(args.rig_config)
    no_crop = replace(config.workspace_crop, enabled=False)
    no_sampling = replace(config.sampling, enabled=False)
    scenario_configs = {
        "reconstruction_only": replace(
            config, workspace_crop=no_crop, sampling=no_sampling
        ),
        "reconstruction_crop": replace(config, sampling=no_sampling),
        "reconstruction_crop_sampling": config,
    }
    indices = tuple(range(args.start_frame, args.start_frame + args.frames))
    capture_report = None
    if args.input_mode == "replay":
        factories = {
            name: (
                lambda selected=selected: build_replay_rig(selected, device=args.device)
            )
            for name, selected in scenario_configs.items()
        }
        overhead_factories = (
            lambda: build_replay_rig(config, device=args.device),
            lambda: build_replay_rig(config, device=args.device),
        )
    else:
        frame_sets, match_wait_ms, runtimes = _capture_frozen_live_inputs(
            config, args.device, args.start_frame + args.frames
        )
        selected_sets = tuple(frame_sets[index] for index in indices)
        selected_waits = tuple(match_wait_ms[index] for index in indices)
        indices = tuple(range(len(selected_sets)))
        factories = {
            name: _live_factory(selected, runtimes, selected_sets, selected_waits)
            for name, selected in scenario_configs.items()
        }
        overhead_factories = (
            _live_factory(config, runtimes, selected_sets, selected_waits),
            _live_factory(config, runtimes, selected_sets, selected_waits),
        )
        capture_report = {
            "matched_set_count": len(selected_sets),
            "frame_match_wait_ms": _summary(selected_waits),
        }
    report = benchmark_reconstruction_scenarios(
        factories, frame_indices=indices, warmup=min(args.warmup, len(indices))
    )
    report["instrumentation_overhead"] = benchmark_timing_overhead(
        overhead_factories[0],
        overhead_factories[1],
        frame_indices=indices,
        warmup=min(args.warmup, len(indices)),
    )
    backend_binding = _production_backend_binding(config)
    rig_calibration = configured_rig_calibration_provenance(config)
    if rig_calibration["production_applied"] is not True:
        raise ValueError(
            "production benchmark requires a validated multi-pose deployment"
        )
    report.update(
        {
            "rig_schema_version": config.schema_version,
            "camera_count": len(config.enabled_cameras),
            "camera_names": [camera.name for camera in config.enabled_cameras],
            "depth_sources": {
                camera.name: camera.depth.mode for camera in config.enabled_cameras
            },
            "rig_calibration": rig_calibration,
            "production_backend_binding": backend_binding,
            "passed": bool(
                report["same_inputs"]
                and report["instrumentation_overhead"]["passed"]
                and backend_binding["passed"]
            ),
            "device": args.device,
            "input_mode": args.input_mode,
            "capture": capture_report,
            "production_order_note": (
                "Existing M8 workspace-crop/fusion semantics are retained; scenario "
                "timings report the actual execution order."
            ),
        }
    )
    destination = Path(args.report)
    if destination.exists():
        raise FileExistsError(f"benchmark report already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


class _FrozenFrameSetPipeline:
    def __init__(
        self,
        processor: RigFrameProcessor,
        frame_sets: tuple[object, ...],
        match_wait_ms: tuple[float, ...],
    ) -> None:
        self.processor = processor
        self.frame_sets = frame_sets
        self.match_wait_ms = match_wait_ms

    def build(self, index: int):
        return self.processor.process_frame_set(
            self.frame_sets[index], frame_match_ms=None
        )


def _capture_frozen_live_inputs(config, device: str, count: int):
    live = build_live_rig(config, device=device)
    frame_sets = []
    match_wait_ms = []
    live.acquisition.start()
    try:
        for _ in range(count):
            started = time.perf_counter()
            frame_set = live.acquisition.next_frame_set()
            if frame_set is None:
                raise TimeoutError(
                    "live benchmark did not receive a complete frame set"
                )
            match_wait_ms.append((time.perf_counter() - started) * 1000.0)
            frame_sets.append(frame_set)
    finally:
        live.acquisition.stop()
    return tuple(frame_sets), tuple(match_wait_ms), live.processor.runtimes


def _live_factory(config, base_runtimes, frame_sets, match_wait_ms):
    def build():
        runtimes = {}
        for name, runtime in base_runtimes.items():
            pipeline = runtime.pipeline
            runtimes[name] = RigCameraRuntime(
                source=SimpleNamespace(camera_name=name),
                pipeline=SingleCameraWorkspacePipeline(
                    pipeline.context,
                    workspace_crop=config.workspace_crop,
                    provision_sha256=pipeline.provision_sha256,
                ),
                provenance=dict(runtime.provenance),
            )
        return _FrozenFrameSetPipeline(
            RigFrameProcessor(config, runtimes), frame_sets, match_wait_ms
        )

    return build


def _summary(values: tuple[float, ...]) -> dict[str, float]:
    from pointcloud_builder.reconstruction_timing import summarize_ms

    return summarize_ms(list(values))


def _production_backend_binding(config) -> dict[str, object]:
    cameras = rig_backend_provenance(config)
    try:
        validate_production_ffs_provenance(cameras, tuple(cameras))
        passed = True
    except ValueError:
        passed = False
    return {
        "required_backend": "tensorrt_plugin",
        "cameras": cameras,
        "passed": passed,
    }


if __name__ == "__main__":
    main()
