#!/usr/bin/env python3
"""View validated candidate multi-camera geometry live without production writeback."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pointcloud_builder.rig import build_live_rig, load_rig_config
from pointcloud_builder.rig_calibration.artifact import load_solution
from pointcloud_builder.rig_calibration.diagnostics import (
    apply_candidate_to_live_pipeline,
)
from pointcloud_builder.visualization.rerun import RerunOutputConfig, RerunViewerProcess
from pointcloud_builder.visualization.rerun.conversion import packet_from_rig_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-config", required=True)
    parser.add_argument("--candidate-solution", required=True)
    parser.add_argument("--candidate-validation", required=True)
    parser.add_argument(
        "--matched-sets",
        type=int,
        help="optional finite live-view length; omit to run until Ctrl-C",
    )
    parser.add_argument("--viewer-point-budget", type=int, default=150_000)
    parser.add_argument("--rerun-record")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.matched_sets is not None and args.matched_sets <= 0:
        raise ValueError("--matched-sets must be positive when provided")
    if args.viewer_point_budget <= 0:
        raise ValueError("--viewer-point-budget must be positive")

    rig_path = _private_existing(args.rig_config, "rig config")
    solution_path = _private_existing(args.candidate_solution, "candidate solution")
    validation_path = _private_existing(
        args.candidate_validation, "candidate validation"
    )
    record_path = None
    if args.rerun_record is not None:
        record_path = _private_path(args.rerun_record, "Rerun recording")
        if record_path.suffix != ".rrd":
            raise ValueError("--rerun-record must end in .rrd")
        if record_path.exists():
            raise FileExistsError(f"Rerun recording already exists: {record_path}")

    rig = load_rig_config(rig_path)
    solution = load_solution(solution_path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    pipeline = build_live_rig(rig, device="cuda")
    contract = apply_candidate_to_live_pipeline(
        pipeline, rig, solution, validation
    )
    viewer = RerunViewerProcess(
        RerunOutputConfig(
            spawn=True,
            record_path=None if record_path is None else str(record_path),
        )
    )

    captured = 0
    started = time.perf_counter()
    viewer.start()
    print(
        "CANDIDATE_LIVE_VIEW=ACTIVE; candidate_only=true; "
        "production_applied=false; press Ctrl-C to stop",
        flush=True,
    )
    try:
        pipeline.acquisition.start()
        while args.matched_sets is None or captured < args.matched_sets:
            built = pipeline.capture_next()
            captured += 1
            elapsed_s = max(time.perf_counter() - started, 1e-9)
            result = built.result
            packet = packet_from_rig_result(
                result,
                pipeline.processor.runtimes,
                point_budget=args.viewer_point_budget,
                metrics={
                    "candidate_only": 1.0,
                    "production_applied": 0.0,
                    "skew_ms": float(result.frame_match.maximum_skew_ms),
                    "capture_fps": captured / elapsed_s,
                    "processing_fps": 1000.0 / max(built.processing_ms, 1e-9),
                    "input_points": float(result.concatenated.points.shape[0]),
                },
            )
            if not viewer.submit(packet):
                status = viewer.telemetry()
                raise RuntimeError(
                    status.child_error or "Rerun candidate viewer process stopped"
                )
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.acquisition.stop()
        telemetry = viewer.close(timeout_s=10.0)
    print(
        json.dumps(
            {
                "status": "STOPPED",
                "candidate_only": True,
                "production_applied": False,
                "solution_fingerprint": contract["solution_fingerprint"],
                "matched_sets": captured,
                "viewer_dropped_packets": telemetry.dropped_packets,
                "worker_errors": pipeline.acquisition.report()["worker_errors"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _private_existing(value: str, label: str) -> Path:
    path = _private_path(value, label)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _private_path(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    local = (Path.cwd() / ".local").resolve()
    if not path.is_relative_to(local):
        raise ValueError(f"{label} must stay under .local/")
    return path


if __name__ == "__main__":
    main()
