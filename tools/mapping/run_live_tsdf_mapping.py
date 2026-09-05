#!/usr/bin/env python3
"""Run current-snapshot fusion with an independent persistent TSDF mapper."""

from __future__ import annotations

import argparse
import json
import resource
import signal
import statistics
import sys
import tempfile
import time
from collections import deque
from collections.abc import MutableSequence, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from pointcloud_builder.mapping.artifact import (
    validate_tsdf_map_artifact,
    validate_tsdf_map_rig_calibration_compatibility,
    write_tsdf_map_artifact,
)
from pointcloud_builder.mapping.config import load_tsdf_config
from pointcloud_builder.mapping.open3d import Open3dTsdfMap
from pointcloud_builder.mapping.performance import evaluate_rss_plateau
from pointcloud_builder.mapping.process import AsyncTsdfMapper, MapperProcessConfig
from pointcloud_builder.mapping.provenance import rig_backend_provenance
from pointcloud_builder.mapping.recording import (
    RigDepthRecordingWriter,
    validate_rig_depth_recording,
)
from pointcloud_builder.mapping.validation import sha256_file
from pointcloud_builder.reconstruction_timing import summarize_ms
from pointcloud_builder.rig import build_live_rig, load_rig_config
from pointcloud_builder.rig_calibration.deployment import (
    configured_rig_calibration_provenance,
)

DEFAULT_FINITE_MATCHED_SETS = 300
DEFAULT_FINITE_VIEWER_POINT_BUDGET = 30_000
DEFAULT_INTERACTIVE_VIEWER_POINT_BUDGET = 100_000
DEFAULT_INTERACTIVE_STATS_WINDOW = 3_000


class _StopRequest:
    """Signal-safe cooperative stop state for operator mode."""

    def __init__(self) -> None:
        self.signal_name: str | None = None

    @property
    def requested(self) -> bool:
        return self.signal_name is not None

    def request(self, signum: int) -> None:
        self.signal_name = signal.Signals(signum).name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rig-config", required=True)
    parser.add_argument("--tsdf-config", required=True)
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--matched-sets", type=int)
    run_mode.add_argument(
        "--interactive",
        action="store_true",
        help="run until SIGINT/SIGTERM with Rerun spawned by default",
    )
    parser.add_argument("--build-warmup-sets", type=int, default=0)
    parser.add_argument("--initial-map")
    parser.add_argument("--map-output")
    parser.add_argument("--recording-output")
    parser.add_argument("--snapshot-baseline-report")
    parser.add_argument("--report")
    parser.add_argument("--viewer", choices=("none", "rerun"))
    parser.add_argument("--rerun-connect")
    parser.add_argument("--rerun-spawn", action="store_true")
    parser.add_argument("--rerun-record")
    parser.add_argument("--viewer-point-budget", type=int)
    parser.add_argument(
        "--interactive-stats-window",
        type=int,
        default=DEFAULT_INTERACTIVE_STATS_WINDOW,
    )
    return parser


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve mode-dependent defaults without weakening finite acceptance."""

    if args.matched_sets is None and not args.interactive:
        args.matched_sets = DEFAULT_FINITE_MATCHED_SETS
    if args.viewer is None:
        args.viewer = "rerun" if args.interactive else "none"
    if args.viewer_point_budget is None:
        args.viewer_point_budget = (
            DEFAULT_INTERACTIVE_VIEWER_POINT_BUDGET
            if args.interactive
            else DEFAULT_FINITE_VIEWER_POINT_BUDGET
        )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = resolve_args(build_parser().parse_args(argv))
    if (
        (args.matched_sets is not None and args.matched_sets <= 0)
        or args.build_warmup_sets < 0
        or args.viewer_point_budget <= 0
        or args.interactive_stats_window <= 0
    ):
        raise ValueError("counts, point budget, and statistics window must be positive")
    if args.interactive and args.map_output is not None:
        raise ValueError(
            "--interactive cannot publish a formally accepted --map-output"
        )
    if args.interactive and args.snapshot_baseline_report is not None:
        raise ValueError("--interactive does not use a finite snapshot baseline report")
    report_path = _private_output(args.report) if args.report else None
    map_output = _private_output(args.map_output) if args.map_output else None
    recording_output = (
        _private_output(args.recording_output) if args.recording_output else None
    )
    rerun_record = _private_output(args.rerun_record) if args.rerun_record else None
    if report_path is not None and report_path.exists():
        raise FileExistsError(f"live TSDF report already exists: {report_path}")
    if rerun_record is not None and rerun_record.exists():
        raise FileExistsError(f"Rerun recording already exists: {rerun_record}")
    if map_output is not None and recording_output is None:
        raise ValueError("--map-output requires --recording-output provenance")
    if map_output is not None and args.snapshot_baseline_report is None:
        raise ValueError("--map-output requires --snapshot-baseline-report")

    rig_config = load_rig_config(args.rig_config)
    live_calibration_provenance = configured_rig_calibration_provenance(rig_config)
    if live_calibration_provenance["production_applied"] is not True:
        raise ValueError(
            "live TSDF production requires a validated multi-pose deployment"
        )
    tsdf_config = load_tsdf_config(args.tsdf_config)
    baseline = (
        None
        if args.interactive
        else _load_snapshot_baseline(
            args.snapshot_baseline_report,
            rig_config_sha256=sha256_file(args.rig_config),
            matched_sets=args.matched_sets,
        )
    )
    source_modes = {camera.depth.mode for camera in rig_config.enabled_cameras}
    if source_modes != {tsdf_config.integration.source}:
        raise ValueError("live rig depth source differs from TSDF config")
    initial_volume = None
    if args.initial_map:
        initial_root = Path(args.initial_map)
        manifest = validate_tsdf_map_artifact(initial_root)
        if manifest["workspace_frame"] != rig_config.output_frame:
            raise ValueError("initial map workspace differs from live rig")
        validate_tsdf_map_rig_calibration_compatibility(
            initial_root, live_calibration_provenance
        )
        initial_config = load_tsdf_config(initial_root / "config.resolved.yaml")
        _validate_initial_map_compatibility(initial_config, tsdf_config)
        initial_volume = str(initial_root / "volume.npz")
    if tsdf_config.dynamic.mode == "frozen_static" and initial_volume is None:
        raise ValueError("frozen_static mode requires --initial-map")
    if tsdf_config.dynamic.mode == "frozen_static" and args.build_warmup_sets:
        raise ValueError("frozen_static mode cannot have build warmup sets")

    pipeline = build_live_rig(rig_config, device="cuda")
    mapper = AsyncTsdfMapper(
        MapperProcessConfig(tsdf_config, rig_config.output_frame, initial_volume)
    )
    viewer = None
    viewer_error = None
    viewer_telemetry = None
    writer = None
    writer_published = False
    map_published = False
    publication_error = None
    mapper_started = False
    mapper_telemetry = None
    acquisition_started = False
    mapper_overheads = _history(args)
    snapshot_latencies = _history(args)
    snapshot_match_wait_latencies = _history(args)
    snapshot_processing_latencies = _history(args)
    mapper_submit_accepted = 0
    mapper_submit_rejected = 0
    warmup_submit_accepted = 0
    warmup_submit_rejected = 0
    viewer_overheads = _history(args)
    latest_snapshot = None
    last_revision = None
    matched_set_count = 0
    stop_request = _StopRequest()
    original_signal_handlers = (
        _install_signal_handlers(stop_request) if args.interactive else {}
    )
    try:
        viewer, viewer_error = _start_viewer(args, rerun_record)
        if args.interactive and viewer_error is not None:
            raise RuntimeError(f"interactive Rerun startup failed: {viewer_error}")
        writer = (
            RigDepthRecordingWriter(
                recording_output,
                depth_source=tsdf_config.integration.source,
                backend_provenance=(
                    rig_backend_provenance(rig_config)
                    if tsdf_config.integration.source == "ffs_stereo"
                    else None
                ),
            )
            if recording_output is not None
            else None
        )
        mapper.start()
        mapper_started = True
        if initial_volume is not None:
            # Loading/extracting a large frozen map is startup work. Receive the
            # initial snapshot before capture so it cannot contend with the live
            # latency interval or remain in the multiprocessing feeder pipe.
            latest_snapshot = mapper.wait_for_snapshot()
        pipeline.acquisition.start()
        acquisition_started = True
        for _ in range(args.build_warmup_sets):
            if stop_request.requested:
                break
            built = pipeline.capture_next()
            if writer is not None:
                writer.append(built.result.depth_frame_set)
            if mapper.submit(built.result.depth_frame_set):
                warmup_submit_accepted += 1
            else:
                warmup_submit_rejected += 1
            snapshot = mapper.poll_snapshot()
            if snapshot is not None:
                latest_snapshot = snapshot
        if args.build_warmup_sets:
            mapper.flush()
            mapper.reset_acceptance_window()
        started = time.perf_counter()
        while not stop_request.requested and (
            args.interactive or matched_set_count < args.matched_sets
        ):
            index = matched_set_count
            built = pipeline.capture_next()
            matched_set_count += 1
            snapshot_latencies.append(built.total_ms)
            snapshot_match_wait_latencies.append(built.match_wait_ms)
            snapshot_processing_latencies.append(built.processing_ms)
            result = built.result
            if writer is not None:
                writer.append(result.depth_frame_set)
            submit_started = time.perf_counter()
            if tsdf_config.dynamic.mode == "frozen_static":
                # A frozen map is read-only by contract. Per-frame depth would be
                # discarded by the child and only add serialization contention.
                mapper.sample_resources(index)
            else:
                if mapper.submit(result.depth_frame_set):
                    mapper_submit_accepted += 1
                else:
                    mapper_submit_rejected += 1
            mapper_overheads.append((time.perf_counter() - submit_started) * 1000.0)
            snapshot = mapper.poll_snapshot()
            if snapshot is not None:
                latest_snapshot = snapshot
            if viewer is not None:
                view_started = time.perf_counter()
                try:
                    from pointcloud_builder.visualization.rerun.conversion import (
                        map_visualization_from_snapshot,
                        packet_from_rig_result,
                    )

                    metrics = {
                        "skew_ms": float(result.frame_match.maximum_skew_ms),
                        "processing_fps": 1000.0 / max(built.processing_ms, 1e-9),
                        "capture_fps": matched_set_count
                        / max(time.perf_counter() - started, 1e-9),
                        "match_fps": matched_set_count
                        / max(time.perf_counter() - started, 1e-9),
                        "input_points": float(result.concatenated.points.shape[0]),
                        "fused_voxels": float(result.fused.points.shape[0]),
                        "production_applied": 1.0,
                        "actual_concatenated_points": float(
                            result.concatenated.points.shape[0]
                        ),
                        "actual_fused_points": float(result.fused.points.shape[0]),
                        "actual_sampled_points": float(result.sampled.points.shape[0]),
                        "viewer_point_budget": float(args.viewer_point_budget),
                        "map_update_ms": (
                            0.0
                            if latest_snapshot is None
                            else latest_snapshot.integration_ms
                        ),
                        "memory": _process_peak_rss_mb(),
                    }
                    if latest_snapshot is not None:
                        metrics.update(
                            {
                                "map_active_blocks": float(
                                    latest_snapshot.map_state.active_block_count
                                ),
                                "map_active_voxels": float(
                                    latest_snapshot.active_voxel_count
                                ),
                                "map_integrated_frame_sets": float(
                                    latest_snapshot.map_state.integrated_frame_sets
                                ),
                                "map_surface_points": float(
                                    latest_snapshot.extraction.point_count
                                ),
                            }
                        )
                        for dynamic_report in latest_snapshot.dynamic_reports:
                            prefix = f"{dynamic_report.camera_name}_"
                            metrics.update(
                                {
                                    prefix + name: value
                                    for name, value in dynamic_report.metrics.items()
                                }
                            )
                    packet = packet_from_rig_result(
                        result,
                        pipeline.processor.runtimes,
                        point_budget=args.viewer_point_budget,
                        metrics=metrics,
                    )
                    if latest_snapshot is not None:
                        reset = (
                            last_revision is not None
                            and latest_snapshot.map_state.map_revision < last_revision
                        )
                        include_static = (
                            reset
                            or last_revision is None
                            or latest_snapshot.map_state.map_revision != last_revision
                        )
                        packet = replace(
                            packet,
                            map=map_visualization_from_snapshot(
                                latest_snapshot,
                                result.fused.points,
                                point_budget=args.viewer_point_budget,
                                reset=reset,
                                include_static=include_static,
                            ),
                        )
                        last_revision = latest_snapshot.map_state.map_revision
                    if not viewer.submit(packet):
                        raise RuntimeError(
                            viewer.telemetry().child_error
                            or "Rerun viewer process stopped"
                        )
                except Exception as error:
                    viewer_error = f"{type(error).__name__}: {str(error)[:500]}"
                    viewer_telemetry = viewer.close(timeout_s=5.0).__dict__
                    viewer = None
                    if args.interactive:
                        raise RuntimeError(
                            f"interactive Rerun viewer failed: {viewer_error}"
                        ) from error
                finally:
                    viewer_overheads.append(
                        (time.perf_counter() - view_started) * 1000.0
                    )
        pipeline.acquisition.stop()
        acquisition_started = False
        acquisition = pipeline.acquisition.report()
        duration_s = time.perf_counter() - started
        performance = (
            {
                "evaluated": False,
                "passed": True,
                "reason": "interactive operator mode has no finite acceptance baseline",
            }
            if args.interactive
            else _performance_comparison(
                baseline,
                matched_sets=matched_set_count,
                duration_s=duration_s,
                snapshot_latencies_ms=snapshot_latencies,
                match_wait_latencies_ms=snapshot_match_wait_latencies,
                processing_latencies_ms=snapshot_processing_latencies,
            )
        )
        acquisition_clean = (
            not acquisition["workers_alive"] and not acquisition["worker_errors"]
        )
        if map_output is not None and (
            not performance["passed"] or not acquisition_clean
        ):
            raise RuntimeError(
                "live TSDF map publication blocked by snapshot performance/acquisition gate"
            )

        recording_sha = None
        recording_calibration_provenance = live_calibration_provenance
        if writer is not None:
            writer.finalize(
                report={
                    "schema_version": "pointcloud-builder.live-tsdf-source.v1",
                    "matched_sets": matched_set_count + args.build_warmup_sets,
                    "build_warmup_sets": args.build_warmup_sets,
                    "acceptance_sets": (
                        None if args.interactive else matched_set_count
                    ),
                    "interactive": args.interactive,
                    "depth_source": tsdf_config.integration.source,
                }
            )
            writer_published = True
            assert recording_output is not None
            recording_sha = sha256_file(recording_output / "manifest.json")
            recording_calibration_provenance = validate_rig_depth_recording(
                recording_output
            )["rig_calibration"]

        if map_output is not None:
            mapper.freeze()
            local_root = Path.cwd() / ".local"
            local_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="tsdf-publish-", dir=local_root
            ) as temp:
                volume_path = Path(temp) / "volume.npz"
                mapper.save_volume(volume_path)
                mapper_telemetry = mapper.close()
                mapper_started = False
                try:
                    _require_publishable_mapper(
                        mapper_telemetry,
                        mode=tsdf_config.dynamic.mode,
                        memory_plateau=evaluate_rss_plateau(
                            mapper_telemetry.child_rss_samples_mb
                        ),
                    )
                except RuntimeError as error:
                    publication_error = str(error)
                else:
                    published = Open3dTsdfMap(
                        tsdf_config, workspace_frame=rig_config.output_frame
                    )
                    try:
                        published.load(volume_path)
                        write_tsdf_map_artifact(
                            map_output,
                            mapper=published,
                            source_recording_sha256=str(recording_sha),
                            integration_metrics={
                                "live_matched_sets": (
                                    matched_set_count + args.build_warmup_sets
                                ),
                                "build_warmup_sets": args.build_warmup_sets,
                                "acceptance_sets": matched_set_count,
                                "duration_s": duration_s,
                                "mapper_telemetry": mapper_telemetry.__dict__,
                            },
                            rig_calibration_provenance=(
                                recording_calibration_provenance
                            ),
                        )
                        map_published = True
                    finally:
                        published.close()
        else:
            mapper_telemetry = mapper.close()
            mapper_started = False
        memory_plateau = evaluate_rss_plateau(mapper_telemetry.child_rss_samples_mb)
        if viewer is not None:
            viewer_telemetry = viewer.close().__dict__
            viewer = None
        frozen_static = tsdf_config.dynamic.mode == "frozen_static"
        submissions_complete = (
            mapper_submit_accepted == 0
            and mapper_submit_rejected == 0
            and mapper_telemetry.submitted_frame_sets == 0
            and mapper_telemetry.child_received_frame_sets == 0
            if frozen_static
            else mapper_submit_accepted == matched_set_count
            and mapper_submit_rejected == 0
            and warmup_submit_accepted == args.build_warmup_sets
            and warmup_submit_rejected == 0
        )
        mapping_gates = {
            "acquisition_clean": acquisition_clean,
            "snapshot_performance": bool(performance["passed"]),
            "mapper_child_clean": mapper_telemetry.child_error is None
            and not mapper_telemetry.running,
            "mapper_queue_bounded": mapper_telemetry.maximum_queue_depth
            <= tsdf_config.integration.queue_capacity,
            "mapper_submission_policy": submissions_complete,
        }
        if not args.interactive:
            mapping_gates["mapper_memory_plateau"] = bool(memory_plateau["passed"])
        if args.interactive and viewer_telemetry is not None:
            mapping_gates["viewer_child_clean"] = (
                viewer_error is None
                and viewer_telemetry["child_error"] is None
                and not viewer_telemetry["running"]
            )
        if map_output is not None:
            mapping_gates["map_publication"] = map_published
        report = {
            "schema_version": "pointcloud-builder.live-tsdf-report.v1",
            "mode": "interactive" if args.interactive else "finite_acceptance",
            "interrupted_by": stop_request.signal_name,
            "matched_sets": matched_set_count,
            "build_warmup_sets": args.build_warmup_sets,
            "recording_matched_sets": matched_set_count + args.build_warmup_sets,
            "duration_s": duration_s,
            "snapshot_fps": matched_set_count / max(duration_s, 1e-9),
            "statistics_window": (
                args.interactive_stats_window if args.interactive else None
            ),
            "snapshot_latency_ms": _summary(snapshot_latencies),
            "snapshot_match_wait_latency_ms": _summary(snapshot_match_wait_latencies),
            "snapshot_processing_latency_ms": _summary(snapshot_processing_latencies),
            "performance_comparison": performance,
            "memory_plateau": memory_plateau,
            "acquisition": acquisition,
            "acquisition_clean": acquisition_clean,
            "viewer_point_budget": args.viewer_point_budget,
            "mapper": mapper_telemetry.__dict__,
            "tsdf_timing_ms": _timing_stage_summary(
                mapper_telemetry.child_timing_samples_ms
            ),
            "mapper_submit_accepted": mapper_submit_accepted,
            "mapper_submit_rejected": mapper_submit_rejected,
            "warmup_submit_accepted": warmup_submit_accepted,
            "warmup_submit_rejected": warmup_submit_rejected,
            "mapper_depth_submission_policy": (
                "initial_frozen_map_only" if frozen_static else "per_frame_depth"
            ),
            "mapper_producer_overhead_ms": _summary(mapper_overheads),
            "mapper_producer_overhead_p95_le_5ms": bool(
                not mapper_overheads or np.quantile(mapper_overheads, 0.95) <= 5.0
            ),
            "viewer": {
                "error": viewer_error,
                "telemetry": viewer_telemetry,
                "producer_overhead_ms": _summary(viewer_overheads),
            },
            "map_published": map_published,
            "publication_error": publication_error,
            "recording_published": writer_published,
            "rig_calibration": live_calibration_provenance,
            "gates": mapping_gates,
            "passed": all(mapping_gates.values()),
        }
        if report_path is not None:
            _write_report(report_path, report)
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(json.dumps(_stdout_summary(report), indent=2, sort_keys=True))
        if args.interactive and not report["passed"]:
            failed = [name for name, passed in mapping_gates.items() if not passed]
            raise RuntimeError(
                "interactive mapping failed operational gates: " + ", ".join(failed)
            )
        if publication_error is not None:
            raise RuntimeError(publication_error)
    except BaseException:
        if writer is not None and not writer_published:
            writer.abort()
        raise
    finally:
        if acquisition_started:
            pipeline.acquisition.stop()
        if mapper_started:
            mapper.close(timeout_s=10.0)
        if viewer is not None:
            viewer.close(timeout_s=10.0)
        _restore_signal_handlers(original_signal_handlers)


def _start_viewer(args: Any, record_path: Path | None):
    if args.viewer == "none":
        if any((args.rerun_connect, args.rerun_spawn, record_path)):
            raise ValueError("Rerun flags require --viewer rerun")
        return None, None
    from pointcloud_builder.visualization.rerun import (
        RerunOutputConfig,
        RerunViewerProcess,
    )

    implicit_spawn = (args.interactive and args.rerun_connect is None) or not any(
        (args.rerun_spawn, args.rerun_connect, record_path)
    )
    viewer = RerunViewerProcess(
        RerunOutputConfig(
            spawn=bool(args.rerun_spawn or implicit_spawn),
            connect_url=args.rerun_connect,
            record_path=None if record_path is None else str(record_path),
        )
    )
    try:
        viewer.start()
    except Exception as error:
        return None, f"{type(error).__name__}: {str(error)[:500]}"
    return viewer, None


def _private_output(value: str) -> Path:
    output = Path(value).resolve()
    local = (Path.cwd() / ".local").resolve()
    if not output.is_relative_to(local):
        raise ValueError("real mapping outputs must be written under .local/")
    return output


def _history(args: argparse.Namespace) -> MutableSequence[float]:
    """Return finite full history or a bounded interactive rolling window."""

    if args.interactive:
        return deque(maxlen=args.interactive_stats_window)
    return []


def _install_signal_handlers(
    stop_request: _StopRequest,
) -> dict[int, Any]:
    originals = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }

    def request_stop(signum: int, _frame: Any) -> None:
        stop_request.request(signum)

    for signum in originals:
        signal.signal(signum, request_stop)
    return originals


def _restore_signal_handlers(originals: dict[int, Any]) -> None:
    for signum, handler in originals.items():
        signal.signal(signum, handler)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _stdout_summary(report: dict[str, Any]) -> dict[str, Any]:
    viewer = report["viewer"]
    telemetry = viewer.get("telemetry") or {}
    acquisition = report.get("acquisition") or {}
    return {
        "mode": report["mode"],
        "interrupted_by": report["interrupted_by"],
        "matched_sets": report["matched_sets"],
        "duration_s": report["duration_s"],
        "snapshot_fps": report["snapshot_fps"],
        "viewer_dropped_packets": telemetry.get("dropped_packets", 0),
        "worker_errors": acquisition.get("worker_errors", []),
        "passed": report["passed"],
    }


def _summary(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "p50": statistics.median(values),
        "p95": float(np.quantile(values, 0.95)),
        "mean": statistics.mean(values),
        "maximum": max(values),
    }


def _timing_stage_summary(
    samples: tuple[dict[str, float], ...],
) -> dict[str, dict[str, float]]:
    if not samples:
        return {}
    names = sorted(set().union(*(sample.keys() for sample in samples)))
    return {
        name: summarize_ms(
            [float(sample[name]) for sample in samples if name in sample]
        )
        for name in names
    }


def _validate_initial_map_compatibility(initial: Any, requested: Any) -> None:
    mismatches = []
    for name in ("backend", "volume", "depth"):
        if getattr(initial, name) != getattr(requested, name):
            mismatches.append(name)
    if initial.integration.source != requested.integration.source:
        mismatches.append("integration.source")
    if initial.extraction.weight_threshold != requested.extraction.weight_threshold:
        mismatches.append("extraction.weight_threshold")
    if mismatches:
        raise ValueError(
            "initial map is incompatible with live TSDF config: "
            + ", ".join(mismatches)
        )


def _load_snapshot_baseline(
    value: str | None, *, rig_config_sha256: str, matched_sets: int
) -> dict[str, Any] | None:
    if value is None:
        return None
    baseline = json.loads(Path(value).read_text(encoding="utf-8"))
    schema = baseline.get("schema_version") if isinstance(baseline, dict) else None
    formal_fusion = schema == "pointcloud-builder.real-multicamera-fusion.v1"
    profile_viewer = baseline.get("viewer") if isinstance(baseline, dict) else None
    dense_profile = bool(
        schema == "pointcloud-builder.live-reconstruction-profile.v1"
        and baseline.get("profile") == "dense"
        and isinstance(profile_viewer, dict)
        and profile_viewer.get("requested") is False
    )
    if (
        not isinstance(baseline, dict)
        or not (formal_fusion or dense_profile)
        or baseline.get("snapshot_only") is not True
        or baseline.get("persistent_mapping") is not False
        or baseline.get("passed") is not True
        or baseline.get("rig_config_sha256") != rig_config_sha256
        or baseline.get("matched_sets") != matched_sets
    ):
        raise ValueError(
            "snapshot baseline must be a passing same-config/same-count snapshot-only report"
        )
    try:
        baseline_fps = float(baseline["throughput_fps"])
        baseline_p95_ms = float(baseline["latency_ms"]["p95"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "snapshot baseline lacks finite performance metrics"
        ) from error
    if (
        not np.isfinite((baseline_fps, baseline_p95_ms)).all()
        or baseline_fps <= 0
        or baseline_p95_ms < 0
    ):
        raise ValueError("snapshot baseline lacks finite performance metrics")
    return baseline


def _performance_comparison(
    baseline: dict[str, Any] | None,
    *,
    matched_sets: int,
    duration_s: float,
    snapshot_latencies_ms: list[float],
    match_wait_latencies_ms: list[float],
    processing_latencies_ms: list[float],
) -> dict[str, Any]:
    live_fps = matched_sets / duration_s
    live_p95_ms = float(np.quantile(snapshot_latencies_ms, 0.95))
    if baseline is None:
        return {
            "evaluated": False,
            "passed": False,
            "reason": "snapshot baseline report not supplied",
            "live_fps": live_fps,
            "live_p95_ms": live_p95_ms,
        }
    baseline_fps = float(baseline["throughput_fps"])
    baseline_p95_ms = float(baseline["latency_ms"]["p95"])
    fps_loss_ratio = max(0.0, (baseline_fps - live_fps) / baseline_fps)
    p95_increase_ms = live_p95_ms - baseline_p95_ms
    gates = {
        "snapshot_fps_loss_le_10_percent": fps_loss_ratio <= 0.10,
        "snapshot_p95_increase_le_5ms": p95_increase_ms <= 5.0,
    }
    comparison = {
        "evaluated": True,
        "passed": all(gates.values()),
        "baseline_fps": baseline_fps,
        "live_fps": live_fps,
        "fps_loss_ratio": fps_loss_ratio,
        "baseline_p95_ms": baseline_p95_ms,
        "live_p95_ms": live_p95_ms,
        "p95_increase_ms": p95_increase_ms,
        "gates": gates,
    }
    baseline_match_wait = baseline.get("match_wait_latency_ms")
    baseline_processing = baseline.get("processing_latency_ms")
    if isinstance(baseline_match_wait, dict) and isinstance(baseline_processing, dict):
        try:
            comparison["p95_breakdown_ms"] = {
                "baseline_match_wait": float(baseline_match_wait["p95"]),
                "live_match_wait": float(np.quantile(match_wait_latencies_ms, 0.95)),
                "baseline_processing": float(baseline_processing["p95"]),
                "live_processing": float(np.quantile(processing_latencies_ms, 0.95)),
            }
        except (KeyError, TypeError, ValueError):
            pass
    return comparison


def _require_publishable_mapper(
    telemetry: Any, *, mode: str, memory_plateau: dict[str, Any]
) -> None:
    if telemetry.child_error is not None or telemetry.running:
        raise RuntimeError("TSDF mapper did not close cleanly for publication")
    if mode != "frozen_static" and telemetry.child_integrated_frame_sets <= 0:
        raise RuntimeError("TSDF mapper integrated no frame sets")
    if not memory_plateau["passed"]:
        raise RuntimeError("TSDF mapper RSS did not reach the required plateau")


def _process_peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


if __name__ == "__main__":
    main()
