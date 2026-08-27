#!/usr/bin/env python3
"""Run bounded, concurrent live-rig acquisition through the shared M6 processor."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import yaml

from pointcloud_builder.rig import build_live_rig, load_rig_config
from pointcloud_builder.visualization import save_ascii_ply
from pointcloud_builder.workspace import (
    ExpectedPlaneRegion,
    evaluate_expected_plane,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rig-config", required=True)
    parser.add_argument("--mapping-config", required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--reopen-frames", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--memory-sample-every", type=int, default=0)
    parser.add_argument("--memory-samples")
    parser.add_argument("--no-evidence", action="store_true")
    parser.add_argument(
        "--acceptance-scope",
        choices=("full", "capture_matching"),
        default="full",
        help=(
            "full preserves the legacy geometry/performance gates; capture_matching "
            "enforces only concurrent capture, matched-set delivery, and lifecycle"
        ),
    )
    args = parser.parse_args()
    if args.frames <= 0 or args.reopen_frames < 0:
        raise ValueError("--frames must be positive and --reopen-frames non-negative")
    if args.memory_sample_every < 0:
        raise ValueError("--memory-sample-every must be non-negative")
    if bool(args.memory_samples) != bool(args.memory_sample_every):
        raise ValueError("--memory-samples and --memory-sample-every must be used together")

    config = load_rig_config(args.rig_config)
    mapping = yaml.safe_load(Path(args.mapping_config).read_text(encoding="utf-8"))
    plane = _plane(mapping["expected_plane"], config.output_frame)
    output = Path(args.output)
    report_path = Path(args.report)
    output.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    memory_samples: list[dict[str, int | None]] = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if args.memory_sample_every:
        memory_samples.append(_memory_sample(0, process))

    runs = []
    for run_index, frame_count in enumerate((args.frames, args.reopen_frames)):
        if frame_count == 0:
            continue
        pipeline = build_live_rig(config, device="cuda")
        runs.append(
            _run_once(
                pipeline,
                frame_count,
                plane,
                output,
                "primary" if run_index == 0 else "reopen",
                evidence=not args.no_evidence,
                memory_sample_every=args.memory_sample_every if run_index == 0 else 0,
                memory_samples=memory_samples,
                process=process,
            )
        )

    primary = runs[0]
    matcher = primary["acquisition"]["matcher"]
    cameras = primary["acquisition"]["cameras"]
    depth_modes = {
        camera.name: camera.depth.mode for camera in config.enabled_cameras
    }
    is_ffs = all(mode == "ffs_stereo" for mode in depth_modes.values())
    processed_fps = primary["received_frames"] / primary["duration_s"]
    matcher_passed = _matcher_passed(
        matcher,
        received_frames=primary["received_frames"],
        requested_frames=args.frames,
        reference_camera=config.timing.reference_camera or min(cameras),
        require_match_ratio=args.acceptance_scope == "full",
    )
    camera_passed = all(
        item["captured"] >= args.frames
        and item["capture_fps"] >= 27.0
        and item["host_timestamp_monotonic"]
        and not item["required_stream_missing"]
        and item["timeout_count"] == 0
        and item["capture_error_count"] == 0
        and item["open_error_count"] == 0
        and item["close_error_count"] == 0
        and item["session_opened"]
        and item["session_closed"]
        for item in cameras.values()
    )
    geometry_passed = all(
        item["median_abs_z_m"] <= 0.020
        and item["p95_abs_z_m"] <= 0.040
        and item["maximum_normal_angle_deg"] <= 5.0
        for item in primary["per_camera_plane"].values()
    )
    lifecycle_passed = bool(
        not primary["acquisition"]["workers_alive"]
        and not primary["acquisition"]["worker_errors"]
        and all(item["closed"] for item in primary["acquisition"]["buffers"].values())
        and (
            len(runs) == 1
            or (
                runs[1]["received_frames"] == args.reopen_frames
                and not runs[1]["acquisition"]["workers_alive"]
                and not runs[1]["acquisition"]["worker_errors"]
            )
        )
    )
    throughput_passed = not is_ffs or processed_fps >= 15.0
    latency_p95 = float(primary["timing_ms"]["total_ms"]["p95"])
    latency_passed = not is_ffs or latency_p95 <= 66.8
    gates = {
        "camera_capture": camera_passed,
        "matcher": matcher_passed,
        "geometry": geometry_passed,
        "lifecycle_and_reopen": lifecycle_passed,
        "ffs_processed_fps": throughput_passed,
        "ffs_end_to_end_p95": latency_passed,
    }
    enforced_gates = (
        tuple(gates)
        if args.acceptance_scope == "full"
        else ("camera_capture", "matcher", "lifecycle_and_reopen")
    )
    report = {
        "schema_version": "pointcloud-builder.live-rig-acceptance.v1",
        "snapshot_only": True,
        "acceptance_scope": args.acceptance_scope,
        "enforced_gates": list(enforced_gates),
        "depth_modes": depth_modes,
        "runs": runs,
        "processed_fps": processed_fps,
        "matcher_retention_ratio_passed": bool(matcher["match_ratio"] >= 0.95),
        "gates": gates,
        "passed": bool(all(gates[name] for name in enforced_gates)),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.memory_samples:
        Path(args.memory_samples).parent.mkdir(parents=True, exist_ok=True)
        Path(args.memory_samples).write_text(
            json.dumps({"samples": memory_samples}, indent=2) + "\n", encoding="utf-8"
        )
        _render_memory_timeline(memory_samples, output / "memory_timeline.png")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "acceptance_scope": args.acceptance_scope,
                "enforced_gates": report["enforced_gates"],
                "depth_modes": depth_modes,
                "processed_fps": processed_fps,
                "gates": report["gates"],
                "primary_matcher": {
                    "matched_sets": matcher["matched_sets"],
                    "match_ratio": matcher["match_ratio"],
                    "maximum_absolute_skew_ms": matcher["maximum_absolute_skew_ms"],
                    "absolute_skew_ms": matcher["absolute_skew_ms"],
                    "frame_reuse_violations": matcher["frame_reuse_violations"],
                },
                "primary_per_camera_plane": primary["per_camera_plane"],
                "primary_per_camera_stage": primary["per_camera_stage"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit("live rig acceptance failed")


def _matcher_passed(
    matcher: dict[str, Any],
    *,
    received_frames: int,
    requested_frames: int,
    reference_camera: str,
    require_match_ratio: bool,
) -> bool:
    """Validate delivered matches without conflating latest-biased retention with sync."""

    return bool(
        received_frames == requested_frames
        and matcher["matched_sets"] >= requested_frames
        and (not require_match_ratio or matcher["match_ratio"] >= 0.95)
        and matcher["absolute_skew_ms"][reference_camera]["p95"] == 0.0
        and max(
            float(item["p95"] or 0.0)
            for item in matcher["absolute_skew_ms"].values()
        )
        <= 33.4
        and float(matcher["maximum_absolute_skew_ms"]) <= 66.8
        and matcher["frame_reuse_violations"] == 0
        and matcher["wait_timeouts"] == 0
    )


def _run_once(
    pipeline: Any,
    frames: int,
    plane: ExpectedPlaneRegion,
    output: Path,
    label: str,
    *,
    evidence: bool,
    memory_sample_every: int,
    memory_samples: list[dict[str, int | None]],
    process: psutil.Process,
) -> dict[str, Any]:
    plane_records: dict[str, list[dict[str, Any]]] = {}
    stage_records: dict[str, list[dict[str, Any]]] = {}
    timing_records: list[dict[str, float]] = []
    timeline: list[dict[str, Any]] = []
    selected: dict[str, tuple[float, Any]] = {}
    start = time.perf_counter()
    captured = 0
    try:
        pipeline.acquisition.start()
        for index in range(frames):
            built = pipeline.capture_next()
            captured += 1
            timing_records.append(
                {
                    "match_wait_ms": built.match_wait_ms,
                    "processing_ms": built.processing_ms,
                    "total_ms": built.total_ms,
                }
            )
            timeline.append(
                {
                    "match_index": index,
                    "per_camera_frame_index": {
                        name: envelope.frame_index
                        for name, envelope in built.result.frame_match.envelopes.items()
                    },
                    "per_camera_signed_skew_ms": dict(
                        built.result.frame_match.per_camera_delta_ms
                    ),
                    "match_wait_ms": built.match_wait_ms,
                    "processing_ms": built.processing_ms,
                    "total_ms": built.total_ms,
                }
            )
            frame_score = 0.0
            for name, item in built.result.per_camera_stage_statistics.items():
                stage_records.setdefault(name, []).append(dict(item))
            for item in built.result.per_camera_workspace:
                metrics = evaluate_expected_plane(item.cloud, plane).to_dict()
                board_points = _select_plane_xy(item.cloud.points[:, :3], plane)
                metrics["outlier_ratio"] = float(
                    (torch.abs(board_points[:, 2] - plane.expected_z_m) > 0.040)
                    .float()
                    .mean()
                    .item()
                )
                plane_records.setdefault(item.camera_name, []).append(metrics)
                frame_score = max(frame_score, float(metrics["p95_abs_z_m"]))
            if evidence:
                _retain_evidence(selected, frame_score, built.result, index, frames)
            if memory_sample_every and captured % memory_sample_every == 0:
                memory_samples.append(_memory_sample(captured, process))
    finally:
        pipeline.acquisition.stop()
    duration = time.perf_counter() - start
    if evidence:
        _save_evidence(selected, output, label)
    _render_timelines(timeline, output / f"{label}_capture_match_timeline.png")
    return {
        "requested_frames": frames,
        "received_frames": captured,
        "duration_s": duration,
        "timing_ms": _timing_summary(timing_records),
        "timeline": timeline,
        "per_camera_plane": {
            name: _plane_summary(records) for name, records in plane_records.items()
        },
        "per_camera_stage": {
            name: _stage_summary(records) for name, records in stage_records.items()
        },
        "acquisition": pipeline.acquisition.report(),
    }


def _retain_evidence(
    selected: dict[str, tuple[float, Any]], score: float, result: Any, index: int, total: int
) -> None:
    cpu_clouds = None
    if "best" not in selected or score < selected["best"][0]:
        cpu_clouds = _cpu_snapshot(result)
        selected["best"] = (score, cpu_clouds)
    if "worst" not in selected or score > selected["worst"][0]:
        cpu_clouds = cpu_clouds or _cpu_snapshot(result)
        selected["worst"] = (score, cpu_clouds)
    if index == total // 2:
        selected["median_sequence"] = (score, cpu_clouds or _cpu_snapshot(result))


def _cpu_snapshot(result: Any) -> dict[str, torch.Tensor]:
    return {
        item.camera_name: item.cloud.points.detach().cpu().clone()
        for item in result.per_camera_workspace
    }


def _save_evidence(
    selected: dict[str, tuple[float, dict[str, torch.Tensor]]], output: Path, label: str
) -> None:
    for rank, (_, clouds) in selected.items():
        for camera_name, points in clouds.items():
            save_ascii_ply(points, output / f"{label}_{rank}_{camera_name}.ply")
        _render_colored(clouds, output / f"{label}_{rank}_camera_overlay.png")


def _render_colored(clouds: dict[str, torch.Tensor], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = ("tab:red", "tab:blue", "tab:green", "tab:orange")
    figure = plt.figure(figsize=(9, 7), dpi=140)
    axis = figure.add_subplot(111, projection="3d")
    for (name, tensor), color in zip(sorted(clouds.items()), palette, strict=False):
        points = tensor[:, :3].numpy()
        stride = max(1, points.shape[0] // 8000)
        points = points[::stride]
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.35, alpha=0.35, c=color, label=name)
    axis.set(xlabel="workspace x (m)", ylabel="workspace y (m)", zlabel="workspace z (m)")
    axis.view_init(elev=28, azim=-55)
    axis.legend(markerscale=8)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _render_timelines(records: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indices = [item["match_index"] for item in records]
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), dpi=130, sharex=True)
    camera_names = sorted(records[0]["per_camera_frame_index"])
    for name in camera_names:
        axes[0].plot(
            indices,
            [item["per_camera_frame_index"][name] for item in records],
            linewidth=0.8,
            label=name,
        )
        axes[1].plot(
            indices,
            [item["per_camera_signed_skew_ms"][name] for item in records],
            linewidth=0.8,
            label=name,
        )
    for key in ("match_wait_ms", "processing_ms", "total_ms"):
        axes[2].plot(indices, [item[key] for item in records], linewidth=0.8, label=key)
    axes[0].set_ylabel("capture frame index")
    axes[1].set_ylabel("signed skew (ms)")
    axes[2].set_ylabel("latency (ms)")
    axes[2].set_xlabel("matched frame set")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _render_memory_timeline(records: list[dict[str, int | None]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indices = [int(item["frame_index"] or 0) for item in records]
    figure, axis = plt.subplots(figsize=(11, 5), dpi=130)
    for key in ("rss_bytes", "cuda_allocated_bytes", "cuda_reserved_bytes"):
        axis.plot(
            indices,
            [float(item[key] or 0) / (1024.0**2) for item in records],
            marker=".",
            linewidth=1.0,
            label=key,
        )
    axis.set(xlabel="matched frame set", ylabel="memory (MiB)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _plane_summary(records: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "frame_count": len(records),
        "minimum_point_count": min(int(item["point_count"]) for item in records),
        "median_abs_z_m": statistics.median(float(item["median_abs_z_m"]) for item in records),
        "p95_abs_z_m": _quantile([float(item["p95_abs_z_m"]) for item in records], 0.95),
        "maximum_normal_angle_deg": max(float(item["normal_angle_to_expected_deg"]) for item in records),
        "rmse_m": statistics.median(float(item["rmse_m"]) for item in records),
        "outlier_ratio": statistics.median(
            float(item["outlier_ratio"]) for item in records
        ),
    }


def _select_plane_xy(points: torch.Tensor, plane: ExpectedPlaneRegion) -> torch.Tensor:
    return points[
        (points[:, 0] >= plane.x[0])
        & (points[:, 0] <= plane.x[1])
        & (points[:, 1] >= plane.y[0])
        & (points[:, 1] <= plane.y[1])
    ]


def _stage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid_depth = [float(item["valid_depth_ratio"]) for item in records]
    valid_disparity = [
        float(item["valid_disparity_ratio"])
        for item in records
        if item["valid_disparity_ratio"] is not None
    ]
    backends = sorted(
        {str(item["ffs_backend"]) for item in records if item["ffs_backend"] is not None}
    )
    return {
        "frame_count": len(records),
        "depth_mode": records[0]["depth_mode"],
        "ffs_backends": backends,
        "valid_depth_ratio": _summary(valid_depth),
        "valid_disparity_ratio": _summary(valid_disparity) if valid_disparity else None,
    }


def _timing_summary(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: _summary([record[key] for record in records])
        for key in records[0]
    }


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "p50": statistics.median(values),
        "p95": _quantile(values, 0.95),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _plane(raw: dict[str, Any], frame: str) -> ExpectedPlaneRegion:
    return ExpectedPlaneRegion(
        frame=frame,
        x=tuple(float(value) for value in raw["x"]),
        y=tuple(float(value) for value in raw["y"]),
        expected_z_m=float(raw.get("expected_z_m", 0.0)),
        z_search_range_m=tuple(float(value) for value in raw["z_search_range_m"]),
    )


def _memory_sample(frame_index: int, process: psutil.Process) -> dict[str, int | None]:
    return {
        "frame_index": frame_index,
        "rss_bytes": int(process.memory_info().rss),
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0,
    }


if __name__ == "__main__":
    main()
