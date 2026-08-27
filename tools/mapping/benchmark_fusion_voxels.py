#!/usr/bin/env python3
"""Benchmark voxel fusion candidates on one frozen live matched-set capture."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
import torch
import yaml

from pointcloud_builder.fusion import (
    board_surface_metrics,
    detect_cube,
    fusion_geometry_metrics,
    symmetric_overlap_metrics,
    voxel_fuse_workspace_clouds,
)
from pointcloud_builder.rig import WorkspaceCloud, build_live_rig, load_rig_config
from pointcloud_builder.workspace import ExpectedPlaneRegion, WorkspacePointCloud


@dataclass(frozen=True)
class FrozenFusionInput:
    clouds: tuple[WorkspaceCloud, ...]
    concatenated: WorkspacePointCloud
    raw_to_world_cropped_ms: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rig-config", required=True)
    parser.add_argument("--mapping-config", required=True)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--voxel-sizes-mm", type=float, nargs="+", default=(2.5, 5.0, 10.0)
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--rerun-output-dir")
    args = parser.parse_args()
    if args.frames <= 0 or args.warmup < 0:
        raise ValueError("frames must be positive and warmup non-negative")
    if any(not np.isfinite(value) or value <= 0 for value in args.voxel_sizes_mm):
        raise ValueError("voxel sizes must be finite and positive")
    report_path = _private_output(args.report)
    rerun_root = (
        _private_output(args.rerun_output_dir) if args.rerun_output_dir else None
    )
    if report_path.exists() or (rerun_root is not None and rerun_root.exists()):
        raise FileExistsError("benchmark outputs must not already exist")

    config = load_rig_config(args.rig_config)
    capture_config = replace(
        config,
        fusion=replace(config.fusion, enabled=False),
        sampling=replace(config.sampling, enabled=False, mode="fps"),
    )
    plane_raw = yaml.safe_load(
        Path(args.mapping_config).read_text(encoding="utf-8")
    )["expected_plane"]
    plane = ExpectedPlaneRegion(
        frame=config.output_frame,
        x=tuple(float(value) for value in plane_raw["x"]),
        y=tuple(float(value) for value in plane_raw["y"]),
        expected_z_m=float(plane_raw.get("expected_z_m", 0.0)),
        z_search_range_m=tuple(
            float(value) for value in plane_raw["z_search_range_m"]
        ),
    )
    frozen, acquisition = _capture_inputs(
        capture_config, frames=args.frames, warmup=args.warmup
    )
    candidates = {
        f"{value:g}mm": _benchmark_candidate(
            frozen,
            config=replace(
                config,
                fusion=replace(config.fusion, enabled=True, voxel_size_m=value / 1000.0),
                sampling=replace(config.sampling, enabled=False, mode="fps"),
            ),
            plane=plane,
            rerun_path=(
                None if rerun_root is None else rerun_root / f"fusion-{value:g}mm.rrd"
            ),
        )
        for value in args.voxel_sizes_mm
    }
    report = {
        "schema_version": "pointcloud-builder.fusion-voxel-benchmark.v1",
        "input_policy": "one live capture frozen on CPU and replayed identically",
        "input_identical": True,
        "sampling_enabled": False,
        "frame_count": len(frozen),
        "acquisition": acquisition,
        "candidates": candidates,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _capture_inputs(config: Any, *, frames: int, warmup: int):
    pipeline = build_live_rig(config, device="cuda")
    frozen: list[FrozenFusionInput] = []
    try:
        pipeline.acquisition.start()
        for index in range(warmup + frames):
            result = pipeline.capture_next().result
            if index < warmup:
                continue
            timing = result.timing_report_ms["reconstruction"]["processing_only"]
            clouds = tuple(
                WorkspaceCloud(
                    item.camera_name,
                    WorkspacePointCloud(
                        points=item.cloud.points.detach().cpu().clone(),
                        frame=item.cloud.frame,
                        metadata=item.cloud.metadata,
                    ),
                )
                for item in result.per_camera_workspace
            )
            frozen.append(
                FrozenFusionInput(
                    clouds=clouds,
                    concatenated=WorkspacePointCloud(
                        points=result.workspace_cropped.points.detach().cpu().clone(),
                        frame=result.workspace_cropped.frame,
                        metadata=result.workspace_cropped.metadata,
                    ),
                    raw_to_world_cropped_ms=float(
                        timing["stages_ms"]["raw_to_world_cropped_ms"]
                    ),
                )
            )
    finally:
        pipeline.acquisition.stop()
    acquisition = pipeline.acquisition.report()
    if acquisition["workers_alive"] or acquisition["worker_errors"]:
        raise RuntimeError("capture did not close cleanly")
    return tuple(frozen), acquisition


def _benchmark_candidate(
    frozen: tuple[FrozenFusionInput, ...],
    *,
    config: Any,
    plane: ExpectedPlaneRegion,
    rerun_path: Path | None,
) -> dict[str, Any]:
    fusion_ms: list[float] = []
    total_ms: list[float] = []
    input_counts: list[int] = []
    fused_counts: list[int] = []
    completeness: list[float] = []
    board_thickness: list[float] = []
    cube_dimensions: list[list[float]] = []
    cube_failures: list[str] = []
    representative = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for item in frozen:
        clouds = tuple(
            WorkspaceCloud(
                cloud.camera_name,
                WorkspacePointCloud(
                    points=cloud.cloud.points.to("cuda"),
                    frame=cloud.cloud.frame,
                    metadata=cloud.cloud.metadata,
                ),
            )
            for cloud in item.clouds
        )
        before = item.concatenated.points.to("cuda")
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = voxel_fuse_workspace_clouds(clouds, config.fusion)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        fusion_ms.append(elapsed_ms)
        total_ms.append(item.raw_to_world_cropped_ms + elapsed_ms)
        input_counts.append(int(before.shape[0]))
        fused_counts.append(int(result.cloud.points.shape[0]))
        geometry = fusion_geometry_metrics(
            WorkspacePointCloud(points=before, frame=config.output_frame),
            result.cloud,
            board_region=plane,
            voxel_size_m=config.fusion.voxel_size_m,
        )
        completeness.append(float(geometry["completeness_at_1_5_voxels"]))
        board = board_surface_metrics(result.cloud, plane)
        board_thickness.append(float(board["surface_thickness_m"]))
        try:
            cube = detect_cube(
                result.cloud.points,
                board_p95_abs_z_m=float(board["p95_abs_z_m"]),
            )
        except ValueError as error:
            cube_failures.append(str(error))
        else:
            cube_dimensions.append([float(value) for value in cube.dimensions_m])
        representative = (clouds, result.cloud)
    assert representative is not None
    overlap = symmetric_overlap_metrics(
        representative[0][0].cloud.points,
        representative[0][1].cloud.points,
        roi=(plane.x, plane.y, (-0.05, 0.120)),
        voxel_size_m=config.fusion.voxel_size_m,
    )
    fused_cpu = representative[1].points.detach().cpu()
    rgb = fused_cpu[:, 3:6] if fused_cpu.shape[1] == 6 else None
    if rerun_path is not None:
        _write_rerun_cloud(rerun_path, fused_cpu)
    return {
        "voxel_size_m": config.fusion.voxel_size_m,
        "sampling_enabled": False,
        "concatenated_points": _integer_summary(input_counts),
        "fused_points": _integer_summary(fused_counts),
        "point_reduction_ratio": statistics.median(
            1.0 - after / max(before, 1)
            for before, after in zip(input_counts, fused_counts, strict=True)
        ),
        "cross_camera_nn": overlap,
        "board_thickness_m": _summary(board_thickness),
        "known_object_dimensions_m": {
            "successful_frames": len(cube_dimensions),
            "failed_frames": len(cube_failures),
            "median": (
                None
                if not cube_dimensions
                else np.median(np.asarray(cube_dimensions), axis=0).tolist()
            ),
            "failure_reasons": sorted(set(cube_failures)),
        },
        "surface_completeness": _summary(completeness),
        "fusion_latency_ms": _summary(fusion_ms),
        "raw_to_world_fused_ms": _summary(total_ms),
        "gpu_memory": {
            "maximum_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "maximum_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "rgb": {
            "channels": 0 if rgb is None else 3,
            "finite": rgb is None or bool(torch.isfinite(rgb).all()),
            "minimum": None if rgb is None else float(rgb.min()),
            "maximum": None if rgb is None else float(rgb.max()),
            "aggregation": representative[1].metadata["fusion"]["rgb_aggregation"],
        },
        "rerun_record": None if rerun_path is None else str(rerun_path),
    }


def _write_rerun_cloud(path: Path, points: torch.Tensor) -> None:
    import rerun as rr

    path.parent.mkdir(parents=True, exist_ok=True)
    recording = rr.RecordingStream("pointcloud-builder-fusion-voxel-sweep")
    recording.set_sinks(rr.FileSink(path))
    values = points.numpy()
    kwargs = {}
    if values.shape[1] == 6:
        kwargs["colors"] = (np.clip(values[:, 3:6], 0.0, 1.0) * 255).astype(
            np.uint8
        )
    recording.log("/clouds/fused", rr.Points3D(values[:, :3], **kwargs))
    recording.flush(timeout_sec=30.0)
    recording.disconnect()


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "p50": statistics.median(values),
        "p95": float(np.quantile(values, 0.95)),
        "mean": statistics.mean(values),
        "maximum": max(values),
    }


def _integer_summary(values: list[int]) -> dict[str, int | float]:
    return {
        "p50": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _private_output(value: str) -> Path:
    output = Path(value).resolve()
    if not output.is_relative_to((Path.cwd() / ".local").resolve()):
        raise ValueError("benchmark outputs must be written under .local/")
    return output


if __name__ == "__main__":
    main()
