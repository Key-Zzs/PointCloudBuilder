#!/usr/bin/env python3
"""Run the bounded preregistered TSDF parameter grid on one depth recording."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
import yaml

from pointcloud_builder.fusion import board_surface_metrics, detect_cube
from pointcloud_builder.mapping.config import load_tsdf_config
from pointcloud_builder.mapping.open3d import Open3dTsdfMap
from pointcloud_builder.mapping.recording import (
    iter_rig_depth_recording,
    validate_rig_depth_recording,
)
from pointcloud_builder.workspace import ExpectedPlaneRegion, WorkspacePointCloud


def main() -> None:
    sweep_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mapping-config", required=True)
    parser.add_argument("--snapshot-report", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report_path = _private_output(args.report)
    if report_path.exists():
        raise FileExistsError(f"TSDF sweep report already exists: {report_path}")
    manifest = validate_rig_depth_recording(args.recording)
    frames = list(iter_rig_depth_recording(args.recording))
    base = load_tsdf_config(args.config)
    mapping = yaml.safe_load(Path(args.mapping_config).read_text(encoding="utf-8"))
    plane = _plane(mapping["expected_plane"], manifest["workspace_frame"])
    snapshot = json.loads(Path(args.snapshot_report).read_text(encoding="utf-8"))
    if not snapshot.get("passed"):
        raise ValueError("TSDF sweep requires a frozen passing snapshot report")
    snapshot_thickness = float(
        snapshot["selected_snapshot"]["board"]["fused"]["surface_thickness_m"]
    )
    results = []
    for voxel_size_m in (0.003, 0.005, 0.010):
        for truncation in (4.0, 8.0):
            for frame_stride in (1, 5, 10):
                config = replace(
                    base,
                    volume=replace(
                        base.volume,
                        voxel_size_m=voxel_size_m,
                        trunc_voxel_multiplier=truncation,
                    ),
                    integration=replace(base.integration, frame_stride=frame_stride),
                )
                mapper = Open3dTsdfMap(
                    config, workspace_frame=manifest["workspace_frame"]
                )
                integrations = []
                try:
                    for frame in frames:
                        value = mapper.integrate(frame)
                        if not value.skipped:
                            integrations.append(value.integration_ms)
                    extraction = mapper.extract()
                    volume = mapper.volume_statistics()
                finally:
                    mapper.close()
                cloud = WorkspacePointCloud(
                    points=torch.from_numpy(extraction.points.copy()),
                    frame=manifest["workspace_frame"],
                )
                board = board_surface_metrics(cloud, plane)
                cube_error = None
                try:
                    cube = detect_cube(
                        cloud.points,
                        board_p95_abs_z_m=float(board["p95_abs_z_m"]),
                    ).to_dict()
                except ValueError as error:
                    cube_error = str(error)
                    cube = {
                        "passed": False,
                        "mean_absolute_dimension_error_m": None,
                    }
                eligible = bool(
                    board["median_abs_z_m"] <= 0.010
                    and board["p95_abs_z_m"] <= 0.020
                    and board["surface_thickness_m"] <= snapshot_thickness + 0.001
                    and cube["passed"]
                )
                geometry_score = (
                    1_000_000.0
                    if cube_error is not None
                    else float(
                        board["surface_thickness_m"]
                        + cube["mean_absolute_dimension_error_m"]
                    )
                )
                resource_penalty = 0.001 * (
                    float(np.quantile(integrations, 0.95)) / 5.0
                    + float(volume["active_block_count"]) / 1500.0
                    + float(extraction.triangle_count) / 350_000.0
                )
                score = geometry_score + resource_penalty
                results.append(
                    {
                        "voxel_size_m": voxel_size_m,
                        "trunc_voxel_multiplier": truncation,
                        "frame_stride": frame_stride,
                        "integrated_frame_sets": len(integrations),
                        "integration_ms": {
                            "p50": statistics.median(integrations),
                            "p95": float(np.quantile(integrations, 0.95)),
                        },
                        "board": board,
                        "snapshot_board_thickness_m": snapshot_thickness,
                        "cube": cube,
                        "cube_error": cube_error,
                        "point_count": extraction.point_count,
                        "triangle_count": extraction.triangle_count,
                        "active_blocks": volume["active_block_count"],
                        "estimated_attribute_bytes": config.estimated_attribute_bytes,
                        "eligible": eligible,
                        "score": score,
                        "geometry_score": geometry_score,
                        "resource_penalty": resource_penalty,
                    }
                )
    ranking = sorted(
        results,
        key=lambda item: (
            not item["eligible"],
            item["score"],
            abs(item["voxel_size_m"] - 0.005),
            item["frame_stride"],
        ),
    )
    report = {
        "schema_version": "pointcloud-builder.tsdf-sweep.v1",
        "bounded_grid": {
            "voxel_size_m": [0.003, 0.005, 0.010],
            "trunc_voxel_multiplier": [4.0, 8.0],
            "frame_stride": [1, 5, 10],
        },
        "candidate_count": len(results),
        "ranking_policy": {
            "eligible_first": True,
            "geometry_score": "board thickness + cube mean dimension error",
            "resource_penalty": "0.001 * (p95_ms/5 + active_blocks/1500 + triangles/350000)",
            "tie_breaker": "closest to 5 mm, then lower frame stride",
        },
        "results": results,
        "ranking": [
            {
                "rank": index + 1,
                "voxel_size_m": item["voxel_size_m"],
                "trunc_voxel_multiplier": item["trunc_voxel_multiplier"],
                "frame_stride": item["frame_stride"],
                "eligible": item["eligible"],
                "score": item["score"],
            }
            for index, item in enumerate(ranking)
        ],
        "selected": ranking[0] if ranking[0]["eligible"] else None,
        "elapsed_s": time.perf_counter() - sweep_started,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["ranking"], indent=2, sort_keys=True))


def _plane(raw: dict, frame: str) -> ExpectedPlaneRegion:
    return ExpectedPlaneRegion(
        frame=frame,
        x=tuple(float(x) for x in raw["x"]),
        y=tuple(float(x) for x in raw["y"]),
        expected_z_m=float(raw.get("expected_z_m", 0.0)),
        z_search_range_m=tuple(float(x) for x in raw["z_search_range_m"]),
    )


def _private_output(value: str) -> Path:
    output = Path(value).resolve()
    if not output.is_relative_to((Path.cwd() / ".local").resolve()):
        raise ValueError("real TSDF sweep reports must be written under .local/")
    return output


if __name__ == "__main__":
    main()
