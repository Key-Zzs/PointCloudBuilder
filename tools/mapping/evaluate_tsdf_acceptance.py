#!/usr/bin/env python3
"""Evaluate real native/FFS TSDF geometry against frozen snapshot metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from pointcloud_builder.fusion import board_surface_metrics, detect_cube
from pointcloud_builder.mapping.artifact import (
    load_tsdf_map_artifact,
    validate_tsdf_map_artifact,
)
from pointcloud_builder.mapping.validation import load_json
from pointcloud_builder.workspace import ExpectedPlaneRegion, WorkspacePointCloud


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-map", required=True)
    parser.add_argument("--ffs-map", required=True)
    parser.add_argument("--snapshot-report", required=True)
    parser.add_argument("--mapping-config", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    output = _private_output(args.report)
    if output.exists():
        raise FileExistsError(f"TSDF acceptance report already exists: {output}")
    snapshot = load_json(args.snapshot_report)
    if not snapshot.get("passed"):
        raise ValueError("snapshot comparison report must already be a frozen PASS")
    mapping = yaml.safe_load(Path(args.mapping_config).read_text(encoding="utf-8"))
    plane = _plane(mapping["expected_plane"])
    snapshot_board = snapshot["selected_snapshot"]["board"]["fused"]
    snapshot_cube = snapshot["selected_snapshot"]["cube"]["fused"]
    maps = {
        "native": _evaluate(args.native_map, plane, snapshot_board, snapshot_cube),
        "ffs_stereo": _evaluate(args.ffs_map, plane, snapshot_board, snapshot_cube),
    }
    report = {
        "schema_version": "pointcloud-builder.real-tsdf-acceptance.v1",
        "snapshot_comparison": {
            "board": snapshot_board,
            "cube": snapshot_cube,
        },
        "maps": maps,
        "passed": all(item["passed"] for item in maps.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("real TSDF acceptance failed")


def _evaluate(
    root: str,
    plane: ExpectedPlaneRegion,
    snapshot_board: dict,
    snapshot_cube: dict,
) -> dict:
    manifest = validate_tsdf_map_artifact(root)
    metrics = load_json(Path(root) / "metrics.json")
    mapper = load_tsdf_map_artifact(root)
    try:
        extraction = mapper.extract()
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
                "dimensions_m": [],
                "mean_absolute_dimension_error_m": None,
                "maximum_dimension_error_m": None,
                "passed": False,
            }
    finally:
        mapper.close()
    dimensions = [float(x) for x in cube["dimensions_m"]]
    gates = {
        "workspace_frame": manifest["workspace_frame"] == plane.frame,
        "board_median": float(board["median_abs_z_m"]) <= 0.010,
        "board_p95": float(board["p95_abs_z_m"]) <= 0.020,
        "thickness_vs_snapshot": float(board["surface_thickness_m"])
        <= float(snapshot_board["surface_thickness_m"]) + 0.001,
        "cube_dimensions": len(dimensions) == 3
        and all(0.055 <= value <= 0.085 for value in dimensions),
        "cube_mean_error": cube_error is None
        and float(cube["mean_absolute_dimension_error_m"]) <= 0.010,
        "cube_maximum_error": cube_error is None
        and float(cube["maximum_dimension_error_m"]) <= 0.015,
        "orientation": float(board["normal_angle_to_expected_deg"]) <= 10.0,
        "save_load_parity": bool(metrics["save_load_parity"]["passed"]),
    }
    return {
        "board": board,
        "cube": cube,
        "cube_error": cube_error,
        "snapshot_board_thickness_m": snapshot_board["surface_thickness_m"],
        "snapshot_cube_dimensions_m": snapshot_cube["dimensions_m"],
        "point_count": extraction.point_count,
        "triangle_count": extraction.triangle_count,
        "active_blocks": metrics["volume"]["active_block_count"],
        "integration_latency_ms": metrics["integration"]["latency_ms"],
        "extraction_ms": metrics["extraction"]["extraction_ms"],
        "gates": gates,
        "passed": all(gates.values()),
    }


def _plane(raw: dict) -> ExpectedPlaneRegion:
    return ExpectedPlaneRegion(
        frame=str(raw.get("frame", "workspace")),
        x=tuple(float(x) for x in raw["x"]),
        y=tuple(float(x) for x in raw["y"]),
        expected_z_m=float(raw.get("expected_z_m", 0.0)),
        z_search_range_m=tuple(float(x) for x in raw["z_search_range_m"]),
    )


def _private_output(value: str) -> Path:
    output = Path(value).resolve()
    if not output.is_relative_to((Path.cwd() / ".local").resolve()):
        raise ValueError("real TSDF acceptance reports must be written under .local/")
    return output


if __name__ == "__main__":
    main()
