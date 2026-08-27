#!/usr/bin/env python3
"""Evaluate production FFS TSDF and report native as an optional baseline."""

from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path

import torch
import yaml

from pointcloud_builder.fusion import board_surface_metrics, detect_cube
from pointcloud_builder.mapping.artifact import (
    load_tsdf_map_artifact,
    validate_tsdf_map_artifact,
)
from pointcloud_builder.mapping.provenance import validate_production_ffs_provenance
from pointcloud_builder.mapping.recording import validate_rig_depth_recording
from pointcloud_builder.mapping.validation import sha256_file
from pointcloud_builder.mapping.validation import load_json
from pointcloud_builder.workspace import ExpectedPlaneRegion, WorkspacePointCloud


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-map")
    parser.add_argument("--ffs-map", required=True)
    parser.add_argument("--ffs-recording", required=True)
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
    recording = validate_rig_depth_recording(args.ffs_recording)
    backend_provenance = validate_production_ffs_provenance(
        recording.get("backend_provenance"), recording["camera_names"]
    )
    source = load_json(Path(args.ffs_map) / "source_recording.json")
    lineage_passed = source.get("recording_manifest_sha256") == sha256_file(
        Path(args.ffs_recording) / "manifest.json"
    )
    ffs = _evaluate(args.ffs_map, plane, snapshot_board, snapshot_cube)
    ffs["gates"]["production_backend_provenance"] = True
    ffs["gates"]["source_recording_lineage"] = lineage_passed
    ffs["passed"] = all(ffs["gates"].values())
    maps = {"ffs_stereo": ffs}
    if args.native_map is not None:
        maps["native"] = _evaluate(
            args.native_map, plane, snapshot_board, snapshot_cube
        )
    native_status = (
        "NOT_RUN"
        if args.native_map is None
        else ("PASS" if maps["native"]["passed"] else "DEGRADED_GEOMETRY")
    )
    report = {
        "schema_version": "pointcloud-builder.real-tsdf-acceptance.v2",
        "snapshot_comparison": {
            "board": snapshot_board,
            "cube": snapshot_cube,
        },
        "maps": maps,
        "production_depth_source": "ffs_stereo",
        "recommended_production_backend": "tensorrt_plugin",
        "production_backend_provenance": backend_provenance,
        "native_tsdf": {
            "role": "optional_baseline",
            "production_required": False,
            "status": native_status,
            "historical_board_thickness_mm": 3.342,
        },
        "production_status": "PASS" if maps["ffs_stereo"]["passed"] else "FAIL",
        "passed": bool(maps["ffs_stereo"]["passed"]),
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
            points=torch.from_numpy(extraction.raw_points.copy()),
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
    postprocess = mapper.config.postprocess
    cropped = extraction.cropped_points
    sampled = extraction.sampled_points
    crop_is_raw_subset = _rows_are_exact_subset(cropped, extraction.raw_points)
    sample_is_crop_subset = _rows_are_exact_subset(sampled, cropped)
    crop_bounds_satisfied = True
    if postprocess.crop.enabled and len(cropped):
        lower = np.asarray(
            [postprocess.crop.x[0], postprocess.crop.y[0], postprocess.crop.z[0]]
        )
        upper = np.asarray(
            [postprocess.crop.x[1], postprocess.crop.y[1], postprocess.crop.z[1]]
        )
        crop_bounds_satisfied = bool(
            np.all(cropped >= lower) and np.all(cropped <= upper)
        )
    postprocess_gates = {
        "workspace_frame": manifest["workspace_frame"] == postprocess.crop.frame,
        "crop_bounds": crop_bounds_satisfied,
        "sampling_target_count": (
            not postprocess.sampling.enabled
            or len(sampled) == postprocess.sampling.num_points
        ),
        "finite": bool(np.isfinite(cropped).all() and np.isfinite(sampled).all()),
        # Cropping and sampling are selection-only operations.  Exact row
        # membership proves that neither stage scaled or otherwise transformed
        # the workspace coordinates.
        "unit_preserved_meters": crop_is_raw_subset and sample_is_crop_subset,
    }
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
        "postprocess": all(postprocess_gates.values()),
    }
    return {
        "board": board,
        "cube": cube,
        "cube_error": cube_error,
        "snapshot_board_thickness_m": snapshot_board["surface_thickness_m"],
        "snapshot_cube_dimensions_m": snapshot_cube["dimensions_m"],
        "point_count": extraction.point_count,
        "raw_point_count": extraction.raw_point_count,
        "cropped_point_count": extraction.cropped_point_count,
        "sampled_point_count": extraction.sampled_point_count,
        "triangle_count": extraction.triangle_count,
        "active_blocks": metrics["volume"]["active_block_count"],
        "integration_latency_ms": metrics["integration"]["latency_ms"],
        "extraction_ms": metrics["extraction"]["extraction_ms"],
        "extraction_timing_ms": {
            "extract_point_cloud_ms": extraction.extract_point_cloud_ms,
            "extract_mesh_ms": extraction.extract_mesh_ms,
            "post_crop_ms": extraction.post_crop_ms,
            "post_sampling_ms": extraction.post_sampling_ms,
            "map_to_raw_cloud_ms": extraction.extract_raw_world_cloud_ms,
            "map_to_cropped_cloud_ms": extraction.extract_cropped_world_cloud_ms,
            "map_to_sampled_cloud_ms": extraction.extract_sampled_world_cloud_ms,
        },
        "postprocess": {
            "frame": manifest["workspace_frame"],
            "crop": {
                "enabled": postprocess.crop.enabled,
                "frame": postprocess.crop.frame,
                "x": list(postprocess.crop.x) if postprocess.crop.enabled else None,
                "y": list(postprocess.crop.y) if postprocess.crop.enabled else None,
                "z": list(postprocess.crop.z) if postprocess.crop.enabled else None,
            },
            "sampling": postprocess.sampling.__dict__,
            "gates": postprocess_gates,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def _rows_are_exact_subset(candidate: np.ndarray, source: np.ndarray) -> bool:
    """Return whether every candidate XYZ row occurs exactly in source."""

    if candidate.ndim != 2 or source.ndim != 2:
        return False
    if candidate.shape[1:] != source.shape[1:]:
        return False
    if not len(candidate):
        return True
    if not len(source):
        return False
    candidate_rows = np.ascontiguousarray(candidate).view(
        np.dtype((np.void, candidate.dtype.itemsize * candidate.shape[1]))
    )
    source_rows = np.ascontiguousarray(source).view(
        np.dtype((np.void, source.dtype.itemsize * source.shape[1]))
    )
    return bool(np.isin(candidate_rows, source_rows).all())


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
