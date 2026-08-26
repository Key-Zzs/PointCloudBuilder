#!/usr/bin/env python3
"""Run one independent real multi-camera snapshot-fusion acceptance session."""

from __future__ import annotations

import argparse
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
    contribution_metrics,
    cube_box_voxel_count,
    detect_cube,
    fusion_geometry_metrics,
    symmetric_overlap_metrics,
    voxel_centroids,
)
from pointcloud_builder.rig import build_live_rig, load_rig_config
from pointcloud_builder.visualization import save_ascii_ply
from pointcloud_builder.workspace import ExpectedPlaneRegion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rig-config", required=True)
    parser.add_argument("--mapping-config", required=True)
    parser.add_argument("--matched-sets", type=int, default=300)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if args.matched_sets <= 0:
        raise ValueError("--matched-sets must be positive")
    config = load_rig_config(args.rig_config)
    if not config.fusion.enabled or config.fusion.voxel_size_m != 0.005:
        raise ValueError("formal fusion requires enabled 5 mm voxel fusion")
    if not config.sampling.enabled or config.sampling.num_points != 4096:
        raise ValueError("formal fusion requires one enabled 4096-point global sampler")
    if config.timing.maximum_skew_ms != 33.4:
        raise ValueError("formal fusion requires maximum_skew_ms=33.4")
    mapping = yaml.safe_load(Path(args.mapping_config).read_text(encoding="utf-8"))
    board = _plane(mapping["expected_plane"], config.output_frame)
    output = Path(args.output)
    report_path = Path(args.report)
    output.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = build_live_rig(config, device="cuda")
    board_records: dict[str, list[dict[str, Any]]] = {}
    stage_records: dict[str, list[dict[str, Any]]] = {}
    latencies: list[float] = []
    evidence_indices = sorted(
        {round(index * (args.matched_sets - 1) / 4) for index in range(5)}
    )
    retained: dict[int, Any] = {}
    started = time.perf_counter()
    try:
        pipeline.acquisition.start()
        for index in range(args.matched_sets):
            built = pipeline.capture_next()
            latencies.append(built.total_ms)
            result = built.result
            for item in result.per_camera_workspace:
                board_records.setdefault(item.camera_name, []).append(
                    board_surface_metrics(item.cloud, board)
                )
            board_records.setdefault("concatenated", []).append(
                board_surface_metrics(result.workspace_cropped, board)
            )
            board_records.setdefault("fused", []).append(
                board_surface_metrics(result.fused, board)
            )
            for name, item in result.per_camera_stage_statistics.items():
                stage_records.setdefault(name, []).append(dict(item))
            if index in evidence_indices:
                retained[index] = result
    finally:
        pipeline.acquisition.stop()
    duration_s = time.perf_counter() - started
    if set(retained) != set(evidence_indices):
        raise RuntimeError("formal fusion did not retain all evidence snapshots")
    snapshots = [
        {"frame_index": index, **_snapshot_metrics(retained[index], board)}
        for index in evidence_indices
    ]
    selected_metrics = min(
        snapshots, key=lambda item: abs(item["frame_index"] - args.matched_sets // 2)
    )
    selected = retained[int(selected_metrics["frame_index"])]
    detailed_board = selected_metrics["board"]
    overlap = selected_metrics["overlap"]
    cubes = selected_metrics["cube"]
    contribution = selected_metrics["contribution"]
    geometry = selected_metrics["fusion_geometry"]
    stage_summary = {
        name: {
            "depth_mode": records[0]["depth_mode"],
            "ffs_backends": sorted(
                {str(item["ffs_backend"]) for item in records if item["ffs_backend"]}
            ),
            "valid_depth_ratio_p50": statistics.median(
                float(item["valid_depth_ratio"]) for item in records
            ),
            "valid_disparity_ratio_p50": statistics.median(
                float(item["valid_disparity_ratio"]) for item in records
                if item["valid_disparity_ratio"] is not None
            ),
        }
        for name, records in stage_records.items()
    }
    acquisition = pipeline.acquisition.report()
    matcher = acquisition["matcher"]
    board_summary = {
        name: _board_summary(records) for name, records in board_records.items()
    }
    backend_passed = all(
        item["ffs_backends"] == ["tensorrt_plugin"] for item in stage_summary.values()
    )
    no_early_sampling = all(
        int(item["camera_raw_point_count"]) == int(item["camera_sampled_point_count"])
        for records in stage_records.values()
        for item in records
    )
    processing_order_passed = bool(
        selected.processing_metadata["workspace_crop_stage"] == "after_concatenation"
        and selected.processing_metadata["fusion_input_stage"]
        == "per_camera_workspace_cropped"
        and selected.processing_metadata["global_sampling_input_stage"] == "fused"
    )
    matcher_passed = bool(
        max(
            float(item["p95"] or 0.0)
            for item in matcher["absolute_skew_ms"].values()
        )
        <= 33.4
        and matcher["maximum_absolute_skew_ms"] <= 66.8
        and matcher["frame_reuse_violations"] == 0
    )
    throughput_fps = args.matched_sets / duration_s
    gates = {
        "matcher_integrity": matcher_passed,
        "per_camera_board": all(item["gates"]["per_camera_board"] for item in snapshots),
        "fused_board_shift": all(item["gates"]["fused_board_shift"] for item in snapshots),
        "cross_camera_overlap": all(item["gates"]["cross_camera_overlap"] for item in snapshots),
        "cube_evidence_snapshots": all(item["gates"]["cube"] for item in snapshots),
        "cube_same_physical_candidate": all(
            item["gates"]["cube_same_physical_candidate"] for item in snapshots
        ),
        "cube_joint_camera_observation": all(
            item["gates"]["cube_joint_camera_observation"] for item in snapshots
        ),
        "cube_fused_degradation": all(
            item["gates"]["cube_fused_degradation"] for item in snapshots
        ),
        "fusion_contribution": all(item["gates"]["fusion_contribution"] for item in snapshots),
        "fusion_thickness": all(item["gates"]["fusion_thickness"] for item in snapshots),
        "global_sampling": selected.sampled.points.shape[0] == 4096,
        "no_per_camera_early_sampling": no_early_sampling,
        "processing_order": processing_order_passed,
        "deterministic_voxel_fusion": config.fusion.deterministic,
        "tensorrt_plugin": backend_passed,
        "worker_cleanup": not acquisition["workers_alive"]
        and not acquisition["worker_errors"],
    }
    _save_evidence(selected, cubes["fused"], board, output)
    repeatability_projection = {
        "cube_dimensions_m": [
            statistics.median(item["cube"]["fused"]["dimensions_m"][axis] for item in snapshots)
            for axis in range(3)
        ],
        "board_median_abs_z_m": board_summary["fused"]["median_abs_z_m"],
        "overlap_symmetric_median_m": statistics.median(
            item["overlap"]["symmetric"]["median_m"] for item in snapshots
        ),
        "worker_fatal_error": bool(acquisition["worker_errors"]),
    }
    report = {
        "schema_version": "pointcloud-builder.real-multicamera-fusion.v1",
        "snapshot_only": True,
        "persistent_mapping": False,
        "matched_sets": args.matched_sets,
        "duration_s": duration_s,
        "throughput_fps": throughput_fps,
        "latency_ms": _summary(latencies),
        "performance_diagnostics": {
            "throughput_target_15hz_met": throughput_fps >= 15.0,
            "end_to_end_p95_66_8ms_met": float(np.quantile(latencies, 0.95))
            <= 66.8,
        },
        "acquisition": acquisition,
        "stage_summary": stage_summary,
        "board_summary": board_summary,
        "evidence_snapshot_count": len(snapshots),
        "evidence_snapshots": snapshots,
        "selected_snapshot": {
            "frame_index": selected_metrics["frame_index"],
            "board": detailed_board,
            "overlap": overlap,
            "cube": cubes,
            "contribution": contribution,
            "fusion_geometry": geometry,
            "global_sampled_count": int(selected.sampled.points.shape[0]),
        },
        "repeatability_projection": repeatability_projection,
        "gates": gates,
        "passed": all(gates.values()),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "matched_sets": args.matched_sets,
                "throughput_fps": throughput_fps,
                "latency_p95_ms": report["latency_ms"]["p95"],
                "gates": gates,
                "overlap": overlap["symmetric"],
                "cube": cubes,
                "board": detailed_board,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit("real multi-camera fusion acceptance failed")


def _snapshot_metrics(result: Any, board: ExpectedPlaneRegion) -> dict[str, Any]:
    per_camera = {item.camera_name: item.cloud for item in result.per_camera_workspace}
    clouds = {
        **per_camera,
        "concatenated": result.workspace_cropped,
        "fused": result.fused,
    }
    detailed_board = {
        name: board_surface_metrics(cloud, board) for name, cloud in clouds.items()
    }
    overlap = symmetric_overlap_metrics(
        per_camera["camera_a"].points,
        per_camera["camera_b"].points,
        roi=(board.x, board.y, (-0.05, 0.120)),
        voxel_size_m=0.005,
    )
    cubes = {
        name: detect_cube(
            cloud.points,
            board_p95_abs_z_m=float(detailed_board[name]["p95_abs_z_m"]),
        ).to_dict()
        for name, cloud in clouds.items()
    }
    fused_center = np.asarray(cubes["fused"]["center_workspace_m"], dtype=np.float64)
    center_deltas = {
        name: {
            "xy_m": float(
                np.linalg.norm(
                    np.asarray(cubes[name]["center_workspace_m"][:2])
                    - fused_center[:2]
                )
            ),
            "z_m": abs(float(cubes[name]["center_workspace_m"][2]) - fused_center[2]),
        }
        for name in ("camera_a", "camera_b", "concatenated")
    }
    same_candidate = all(
        item["xy_m"] <= 0.040 and item["z_m"] <= 0.020
        for item in center_deltas.values()
    )
    cube_camera_voxels = {
        name: cube_box_voxel_count(
            per_camera[name].points,
            cubes["fused"],
            minimum_z_m=max(0.010, 2.0 * float(detailed_board[name]["p95_abs_z_m"])),
        )
        for name in ("camera_a", "camera_b")
    }
    joint_observation = all(count >= 20 for count in cube_camera_voxels.values())
    best_single_error = min(
        cubes["camera_a"]["mean_absolute_dimension_error_m"],
        cubes["camera_b"]["mean_absolute_dimension_error_m"],
    )
    cube_degradation = bool(
        cubes["fused"]["mean_absolute_dimension_error_m"]
        <= best_single_error + 0.005
    )
    contribution = contribution_metrics(result.fusion_provenance)
    geometry = fusion_geometry_metrics(
        result.workspace_cropped,
        result.fused,
        board_region=board,
        voxel_size_m=0.005,
    )
    return {
        "board": detailed_board,
        "overlap": overlap,
        "cube": cubes,
        "cube_identity": {
            "center_deltas_from_fused": center_deltas,
            "per_camera_above_plane_voxels_in_fused_box": cube_camera_voxels,
        },
        "contribution": contribution,
        "fusion_geometry": geometry,
        "gates": {
            "per_camera_board": all(
                detailed_board[name]["median_abs_z_m"] <= 0.020
                and detailed_board[name]["p95_abs_z_m"] <= 0.040
                for name in ("camera_a", "camera_b")
            ),
            "fused_board_shift": geometry["board_shift_gate_passed"],
            "cross_camera_overlap": overlap["target_passed"],
            "cube": all(item["passed"] for item in cubes.values()),
            "cube_same_physical_candidate": same_candidate,
            "cube_joint_camera_observation": joint_observation,
            "cube_fused_degradation": cube_degradation,
            "fusion_contribution": contribution["passed"],
            "fusion_thickness": geometry["thickness_gate_passed"],
        },
    }


def _board_summary(records: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "frame_count": len(records),
        "minimum_point_count": min(int(item["point_count"]) for item in records),
        "median_abs_z_m": statistics.median(float(item["median_abs_z_m"]) for item in records),
        "p95_abs_z_m": float(np.quantile([item["p95_abs_z_m"] for item in records], 0.95)),
        "rmse_m": statistics.median(float(item["rmse_m"]) for item in records),
        "maximum_normal_angle_deg": max(
            float(item["normal_angle_to_expected_deg"]) for item in records
        ),
        "surface_thickness_m": statistics.median(
            float(item["surface_thickness_m"]) for item in records
        ),
        "outlier_ratio": statistics.median(float(item["outlier_ratio"]) for item in records),
    }


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "p50": statistics.median(values),
        "p95": float(np.quantile(values, 0.95)),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def _save_evidence(result: Any, cube: dict[str, Any], board: ExpectedPlaneRegion, output: Path) -> None:
    clouds = {item.camera_name: item.cloud.points.detach().cpu() for item in result.per_camera_workspace}
    save_ascii_ply(clouds["camera_a"], output / "camera_a_workspace.ply")
    save_ascii_ply(clouds["camera_b"], output / "camera_b_workspace.ply")
    colored = torch.cat(
        (
            torch.cat((clouds["camera_a"][:, :3], torch.tensor((1.0, 0.0, 0.0)).repeat(clouds["camera_a"].shape[0], 1)), dim=1),
            torch.cat((clouds["camera_b"][:, :3], torch.tensor((0.0, 0.0, 1.0)).repeat(clouds["camera_b"].shape[0], 1)), dim=1),
        )
    )
    save_ascii_ply(colored, output / "two_camera_colored_concatenation.ply")
    save_ascii_ply(result.fused.points.detach().cpu(), output / "fused_workspace.ply")
    save_ascii_ply(result.sampled.points.detach().cpu(), output / "globally_sampled.ply")
    _render_single(
        clouds["camera_a"], cube, output / "camera_a_only_views.png", "camera_a"
    )
    _render_single(
        clouds["camera_b"], cube, output / "camera_b_only_views.png", "camera_b"
    )
    _render_views(clouds, cube, output / "camera_colored_views.png")
    _render_single(result.fused.points.detach().cpu(), cube, output / "fused_views.png", "fused")
    _render_single(result.sampled.points.detach().cpu(), cube, output / "sampled_views.png", "global sampled")
    _render_board_residual(result.fused.points.detach().cpu(), board, output / "board_residual_heatmap.png")
    _render_cross_distance(clouds, board, output / "cross_camera_distance_heatmap.png")


def _render_views(clouds: dict[str, torch.Tensor], cube: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(16, 5), dpi=140)
    views = ((90, -90, "top"), (22, -60, "oblique"), (5, 0, "side"))
    for index, (elev, azim, title) in enumerate(views, 1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        for name, color in (("camera_a", "red"), ("camera_b", "blue")):
            points = clouds[name][:, :3].numpy()
            points = points[:: max(1, points.shape[0] // 9000)]
            axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.25, alpha=0.3, c=color, label=name)
        _draw_cube_box(axis, cube)
        axis.view_init(elev=elev, azim=azim)
        axis.set_title(title)
        axis.set(xlabel="x", ylabel="y", zlabel="z")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _render_single(points: torch.Tensor, cube: dict[str, Any], path: Path, label: str) -> None:
    _render_views({"camera_a": points, "camera_b": points[:0]}, cube, path)


def _draw_cube_box(axis: Any, cube: dict[str, Any]) -> None:
    center = np.asarray(cube["center_workspace_m"], dtype=np.float64)
    dims = np.asarray(cube["dimensions_m"], dtype=np.float64)
    yaw = np.deg2rad(float(cube["yaw_deg"]))
    rotation = np.array(
        ((np.cos(yaw), -np.sin(yaw)), (np.sin(yaw), np.cos(yaw)))
    )
    corners = []
    for u in (-dims[0] / 2.0, dims[0] / 2.0):
        for v in (-dims[1] / 2.0, dims[1] / 2.0):
            xy = center[:2] + np.array((u, v)) @ rotation.T
            for z in (center[2] - dims[2] / 2.0, center[2] + dims[2] / 2.0):
                corners.append((xy[0], xy[1], z))
    corners = np.asarray(corners)
    # Corner indices are binary (u, v, z).  Adjacent corners differ in exactly
    # one bit; testing workspace coordinate equality is invalid after XY rotation.
    for left in range(8):
        for bit in (1, 2, 4):
            right = left ^ bit
            if left < right:
                axis.plot(
                    *zip(corners[left], corners[right], strict=True),
                    c="lime",
                    linewidth=1.5,
                )


def _render_board_residual(points: torch.Tensor, board: ExpectedPlaneRegion, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = points[:, :3].numpy()
    mask = (
        (xyz[:, 0] >= board.x[0]) & (xyz[:, 0] <= board.x[1])
        & (xyz[:, 1] >= board.y[0]) & (xyz[:, 1] <= board.y[1])
        & (xyz[:, 2] >= board.z_search_range_m[0]) & (xyz[:, 2] <= board.z_search_range_m[1])
    )
    xyz = xyz[mask]
    figure, axis = plt.subplots(figsize=(8, 6), dpi=140)
    image = axis.scatter(xyz[:, 0], xyz[:, 1], c=np.abs(xyz[:, 2]), s=1.0, cmap="magma", vmin=0.0, vmax=0.04)
    figure.colorbar(image, ax=axis, label="|z| residual (m)")
    axis.set(xlabel="workspace x (m)", ylabel="workspace y (m)", title="board residual")
    axis.set_aspect("equal")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _render_cross_distance(clouds: dict[str, torch.Tensor], board: ExpectedPlaneRegion, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = voxel_centroids(clouds["camera_a"][:, :3], voxel_size_m=0.005)
    b = voxel_centroids(clouds["camera_b"][:, :3], voxel_size_m=0.005)
    def roi(value: torch.Tensor) -> torch.Tensor:
        return value[
            (value[:, 0] >= board.x[0])
            & (value[:, 0] <= board.x[1])
            & (value[:, 1] >= board.y[0])
            & (value[:, 1] <= board.y[1])
            & (value[:, 2] >= -0.05)
            & (value[:, 2] <= 0.12)
        ]
    a, b = roi(a), roi(b)
    distances = torch.cat([torch.cdist(a[start:start + 2048], b).min(dim=1).values for start in range(0, a.shape[0], 2048)]).numpy()
    xyz = a.numpy()
    figure, axis = plt.subplots(figsize=(8, 6), dpi=140)
    image = axis.scatter(xyz[:, 0], xyz[:, 1], c=distances, s=3.0, cmap="viridis", vmin=0.0, vmax=0.03)
    figure.colorbar(image, ax=axis, label="A to B NN distance (m)")
    axis.set(xlabel="workspace x (m)", ylabel="workspace y (m)", title="cross-camera distance")
    axis.set_aspect("equal")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _plane(raw: dict[str, Any], frame: str) -> ExpectedPlaneRegion:
    return ExpectedPlaneRegion(
        frame=frame,
        x=tuple(float(value) for value in raw["x"]),
        y=tuple(float(value) for value in raw["y"]),
        expected_z_m=float(raw.get("expected_z_m", 0.0)),
        z_search_range_m=tuple(float(value) for value in raw["z_search_range_m"]),
    )


if __name__ == "__main__":
    main()
