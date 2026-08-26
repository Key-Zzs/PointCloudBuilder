#!/usr/bin/env python3
"""Generate and validate the deterministic M5 three-camera rig scene."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import torch

from pointcloud_builder.rig import build_synthetic_rig, create_synthetic_scene, parse_rig_config
from pointcloud_builder.visualization import save_ascii_ply


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    report_path = Path(args.report)
    output.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    names = ("camera_a", "camera_b", "camera_c")
    scene = create_synthetic_scene(
        names,
        timestamp_offsets_ns={"camera_b": 4_000_000, "camera_c": -3_000_000},
    )
    runs = {}
    for count in (1, 2, 3):
        active = names[:count]
        started = time.perf_counter()
        result = build_synthetic_rig(parse_rig_config(_config(active)), scene).build(1)
        runs[str(count)] = {
            "canonical_camera_order": list(result.canonical_camera_order),
            "pre_sampling_count": result.sampled.metadata[
                "pre_sampling_count"
            ],
            "sampled_count": int(result.sampled.points.shape[0]),
            "geometry": _geometry(result),
            "timing_report_ms": result.timing_report_ms,
            "wall_time_ms": (time.perf_counter() - started) * 1000.0,
        }
    config = _config(names)
    forward = build_synthetic_rig(parse_rig_config(config), scene).build(1)
    reversed_config = deepcopy(config)
    reversed_config["cameras"] = list(reversed(reversed_config["cameras"]))
    reverse = build_synthetic_rig(parse_rig_config(reversed_config), scene).build(1)
    order_invariant = bool(
        forward.canonical_camera_order == reverse.canonical_camera_order
        and forward.sampled.metadata == reverse.sampled.metadata
        and torch.equal(
            forward.sampled.points,
            reverse.sampled.points,
        )
    )
    nearest_config = _config(names)
    nearest_config["timing"]["mode"] = "nearest_host_timestamp"
    nearest = build_synthetic_rig(parse_rig_config(nearest_config), scene).build(1)
    geometry = _geometry(forward)
    save_ascii_ply(
        forward.workspace_cropped.points,
        output / "concatenated_workspace.ply",
    )
    _render_colored(forward, output / "three_camera_workspace.png")
    report = {
        "schema_version": "pointcloud-builder.rig-acceptance.v1",
        "scene": "analytic_plane_box_v1",
        "runs": runs,
        "order_invariant": order_invariant,
        "nearest_host_timestamp": {
            "reference_camera": nearest.frame_match.reference_camera,
            "per_camera_delta_ms": nearest.frame_match.per_camera_delta_ms,
            "maximum_skew_ms": nearest.frame_match.maximum_skew_ms,
            "unmatched_cameras": list(nearest.frame_match.unmatched_cameras),
        },
        "per_camera_provenance": forward.per_camera_provenance,
        "passed": bool(
            order_invariant
            and nearest.frame_match.maximum_skew_ms <= 5.0
            and not nearest.frame_match.unmatched_cameras
            and all(item["passed"] for item in geometry.values())
            and all(run["sampled_count"] == 1024 for run in runs.values())
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("synthetic rig acceptance failed")


def _config(names: tuple[str, ...]) -> dict:
    return {
        "schema_version": "pointcloud-builder.rig.v1",
        "output_frame": "workspace",
        "cameras": [
            {
                "name": name,
                "enabled": True,
                "source": {
                    "type": "synthetic",
                    "capture_artifact": f"synthetic://{name}",
                    "provision_artifact": f"synthetic://{name}/bundle",
                },
                "depth": {"mode": "native"},
                "pipeline_config": None,
                "local_crop": {"enabled": False},
            }
            for name in names
        ],
        "timing": {
            "mode": "exact_index",
            "reference_camera": "camera_a",
            "maximum_skew_ms": 5.0,
        },
        "workspace_crop": {
            "enabled": True,
            "x": [-0.72, 0.72],
            "y": [-0.58, 0.58],
            "z": [-0.01, 0.55],
        },
        "fusion": {"enabled": False},
        "sampling": {
            "enabled": True,
            "mode": "voxel_fps",
            "num_points": 1024,
            "voxel_size": 0.01,
            "deterministic": True,
            "seed": 17,
        },
    }


def _geometry(result) -> dict[str, dict[str, float | int | bool]]:
    metrics = {}
    for item in result.per_camera_workspace_clouds:
        points = item.cloud.points[:, :3]
        plane = points[points[:, 2].abs() < 0.004]
        top = points[
            (points[:, 0].abs() < 0.16)
            & (points[:, 1].abs() < 0.10)
            & (points[:, 2] > 0.20)
        ]
        known_plane = torch.tensor((0.50, 0.40, 0.0), dtype=points.dtype)
        known_box = torch.tensor((0.0, 0.0, 0.25), dtype=points.dtype)
        plane_error = float(torch.linalg.norm(plane - known_plane, dim=1).min())
        box_error = float(torch.linalg.norm(top - known_box, dim=1).min())
        plane_z = float(plane[:, 2].abs().median())
        box_top_z = float(top[:, 2].median())
        metrics[item.camera_name] = {
            "point_count": int(points.shape[0]),
            "plane_point_count": int(plane.shape[0]),
            "box_top_point_count": int(top.shape[0]),
            "plane_median_abs_z_m": plane_z,
            "box_top_median_z_m": box_top_z,
            "known_plane_point_error_m": plane_error,
            "known_box_point_error_m": box_error,
            "passed": bool(
                plane.shape[0] > 500
                and top.shape[0] > 20
                and plane_z < 0.0015
                and abs(box_top_z - 0.25) < 0.002
                and plane_error < 0.035
                and box_error < 0.035
            ),
        }
    return metrics


def _render_colored(result, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"camera_a": "red", "camera_b": "blue", "camera_c": "green"}
    figure = plt.figure(figsize=(9, 7), dpi=150)
    axis = figure.add_subplot(111, projection="3d")
    for item in result.per_camera_workspace_clouds:
        points = item.cloud.points[:, :3].detach().cpu().numpy()[::4]
        axis.scatter(
            points[:, 0], points[:, 1], points[:, 2], s=0.3, alpha=0.28,
            c=colors[item.camera_name], label=item.camera_name,
        )
    axis.set(xlabel="workspace x (m)", ylabel="workspace y (m)", zlabel="workspace z (m)")
    axis.set_box_aspect((1.44, 1.16, 0.56))
    axis.view_init(elev=25, azim=-60)
    axis.legend(markerscale=12)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


if __name__ == "__main__":
    main()
