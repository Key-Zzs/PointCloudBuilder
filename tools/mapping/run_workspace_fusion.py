#!/usr/bin/env python3
"""Run synthetic multi-camera and real single-camera M6 fusion acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from pointcloud_builder.fusion import synthetic_geometry_metrics
from pointcloud_builder.rig import (
    build_replay_rig,
    build_synthetic_rig,
    create_synthetic_scene,
    parse_rig_config,
)
from pointcloud_builder.visualization import save_ascii_ply
from pointcloud_builder.workspace import ExpectedPlaneRegion, evaluate_expected_plane


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-capture", required=True)
    parser.add_argument("--real-provision", required=True)
    parser.add_argument("--real-mapping-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    report_path = Path(args.report)
    output.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    synthetic = _run_synthetic(output)
    real = _run_real(
        Path(args.real_capture),
        Path(args.real_provision),
        Path(args.real_mapping_config),
        output,
    )
    report = {
        "schema_version": "pointcloud-builder.fusion-acceptance.v1",
        "snapshot_only": True,
        "synthetic": synthetic,
        "real_single_camera": real,
        "passed": bool(synthetic["passed"] and real["passed"]),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("workspace fusion acceptance failed")


def _run_synthetic(output: Path) -> dict:
    names = ("camera_a", "camera_b", "camera_c")
    scene = create_synthetic_scene(names)
    config = parse_rig_config(_synthetic_config(names))
    result = build_synthetic_rig(config, scene).build(1)
    reverse_raw = _synthetic_config(tuple(reversed(names)))
    reverse = build_synthetic_rig(parse_rig_config(reverse_raw), scene).build(1)
    deterministic = bool(
        torch.equal(result.fused.points, reverse.fused.points)
        and torch.equal(result.sampled.points, reverse.sampled.points)
    )
    metrics = synthetic_geometry_metrics(
        result.workspace_cropped.points,
        result.fused.points,
        voxel_size_m=config.fusion.voxel_size_m,
    )
    colors = {
        "camera_a": (1.0, 0.0, 0.0),
        "camera_b": (0.0, 0.0, 1.0),
        "camera_c": (0.0, 1.0, 0.0),
    }
    colored = torch.cat(
        [
            torch.cat(
                (
                    item.cloud.points[:, :3],
                    torch.tensor(colors[item.camera_name], dtype=item.cloud.points.dtype).repeat(
                        item.cloud.points.shape[0], 1
                    ),
                ),
                dim=1,
            )
            for item in result.per_camera_workspace
        ],
        dim=0,
    )
    save_ascii_ply(colored, output / "synthetic_per_camera_colored.ply")
    save_ascii_ply(result.workspace_cropped.points, output / "synthetic_concatenated.ply")
    save_ascii_ply(result.fused.points, output / "synthetic_fused.ply")
    save_ascii_ply(result.sampled.points, output / "synthetic_sampled.ply")
    _render_before_after(result, output / "synthetic_fusion_before_after.png")
    provenance = result.fusion_provenance
    assert provenance is not None
    return {
        "camera_order": list(result.canonical_camera_order),
        "fusion_input_matches_sum_per_camera": provenance.input_point_count
        == sum(item.cloud.points.shape[0] for item in result.per_camera_workspace),
        "deterministic_under_camera_permutation": deterministic,
        "metrics": metrics,
        "provenance": provenance.to_summary(),
        "sampled_count": int(result.sampled.points.shape[0]),
        "passed": bool(
            deterministic
            and provenance.input_point_count
            == sum(item.cloud.points.shape[0] for item in result.per_camera_workspace)
            and metrics["duplicate_surface_thickness_after_m"]
            <= metrics["duplicate_surface_thickness_before_m"]
            and metrics["voxel_occupancy_reduction"] > 0.35
            and metrics["point_to_plane_after_median_m"] < 0.0015
            and metrics["plane_signed_bias_shift_m"] < 0.001
            and metrics["box_top_systematic_shift_m"] < 0.005
            and metrics["nearest_neighbor_residual_p95_m"] < 0.04
            and metrics["completeness_at_1_5_voxels"] > 0.90
            and result.sampled.points.shape[0] == 1024
        ),
    }


def _run_real(capture: Path, provision: Path, mapping_path: Path, output: Path) -> dict:
    mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    config = parse_rig_config(
        {
            "schema_version": "pointcloud-builder.rig.v1",
            "output_frame": "workspace",
            "cameras": [
                {
                    "name": "head",
                    "enabled": True,
                    "source": {
                        "type": "camera_rig_replay",
                        "capture_artifact": str(capture),
                        "provision_artifact": str(provision),
                    },
                    "depth": {"mode": "native"},
                    "pipeline_config": None,
                    "local_crop": {"enabled": False},
                }
            ],
            "timing": {
                "mode": "exact_index",
                "reference_camera": "head",
                "maximum_skew_ms": 33.4,
            },
            "workspace_crop": mapping["workspace_crop"],
            "fusion": {
                "enabled": True,
                "voxel_size_m": 0.005,
                "origin": [-1.0, -1.0, -0.5],
                "deterministic": True,
            },
            "sampling": {
                "enabled": True,
                "mode": "voxel_random",
                "num_points": 4096,
                "voxel_size": 0.005,
                "deterministic": True,
                "seed": 31,
            },
        }
    )
    result = build_replay_rig(config, device="cuda").build(30)
    plane_raw = mapping["expected_plane"]
    plane = ExpectedPlaneRegion(
        frame="workspace",
        x=tuple(float(value) for value in plane_raw["x"]),
        y=tuple(float(value) for value in plane_raw["y"]),
        expected_z_m=float(plane_raw.get("expected_z_m", 0.0)),
        z_search_range_m=tuple(float(value) for value in plane_raw["z_search_range_m"]),
    )
    before = evaluate_expected_plane(result.workspace_cropped, plane).to_dict()
    after = evaluate_expected_plane(result.fused, plane).to_dict()
    save_ascii_ply(result.workspace_cropped.points, output / "real_single_workspace.ply")
    save_ascii_ply(result.fused.points, output / "real_single_fused.ply")
    _render_real(result, output / "real_single_fusion.png")
    provenance = result.fusion_provenance
    assert provenance is not None
    return {
        "camera_count": 1,
        "before_plane": before,
        "after_plane": after,
        "provenance": provenance.to_summary(),
        "sampled_count": int(result.sampled.points.shape[0]),
        "passed": bool(
            before["median_abs_z_m"] <= 0.020
            and before["p95_abs_z_m"] <= 0.040
            and after["median_abs_z_m"] <= 0.020
            and after["p95_abs_z_m"] <= 0.040
            and abs(after["median_abs_z_m"] - before["median_abs_z_m"]) <= 0.005
            and result.sampled.points.shape[0] == 4096
        ),
    }


def _synthetic_config(names: tuple[str, ...]) -> dict:
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
        "fusion": {
            "enabled": True,
            "voxel_size_m": 0.015,
            "origin": [-0.75, -0.60, -0.02],
            "deterministic": True,
        },
        "sampling": {
            "enabled": True,
            "mode": "voxel_fps",
            "num_points": 1024,
            "voxel_size": 0.01,
            "deterministic": True,
            "seed": 23,
        },
    }


def _render_before_after(result, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(15, 5), dpi=140)
    stages = (
        (result.workspace_cropped.points, "concatenated"),
        (result.fused.points, "fused"),
        (result.sampled.points, "sampled"),
    )
    for index, (tensor, title) in enumerate(stages, 1):
        points = tensor[:, :3].detach().cpu().numpy()[:: max(1, tensor.shape[0] // 5000)]
        axis = figure.add_subplot(1, 3, index, projection="3d")
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.25)
        axis.set_title(f"{title}: {tensor.shape[0]} points")
        axis.set(xlabel="x", ylabel="y", zlabel="z")
        axis.view_init(elev=25, azim=-60)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _render_real(result, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=140)
    for axis, tensor, title in zip(
        axes,
        (result.workspace_cropped.points, result.fused.points),
        ("real workspace", "real fused"),
        strict=True,
    ):
        points = tensor[:, :3].detach().cpu().numpy()[:: max(1, tensor.shape[0] // 30000)]
        axis.scatter(points[:, 0], points[:, 2], s=0.2)
        axis.set(xlabel="workspace x (m)", ylabel="workspace z (m)", title=title)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


if __name__ == "__main__":
    main()
