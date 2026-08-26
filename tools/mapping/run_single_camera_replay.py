#!/usr/bin/env python3
"""Run one CameraRig replay through a native workspace point-cloud pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np
import torch
import yaml

from camera_rig.api import ReplayCameraSession, load_provisioned_camera_bundle
from pointcloud_builder.config import CropConfig, SamplingConfig
from pointcloud_builder.integrations.camera_rig import create_native_builder
from pointcloud_builder.visualization import save_ascii_ply
from pointcloud_builder.workspace import (
    ExpectedPlaneRegion,
    SingleCameraWorkspacePipeline,
    evaluate_expected_plane,
    select_expected_plane_points,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--provision", required=True)
    parser.add_argument("--depth-source", choices=("native",), default="native")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    config = _load_yaml(Path(args.config))
    output = Path(args.output)
    report_path = Path(args.report)
    output.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = load_provisioned_camera_bundle(args.provision)
    context = create_native_builder(
        bundle,
        device=str(config.get("device", "auto")),
        crop=_crop(config.get("camera_crop"), frame="camera"),
        sampling=_sampling(config.get("sampling")),
    )
    pipeline = SingleCameraWorkspacePipeline(
        context,
        workspace_crop=_crop(config.get("workspace_crop"), frame=context.workspace_frame),
    )
    plane = _plane(config["expected_plane"], context.workspace_frame)

    records: list[dict[str, Any]] = []
    with ReplayCameraSession.from_artifact(args.capture) as session:
        for index in range(session.frame_count):
            result = pipeline.process(session.capture())
            metrics = evaluate_expected_plane(result.workspace_raw, plane)
            records.append({"frame_index": index, **metrics.to_dict()})

    ranked = sorted(records, key=lambda item: float(item["rmse_m"]))
    selected_frames = {
        "best": int(ranked[0]["frame_index"]),
        "median": int(ranked[len(ranked) // 2]["frame_index"]),
        "worst": int(ranked[-1]["frame_index"]),
    }
    labels_by_index = {index: label for label, index in selected_frames.items()}
    with ReplayCameraSession.from_artifact(args.capture) as session:
        for index in range(session.frame_count):
            frame = session.capture()
            label = labels_by_index.get(index)
            if label is None:
                continue
            result = pipeline.process(frame)
            board = select_expected_plane_points(result.workspace_raw, plane)
            save_ascii_ply(result.workspace_cropped.points, output / f"{label}_workspace.ply")
            _render_xy_xz(
                result.workspace_cropped.points,
                output / f"{label}_workspace.png",
                label,
            )
            _render_open3d_acceptance(
                result.workspace_cropped.points,
                board,
                output / f"{label}_workspace_3d.png",
                title=label,
                region=plane,
                T_workspace_from_camera=context.T_workspace_from_source.matrix,
                intrinsics=context.calibration.intrinsics["depth"],
            )
            if label == "median":
                save_ascii_ply(result.camera_raw.points, output / "camera_frame.ply")
                save_ascii_ply(result.workspace_raw.points, output / "workspace_frame.ply")
                save_ascii_ply(board, output / "board_roi.ply")

    aggregate = _aggregate(records)
    aggregate["passed"] = bool(
        aggregate["minimum_point_count"] >= 500
        and aggregate["median_abs_z_m"] <= 0.020
        and aggregate["p95_abs_z_m"] <= 0.040
        and aggregate["maximum_normal_angle_deg"] <= 5.0
    )
    report = {
        "schema_version": "pointcloud-builder.native-workspace-report.v1",
        "depth_source": args.depth_source,
        "frames": len(records),
        "source_frame": context.source_frame,
        "workspace_frame": context.workspace_frame,
        "rgb_mapping_enabled": False,
        "expected_plane": {
            "frame": plane.frame,
            "x": plane.x,
            "y": plane.y,
            "expected_z_m": plane.expected_z_m,
            "z_search_range_m": plane.z_search_range_m,
        },
        "aggregate": aggregate,
        "selected_frames": selected_frames,
        "per_frame": records,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("frames", "source_frame", "workspace_frame", "aggregate")}, indent=2))
    if not aggregate["passed"]:
        raise SystemExit("native workspace plane acceptance failed")


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("mapping replay config must be a YAML mapping")
    return value


def _range(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element list")
    result = (float(value[0]), float(value[1]))
    if result[0] > result[1]:
        raise ValueError(f"{name} must be ordered")
    return result


def _crop(value: Any, *, frame: str) -> CropConfig:
    raw = value if isinstance(value, dict) else {}
    return CropConfig(
        enabled=bool(raw.get("enabled", False)),
        x=_range(raw.get("x", [-float("inf"), float("inf")]), "crop.x"),
        y=_range(raw.get("y", [-float("inf"), float("inf")]), "crop.y"),
        z=_range(raw.get("z", [-float("inf"), float("inf")]), "crop.z"),
        frame=frame,
    )


def _sampling(value: Any) -> SamplingConfig:
    raw = value if isinstance(value, dict) else {}
    seed = raw.get("seed")
    return SamplingConfig(
        mode=str(raw.get("mode", "voxel_random")),  # type: ignore[arg-type]
        num_points=int(raw.get("num_points", 4096)),
        enabled=bool(raw.get("enabled", True)),
        stride=int(raw.get("stride", 1)),
        voxel_size=float(raw.get("voxel_size", 0.005)),
        seed=None if seed is None else int(seed),
        deterministic=bool(raw.get("deterministic", False)),
        pad_mode=str(raw.get("pad_mode", "repeat")),  # type: ignore[arg-type]
    )


def _plane(value: Any, frame: str) -> ExpectedPlaneRegion:
    if not isinstance(value, dict):
        raise ValueError("expected_plane must be a mapping")
    configured_frame = str(value.get("frame", frame))
    if configured_frame != frame:
        raise ValueError("expected_plane.frame must match the bundle parent frame")
    return ExpectedPlaneRegion(
        frame=frame,
        x=_range(value["x"], "expected_plane.x"),
        y=_range(value["y"], "expected_plane.y"),
        expected_z_m=float(value.get("expected_z_m", 0.0)),
        z_search_range_m=_range(value["z_search_range_m"], "expected_plane.z_search_range_m"),
    )


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "minimum_point_count": min(int(item["point_count"]) for item in records),
        "median_point_count": int(statistics.median(int(item["point_count"]) for item in records)),
        "signed_z_bias_m": statistics.median(float(item["signed_z_bias_m"]) for item in records),
        "median_abs_z_m": statistics.median(float(item["median_abs_z_m"]) for item in records),
        "p95_abs_z_m": _quantile([float(item["p95_abs_z_m"]) for item in records], 0.95),
        "plane_rmse_m": statistics.median(float(item["rmse_m"]) for item in records),
        "maximum_normal_angle_deg": max(
            float(item["normal_angle_to_expected_deg"]) for item in records
        ),
    }


def _quantile(values: list[float], q: float) -> float:
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, q).item())


def _render_xy_xz(points: torch.Tensor, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = points.detach().cpu().numpy()
    stride = max(1, len(xyz) // 30_000)
    xyz = xyz[::stride]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140)
    axes[0].scatter(xyz[:, 0], xyz[:, 1], s=0.2)
    axes[0].set(xlabel="workspace x (m)", ylabel="workspace y (m)", title=f"{title}: XY")
    axes[0].set_aspect("equal", adjustable="box")
    axes[1].scatter(xyz[:, 0], xyz[:, 2], s=0.2)
    axes[1].set(xlabel="workspace x (m)", ylabel="workspace z (m)", title=f"{title}: XZ")
    axes[1].set_aspect("equal", adjustable="box")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _render_open3d_acceptance(
    points: torch.Tensor,
    board_points: torch.Tensor,
    path: Path,
    *,
    title: str,
    region: ExpectedPlaneRegion,
    T_workspace_from_camera: np.ndarray,
    intrinsics: Any,
) -> None:
    """Render a 3D workspace view with axes, ROI, and calibrated camera frustum."""

    import open3d as o3d

    renderer = o3d.visualization.rendering.OffscreenRenderer(1280, 900)
    renderer.scene.set_background(np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32))

    cloud_np = points[:, :3].detach().cpu().numpy()
    stride = max(1, len(cloud_np) // 120_000)
    cloud_geometry = o3d.geometry.PointCloud()
    cloud_geometry.points = o3d.utility.Vector3dVector(cloud_np[::stride])
    cloud_geometry.paint_uniform_color([0.18, 0.42, 0.72])
    point_material = o3d.visualization.rendering.MaterialRecord()
    point_material.shader = "defaultUnlit"
    point_material.point_size = 2.0
    renderer.scene.add_geometry("workspace_cloud", cloud_geometry, point_material)

    board_np = board_points[:, :3].detach().cpu().numpy()
    board_stride = max(1, len(board_np) // 30_000)
    board_geometry = o3d.geometry.PointCloud()
    board_geometry.points = o3d.utility.Vector3dVector(board_np[::board_stride])
    board_geometry.paint_uniform_color([0.90, 0.12, 0.12])
    board_material = o3d.visualization.rendering.MaterialRecord()
    board_material.shader = "defaultUnlit"
    board_material.point_size = 4.0
    renderer.scene.add_geometry("expected_plane_points", board_geometry, board_material)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12, origin=[0.0, 0.0, 0.0])
    axes_material = o3d.visualization.rendering.MaterialRecord()
    axes_material.shader = "defaultLit"
    renderer.scene.add_geometry("workspace_axes", axes, axes_material)

    roi_lines = _roi_line_set(region)
    line_material = o3d.visualization.rendering.MaterialRecord()
    line_material.shader = "unlitLine"
    line_material.line_width = 5.0
    renderer.scene.add_geometry("asymmetric_roi", roi_lines, line_material)

    frustum = _camera_frustum_line_set(T_workspace_from_camera, intrinsics)
    renderer.scene.add_geometry("camera_frustum", frustum, line_material)
    camera_origin = np.asarray(T_workspace_from_camera[:3, 3], dtype=np.float64)
    camera_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
    camera_marker.translate(camera_origin)
    camera_marker.paint_uniform_color([0.10, 0.75, 0.20])
    renderer.scene.add_geometry("camera_origin", camera_marker, axes_material)

    center = np.asarray([0.20, 0.10, 0.02], dtype=np.float32)
    eye = np.asarray([0.90, -1.00, 0.75], dtype=np.float32)
    renderer.setup_camera(55.0, center, eye, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
    image = renderer.render_to_image()
    if not o3d.io.write_image(str(path), image, 9):
        raise RuntimeError(f"Open3D failed to write {title} acceptance image")
    renderer.scene.clear_geometry()


def _roi_line_set(region: ExpectedPlaneRegion):
    import open3d as o3d

    x0, x1 = region.x
    y0, y1 = region.y
    z = region.expected_z_m + 0.002
    dx = (x1 - x0) * 0.22
    dy = (y1 - y0) * 0.22
    points = np.asarray(
        [
            [x0, y0, z],
            [x1, y0, z],
            [x1, y1, z],
            [x0, y1, z],
            [x0 + dx, y0, z],
            [x0, y0 + dy, z],
        ],
        dtype=np.float64,
    )
    lines = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [0, 5]], dtype=np.int32)
    colors = np.tile(np.asarray([[1.0, 0.55, 0.0]], dtype=np.float64), (len(lines), 1))
    result = o3d.geometry.LineSet()
    result.points = o3d.utility.Vector3dVector(points)
    result.lines = o3d.utility.Vector2iVector(lines)
    result.colors = o3d.utility.Vector3dVector(colors)
    return result


def _camera_frustum_line_set(T_workspace_from_camera: np.ndarray, intrinsics: Any):
    import open3d as o3d

    z = 0.18
    pixels = (
        (0.0, 0.0),
        (float(intrinsics.width - 1), 0.0),
        (float(intrinsics.width - 1), float(intrinsics.height - 1)),
        (0.0, float(intrinsics.height - 1)),
    )
    camera_points = [np.zeros(3, dtype=np.float64)]
    for u, v in pixels:
        camera_points.append(
            np.asarray(
                [
                    (u - intrinsics.cx) * z / intrinsics.fx,
                    (v - intrinsics.cy) * z / intrinsics.fy,
                    z,
                ],
                dtype=np.float64,
            )
        )
    camera_points.append(np.asarray([0.0, 0.0, z * 1.65], dtype=np.float64))
    camera_array = np.asarray(camera_points)
    workspace = (
        camera_array @ np.asarray(T_workspace_from_camera[:3, :3], dtype=np.float64).T
        + np.asarray(T_workspace_from_camera[:3, 3], dtype=np.float64)
    )
    lines = np.asarray(
        [
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [2, 3], [3, 4], [4, 1],
            [0, 5],
        ],
        dtype=np.int32,
    )
    colors = np.tile(np.asarray([[0.05, 0.70, 0.18]], dtype=np.float64), (len(lines), 1))
    colors[-1] = [0.80, 0.0, 0.80]
    result = o3d.geometry.LineSet()
    result.points = o3d.utility.Vector3dVector(workspace)
    result.lines = o3d.utility.Vector2iVector(lines)
    result.colors = o3d.utility.Vector3dVector(colors)
    return result


if __name__ == "__main__":
    main()
