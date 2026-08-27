#!/usr/bin/env python3
"""Run CameraRig-derived FFS geometry and native/FFS parity on one replay."""

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
from pointcloud_builder.config import CropConfig, SamplingConfig, load_config
from pointcloud_builder.integrations.camera_rig import create_ffs_builder, create_native_builder
from pointcloud_builder.visualization import save_ascii_ply
from pointcloud_builder.workspace import (
    ExpectedPlaneRegion,
    FramedPointCloud,
    SingleCameraWorkspacePipeline,
    crop_workspace_cloud,
    evaluate_expected_plane,
    select_expected_plane_points,
    transform_point_cloud,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--provision", required=True)
    parser.add_argument("--ffs-config", required=True)
    parser.add_argument("--mapping-config", required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    report_path = Path(args.report)
    output.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    mapping = _load_yaml(Path(args.mapping_config))
    configured = load_config(args.ffs_config)
    if configured.depth_source.ffs is None:
        raise ValueError("FFS config must declare depth_source.mode=ffs_stereo")
    bundle = load_provisioned_camera_bundle(args.provision)
    sampling = _sampling(mapping.get("sampling"))
    native_context = create_native_builder(
        bundle,
        device=configured.device,
        sampling=sampling,
        use_rgb=configured.pointcloud.use_rgb,
    )
    ffs_context = create_ffs_builder(
        bundle,
        ffs_config=configured.depth_source.ffs,
        device=configured.device,
        sampling=sampling,
        use_rgb=configured.pointcloud.use_rgb,
    )
    workspace_crop = _crop(mapping.get("workspace_crop"), native_context.workspace_frame)
    native_pipeline = SingleCameraWorkspacePipeline(
        native_context,
        workspace_crop=workspace_crop,
    )
    plane = _plane(mapping["expected_plane"], native_context.workspace_frame)

    records: list[dict[str, Any]] = []
    frame_limit: int
    with ReplayCameraSession.from_artifact(args.capture) as session:
        frame_limit = min(args.frames, session.frame_count)
        if frame_limit <= 0:
            raise ValueError("--frames must select at least one replay frame")
        for index in range(frame_limit):
            frame = session.capture()
            records.append(
                _process_frame(
                    index,
                    frame,
                    native_context,
                    ffs_context,
                    native_pipeline,
                    workspace_crop,
                    plane,
                )[0]
            )

    ranked = sorted(records, key=lambda item: float(item["ffs_plane"]["rmse_m"]))
    selected_frames = {
        "best": int(ranked[0]["frame_index"]),
        "median": int(ranked[len(ranked) // 2]["frame_index"]),
        "worst": int(ranked[-1]["frame_index"]),
    }
    labels_by_index: dict[int, list[str]] = {}
    for label, index in selected_frames.items():
        labels_by_index.setdefault(index, []).append(label)
    with ReplayCameraSession.from_artifact(args.capture) as session:
        for index in range(frame_limit):
            frame = session.capture()
            labels = labels_by_index.get(index)
            if not labels:
                continue
            _, evidence = _process_frame(
                index,
                frame,
                native_context,
                ffs_context,
                native_pipeline,
                workspace_crop,
                plane,
            )
            for label in labels:
                _save_evidence(output, label, evidence, plane)

    aggregate = _aggregate(records)
    aggregate["passed"] = bool(
        aggregate["minimum_valid_overlap_ratio"] >= 0.75
        and aggregate["median_depth_abs_error_m"] <= 0.030
        and aggregate["p95_depth_abs_error_m"] <= 0.080
        and aggregate["ffs_plane_median_abs_z_m"] <= 0.020
        and aggregate["ffs_plane_p95_abs_z_m"] <= 0.040
    )
    report = {
        "schema_version": "pointcloud-builder.ffs-workspace-parity.v1",
        "backend": configured.depth_source.ffs.backend,
        "frames": frame_limit,
        "native_source_frame": native_context.source_frame,
        "ffs_source_frame": ffs_context.source_frame,
        "workspace_frame": native_context.workspace_frame,
        "bundle_rectification_gate": "passed",
        "rgb_mapping_enabled": False,
        "selected_frames": selected_frames,
        "aggregate": aggregate,
        "per_frame": records,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "backend": report["backend"],
                "frames": frame_limit,
                "native_source_frame": report["native_source_frame"],
                "ffs_source_frame": report["ffs_source_frame"],
                "workspace_frame": report["workspace_frame"],
                "aggregate": aggregate,
            },
            indent=2,
        )
    )
    if not aggregate["passed"]:
        raise SystemExit("FFS workspace/native parity acceptance failed")


def _process_frame(
    index: int,
    frame: Any,
    native_context: Any,
    ffs_context: Any,
    native_pipeline: SingleCameraWorkspacePipeline,
    workspace_crop: CropConfig,
    plane: ExpectedPlaneRegion,
) -> tuple[dict[str, Any], dict[str, Any]]:
    native_result = native_pipeline.process(frame)
    adapted = ffs_context.frame_adapter.adapt(frame)
    perception, ffs_meta = ffs_context.builder.build_perception_stages(adapted)
    ffs_camera = FramedPointCloud(
        perception["raw"],
        ffs_context.source_frame,
        metadata={"stage": "camera_raw", "depth_mode": "ffs_stereo"},
    )
    ffs_workspace = transform_point_cloud(ffs_camera, ffs_context.T_workspace_from_source)
    native_cropped = crop_workspace_cloud(native_result.workspace_raw, workspace_crop)
    ffs_cropped = crop_workspace_cloud(ffs_workspace, workspace_crop)

    native_depth = torch.as_tensor(
        adapted["depth"],
        dtype=torch.float32,
        device=perception["depth"].device,
    ) * float(native_context.calibration.depth_scale_m_per_unit)
    ffs_depth = perception["depth"]
    native_valid = torch.isfinite(native_depth) & (native_depth > 0.0)
    ffs_valid = perception["valid_mask"].to(dtype=torch.bool)
    overlap = native_valid & ffs_valid
    overlap_count = int(overlap.sum().item())
    if overlap_count == 0:
        raise ValueError(f"frame {index} has no native/FFS valid-depth overlap")
    absolute = torch.abs(native_depth[overlap] - ffs_depth[overlap])
    relative = absolute / torch.clamp(native_depth[overlap], min=1e-6)
    denominator = max(int(native_valid.sum().item()), int(ffs_valid.sum().item()), 1)
    native_plane = evaluate_expected_plane(native_result.workspace_raw, plane)
    ffs_plane = evaluate_expected_plane(ffs_workspace, plane)
    workspace_metrics = _workspace_metrics(native_cropped.points, ffs_cropped.points, plane)
    timing = dict(ffs_meta.get("ffs", {}).get("timing_ms", {}))
    record = {
        "frame_index": index,
        "native_valid_ratio": float(native_valid.float().mean().item()),
        "ffs_valid_ratio": float(ffs_valid.float().mean().item()),
        "valid_overlap_ratio": overlap_count / denominator,
        "overlap_count": overlap_count,
        "depth_mae_m": float(absolute.mean().item()),
        "depth_median_abs_error_m": float(torch.median(absolute).item()),
        "depth_p95_abs_error_m": float(torch.quantile(absolute, 0.95).item()),
        "depth_relative_error": float(torch.median(relative).item()),
        "native_plane": native_plane.to_dict(),
        "ffs_plane": ffs_plane.to_dict(),
        "workspace": workspace_metrics,
        "timing_ms": timing,
    }
    evidence = {
        "native_depth": native_depth,
        "ffs_depth": ffs_depth,
        "overlap": overlap,
        "native_workspace": native_cropped.points,
        "ffs_workspace": ffs_cropped.points,
        "native_board": select_expected_plane_points(native_result.workspace_raw, plane),
        "ffs_board": select_expected_plane_points(ffs_workspace, plane),
    }
    return record, evidence


def _workspace_metrics(
    native_points: torch.Tensor,
    ffs_points: torch.Tensor,
    plane: ExpectedPlaneRegion,
) -> dict[str, float | int]:
    from scipy.spatial import cKDTree

    native = native_points[:, :3].detach().cpu().numpy()
    ffs = ffs_points[:, :3].detach().cpu().numpy()
    native = native[:: max(1, len(native) // 20_000)]
    ffs = ffs[:: max(1, len(ffs) // 20_000)]
    if len(native) == 0 or len(ffs) == 0:
        return {
            "nearest_neighbor_median_m": float("inf"),
            "nearest_neighbor_p95_m": float("inf"),
            "nearest_neighbor_overlap_30mm": 0.0,
            "native_surface_thickness_m": float("nan"),
            "ffs_surface_thickness_m": float("nan"),
        }
    distances, _ = cKDTree(native).query(ffs, k=1, workers=-1)
    native_roi = _numpy_roi(native, plane)
    ffs_roi = _numpy_roi(ffs, plane)
    return {
        "nearest_neighbor_median_m": float(np.median(distances)),
        "nearest_neighbor_p95_m": float(np.quantile(distances, 0.95)),
        "nearest_neighbor_overlap_30mm": float(np.mean(distances <= 0.03)),
        "native_surface_thickness_m": float(np.std(native_roi[:, 2])) if len(native_roi) else float("nan"),
        "ffs_surface_thickness_m": float(np.std(ffs_roi[:, 2])) if len(ffs_roi) else float("nan"),
    }


def _numpy_roi(points: np.ndarray, plane: ExpectedPlaneRegion) -> np.ndarray:
    relative_z = points[:, 2] - plane.expected_z_m
    mask = (
        (points[:, 0] >= plane.x[0])
        & (points[:, 0] <= plane.x[1])
        & (points[:, 1] >= plane.y[0])
        & (points[:, 1] <= plane.y[1])
        & (relative_z >= plane.z_search_range_m[0])
        & (relative_z <= plane.z_search_range_m[1])
    )
    return points[mask]


def _save_evidence(
    output: Path,
    label: str,
    evidence: dict[str, Any],
    plane: ExpectedPlaneRegion,
) -> None:
    if label == "median":
        save_ascii_ply(evidence["native_workspace"], output / "native_workspace.ply")
        save_ascii_ply(evidence["ffs_workspace"], output / "ffs_workspace.ply")
        save_ascii_ply(evidence["native_board"], output / "board_native.ply")
        save_ascii_ply(evidence["ffs_board"], output / "board_ffs.ply")
        _render_depth_difference(
            evidence["native_depth"],
            evidence["ffs_depth"],
            evidence["overlap"],
            output / "depth_difference.png",
        )
    _render_parity_clouds(
        evidence["native_workspace"],
        evidence["ffs_workspace"],
        output / f"{label}_parity.png",
        label,
        plane,
    )


def _render_depth_difference(
    native_depth: torch.Tensor,
    ffs_depth: torch.Tensor,
    overlap: torch.Tensor,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    difference = (ffs_depth - native_depth).detach().cpu().numpy()
    valid = overlap.detach().cpu().numpy()
    difference[~valid] = np.nan
    figure, axis = plt.subplots(figsize=(8, 6), dpi=140)
    image = axis.imshow(difference, cmap="coolwarm", vmin=-0.08, vmax=0.08)
    axis.set_title("FFS depth - native depth (m)")
    axis.set_axis_off()
    figure.colorbar(image, ax=axis, label="meters")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _render_parity_clouds(
    native_points: torch.Tensor,
    ffs_points: torch.Tensor,
    path: Path,
    label: str,
    plane: ExpectedPlaneRegion,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    native = native_points[:, :3].detach().cpu().numpy()
    ffs = ffs_points[:, :3].detach().cpu().numpy()
    native = native[:: max(1, len(native) // 12_000)]
    ffs = ffs[:: max(1, len(ffs) // 12_000)]
    figure = plt.figure(figsize=(9, 7), dpi=140)
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(native[:, 0], native[:, 1], native[:, 2], s=0.15, c="#2864b7", label="native")
    axis.scatter(ffs[:, 0], ffs[:, 1], ffs[:, 2], s=0.15, c="#ef7d22", label="FFS")
    x0, x1 = plane.x
    y0, y1 = plane.y
    z = plane.expected_z_m
    axis.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [z] * 5, c="red", linewidth=2)
    axis.quiver(0, 0, 0, 0.12, 0, 0, color="red")
    axis.quiver(0, 0, 0, 0, 0.12, 0, color="green")
    axis.quiver(0, 0, 0, 0, 0, 0.12, color="blue")
    axis.set(xlabel="workspace x (m)", ylabel="workspace y (m)", zlabel="workspace z (m)")
    axis.set_title(f"{label}: native / FFS workspace parity")
    axis.legend(markerscale=12)
    axis.view_init(elev=28, azim=-65)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "minimum_valid_overlap_ratio": min(float(item["valid_overlap_ratio"]) for item in records),
        "native_valid_ratio": statistics.median(float(item["native_valid_ratio"]) for item in records),
        "ffs_valid_ratio": statistics.median(float(item["ffs_valid_ratio"]) for item in records),
        "depth_mae_m": statistics.median(float(item["depth_mae_m"]) for item in records),
        "median_depth_abs_error_m": statistics.median(
            float(item["depth_median_abs_error_m"]) for item in records
        ),
        "p95_depth_abs_error_m": _quantile(
            [float(item["depth_p95_abs_error_m"]) for item in records], 0.95
        ),
        "depth_relative_error": statistics.median(
            float(item["depth_relative_error"]) for item in records
        ),
        "native_plane_median_abs_z_m": statistics.median(
            float(item["native_plane"]["median_abs_z_m"]) for item in records
        ),
        "native_plane_p95_abs_z_m": _quantile(
            [float(item["native_plane"]["p95_abs_z_m"]) for item in records], 0.95
        ),
        "ffs_plane_median_abs_z_m": statistics.median(
            float(item["ffs_plane"]["median_abs_z_m"]) for item in records
        ),
        "ffs_plane_p95_abs_z_m": _quantile(
            [float(item["ffs_plane"]["p95_abs_z_m"]) for item in records], 0.95
        ),
        "workspace_nn_overlap_30mm": statistics.median(
            float(item["workspace"]["nearest_neighbor_overlap_30mm"]) for item in records
        ),
        "native_surface_thickness_m": statistics.median(
            float(item["workspace"]["native_surface_thickness_m"]) for item in records
        ),
        "ffs_surface_thickness_m": statistics.median(
            float(item["workspace"]["ffs_surface_thickness_m"]) for item in records
        ),
        "inference_p50_ms": statistics.median(
            float(item["timing_ms"].get("inference", 0.0)) for item in records
        ),
        "inference_p95_ms": _quantile(
            [float(item["timing_ms"].get("inference", 0.0)) for item in records], 0.95
        ),
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("mapping config must be a YAML mapping")
    return value


def _range(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element list")
    result = (float(value[0]), float(value[1]))
    if result[0] > result[1]:
        raise ValueError(f"{name} must be ordered")
    return result


def _crop(value: Any, frame: str) -> CropConfig:
    raw = value if isinstance(value, dict) else {}
    return CropConfig(
        enabled=bool(raw.get("enabled", False)),
        x=_range(raw.get("x", [-float("inf"), float("inf")]), "workspace_crop.x"),
        y=_range(raw.get("y", [-float("inf"), float("inf")]), "workspace_crop.y"),
        z=_range(raw.get("z", [-float("inf"), float("inf")]), "workspace_crop.z"),
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
    if str(value.get("frame", frame)) != frame:
        raise ValueError("expected_plane.frame must match workspace frame")
    return ExpectedPlaneRegion(
        frame=frame,
        x=_range(value["x"], "expected_plane.x"),
        y=_range(value["y"], "expected_plane.y"),
        expected_z_m=float(value.get("expected_z_m", 0.0)),
        z_search_range_m=_range(value["z_search_range_m"], "expected_plane.z_search_range_m"),
    )


def _quantile(values: list[float], q: float) -> float:
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), q).item())


if __name__ == "__main__":
    main()
