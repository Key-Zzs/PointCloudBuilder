#!/usr/bin/env python3
"""Validate one live raw, dense, or compact RGB reconstruction profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml

from pointcloud_builder.fusion import (
    board_surface_metrics,
    contribution_metrics,
    fusion_geometry_metrics,
)
from pointcloud_builder.rig import build_live_rig, load_rig_config
from pointcloud_builder.visualization import save_ascii_ply
from pointcloud_builder.workspace import ExpectedPlaneRegion

ProfileName = Literal["raw", "dense", "compact"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("raw", "dense", "compact"), required=True)
    parser.add_argument("--rig-config", required=True)
    parser.add_argument("--mapping-config", required=True)
    parser.add_argument("--matched-sets", type=int, default=60)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--viewer", choices=("none", "rerun"), default="none")
    parser.add_argument("--rerun-connect")
    parser.add_argument("--rerun-spawn", action="store_true")
    parser.add_argument("--rerun-record")
    parser.add_argument("--viewer-point-budget", type=int, default=100_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.matched_sets <= 0:
        raise ValueError("--matched-sets must be positive")
    if args.viewer_point_budget <= 0:
        raise ValueError("--viewer-point-budget must be positive")
    if args.viewer == "none" and any(
        (args.rerun_connect, args.rerun_spawn, args.rerun_record)
    ):
        raise ValueError("Rerun output flags require --viewer rerun")

    profile: ProfileName = args.profile
    config = load_rig_config(args.rig_config)
    _validate_profile_config(config, profile)
    mapping = yaml.safe_load(Path(args.mapping_config).read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or not isinstance(
        mapping.get("expected_plane"), dict
    ):
        raise TypeError("mapping config must contain expected_plane")
    board = _plane(mapping["expected_plane"], config.output_frame)

    output = _private_output(args.output)
    report_path = _private_output(args.report)
    rerun_record = (
        str(_private_output(args.rerun_record)) if args.rerun_record else None
    )
    if output.exists() or report_path.exists():
        raise FileExistsError("profile output/report must not already exist")
    if rerun_record is not None and Path(rerun_record).exists():
        raise FileExistsError("Rerun recording must not already exist")
    output.mkdir(parents=True, exist_ok=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = build_live_rig(config, device="cuda")
    viewer = None
    viewer_error: str | None = None
    viewer_telemetry: dict[str, Any] | None = None
    if args.viewer == "rerun":
        from pointcloud_builder.visualization.rerun import (
            RerunOutputConfig,
            RerunViewerProcess,
        )

        implicit_spawn = not any((args.rerun_spawn, args.rerun_connect, rerun_record))
        try:
            viewer = RerunViewerProcess(
                RerunOutputConfig(
                    spawn=bool(args.rerun_spawn or implicit_spawn),
                    connect_url=args.rerun_connect,
                    record_path=rerun_record,
                )
            )
            viewer.start()
        except Exception as error:  # noqa: BLE001 - retain reconstruction evidence
            viewer_error = f"{type(error).__name__}: {str(error)[:500]}"
            viewer = None

    matched_count = 0
    selected_result: Any | None = None
    selected_offset = args.matched_sets // 2
    profile_counts: list[int] = []
    latencies_ms: list[float] = []
    stage_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        pipeline.acquisition.start()
        for index in range(args.matched_sets):
            built = pipeline.capture_next()
            matched_count += 1
            profile_counts.append(
                int(_profile_cloud(built.result, profile).points.shape[0])
            )
            if index == selected_offset:
                selected_result = built.result
            latencies_ms.append(float(built.total_ms))
            stage_records.extend(
                dict(item) for item in built.result.per_camera_stage_statistics.values()
            )
            if viewer is not None:
                from pointcloud_builder.visualization.rerun.conversion import (
                    packet_from_rig_result,
                )

                result = built.result
                packet = packet_from_rig_result(
                    result,
                    pipeline.processor.runtimes,
                    point_budget=args.viewer_point_budget,
                    metrics={
                        "skew_ms": float(result.frame_match.maximum_skew_ms),
                        "input_points": float(result.concatenated.points.shape[0]),
                        "fused_voxels": float(result.fused.points.shape[0]),
                        "sampled_points": float(result.sampled.points.shape[0]),
                    },
                )
                if not viewer.submit(packet):
                    status = viewer.telemetry()
                    viewer_error = status.child_error or "Rerun viewer process stopped"
                    viewer_telemetry = vars(viewer.close(timeout_s=5.0))
                    viewer = None
    finally:
        try:
            pipeline.acquisition.stop()
        finally:
            if viewer is not None:
                viewer_telemetry = vars(viewer.close())
    duration_s = time.perf_counter() - started
    if selected_result is None:
        raise RuntimeError("profile validation captured no matched sets")

    selected = _profile_cloud(selected_result, profile)
    selected_metrics = _point_metrics(selected.points)
    board_metrics = board_surface_metrics(selected, board)
    fusion_geometry = None
    contribution = None
    if config.fusion.enabled:
        fusion_geometry = fusion_geometry_metrics(
            selected_result.workspace_cropped,
            selected_result.fused,
            board_region=board,
            voxel_size_m=config.fusion.voxel_size_m,
        )
        contribution = contribution_metrics(selected_result.fusion_provenance)

    selected_path = output / f"{profile}_selected_workspace.ply"
    save_ascii_ply(selected.points, selected_path)
    _render_views(
        selected.points.detach().cpu(), output / f"{profile}_selected_views.png"
    )

    acquisition = pipeline.acquisition.report()
    matcher = acquisition["matcher"]
    stage_backends = sorted(
        {
            str(item["ffs_backend"])
            for item in stage_records
            if item.get("ffs_backend") is not None
        }
    )
    matcher_passed = _matcher_passed(
        matcher,
        requested=args.matched_sets,
        maximum_skew_ms=config.timing.maximum_skew_ms,
    )
    viewer_passed = bool(
        args.viewer == "none"
        or (
            viewer_error is None
            and viewer_telemetry is not None
            and viewer_telemetry["child_error"] is None
            and int(viewer_telemetry["child_logged_packets"]) >= 1
        )
    )
    gates = {
        "matched_sets": matched_count == args.matched_sets,
        "matcher_integrity": matcher_passed,
        "xyzrgb": bool(
            selected_metrics["channels"] == 6
            and selected_metrics["finite"]
            and selected_metrics["rgb_in_unit_interval"]
        ),
        "nonempty": selected_metrics["point_count"] > 0,
        "tensorrt_plugin": stage_backends == ["tensorrt_plugin"],
        "worker_cleanup": bool(
            not acquisition["workers_alive"] and not acquisition["worker_errors"]
        ),
        "viewer": viewer_passed,
    }
    if profile == "dense":
        assert fusion_geometry is not None and contribution is not None
        gates.update(
            {
                "actual_fused_count": (
                    selected_metrics["point_count"]
                    == int(selected_result.fused.points.shape[0])
                ),
                "surface_quality": bool(
                    board_metrics["median_abs_z_m"] <= 0.020
                    and board_metrics["p95_abs_z_m"] <= 0.040
                    and fusion_geometry["thickness_gate_passed"]
                    and fusion_geometry["board_shift_gate_passed"]
                ),
                "fusion_contribution": bool(contribution["passed"]),
            }
        )
    elif profile == "compact":
        gates.update(
            {
                "target_point_count": selected_metrics["point_count"] == 30_000,
                "fps_rgb_preserved": _rows_are_subset(
                    selected.points, selected_result.fused.points
                ),
            }
        )

    report = {
        "schema_version": "pointcloud-builder.live-reconstruction-profile.v1",
        "snapshot_only": True,
        "persistent_mapping": False,
        "profile": profile,
        "rig_config_sha256": _sha256_file(args.rig_config),
        "mapping_config_sha256": _sha256_file(args.mapping_config),
        "matched_sets": matched_count,
        "duration_s": duration_s,
        "throughput_fps": matched_count / max(duration_s, 1e-9),
        "latency_ms": _summary(latencies_ms),
        "selected_frame_offset": selected_offset,
        "selected_cloud": {
            **selected_metrics,
            "path": str(selected_path),
            "count_over_run": _summary([float(value) for value in profile_counts]),
        },
        "surface_quality": board_metrics,
        "fusion_geometry": fusion_geometry,
        "fusion_contribution": contribution,
        "sampling": selected_result.sampled.metadata.get("global_sampling"),
        "stage_backends": stage_backends,
        "viewer": {
            "requested": args.viewer == "rerun",
            "recording_enabled": rerun_record is not None,
            "telemetry": viewer_telemetry,
            "error": viewer_error,
            "passed": viewer_passed,
        },
        "acquisition": acquisition,
        "gates": gates,
        "passed": all(gates.values()),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "profile": profile,
                "matched_sets": matched_count,
                "point_count": selected_metrics["point_count"],
                "channels": selected_metrics["channels"],
                "gates": gates,
                "passed": report["passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit(f"live {profile} reconstruction profile acceptance failed")


def _validate_profile_config(config: Any, profile: ProfileName) -> None:
    cameras = config.enabled_cameras
    if len(cameras) != 2:
        raise ValueError(
            "live reconstruction profiles require exactly two enabled cameras"
        )
    if any(camera.source.type != "camera_rig_live" for camera in cameras):
        raise ValueError("live reconstruction profiles require camera_rig_live sources")
    if any(camera.depth.mode != "ffs_stereo" for camera in cameras):
        raise ValueError("live reconstruction profiles require ffs_stereo depth")
    if any(not camera.pointcloud.use_rgb for camera in cameras):
        raise ValueError("live reconstruction profiles require RGB from every camera")
    if config.timing.mode != "nearest_host_timestamp":
        raise ValueError("live reconstruction profiles require host timestamp matching")
    if profile == "raw":
        if config.fusion.enabled or config.sampling.enabled:
            raise ValueError("raw profile requires fusion OFF and sampling OFF")
        return
    if not config.fusion.enabled or config.fusion.voxel_size_m != 0.0025:
        raise ValueError("dense/compact profiles require deterministic 2.5 mm fusion")
    if not config.fusion.deterministic:
        raise ValueError("dense/compact profiles require deterministic fusion")
    if profile == "dense":
        if config.sampling.enabled:
            raise ValueError("dense profile requires sampling OFF")
        return
    if (
        not config.sampling.enabled
        or config.sampling.mode != "fps"
        or config.sampling.num_points != 30_000
    ):
        raise ValueError("compact profile requires one global 30,000-point FPS")


def _profile_cloud(result: Any, profile: ProfileName) -> Any:
    if profile == "raw":
        return result.concatenated
    if profile == "dense":
        return result.fused
    return result.sampled


def _point_metrics(points: torch.Tensor) -> dict[str, Any]:
    detached = points.detach()
    finite = bool(torch.isfinite(detached).all().item())
    channels = int(detached.shape[1])
    rgb_in_unit_interval = False
    rgb_nonblack_ratio = None
    if channels == 6 and detached.shape[0] > 0:
        rgb = detached[:, 3:6]
        rgb_in_unit_interval = bool(((rgb >= 0.0) & (rgb <= 1.0)).all().item())
        rgb_nonblack_ratio = float((rgb.amax(dim=1) > 0.0).float().mean().item())
    return {
        "point_count": int(detached.shape[0]),
        "channels": channels,
        "finite": finite,
        "rgb_in_unit_interval": rgb_in_unit_interval,
        "rgb_nonblack_ratio": rgb_nonblack_ratio,
    }


def _rows_are_subset(selected: torch.Tensor, source: torch.Tensor) -> bool:
    if selected.ndim != 2 or source.ndim != 2 or selected.shape[1] != source.shape[1]:
        return False
    selected_np = np.ascontiguousarray(selected.detach().cpu().numpy())
    source_np = np.ascontiguousarray(source.detach().cpu().numpy())
    row_dtype = np.dtype((np.void, selected_np.dtype.itemsize * selected_np.shape[1]))
    selected_rows = selected_np.view(row_dtype).reshape(-1)
    source_rows = source_np.view(row_dtype).reshape(-1)
    return bool(np.isin(selected_rows, source_rows).all())


def _matcher_passed(
    matcher: dict[str, Any], *, requested: int, maximum_skew_ms: float
) -> bool:
    return bool(
        int(matcher["matched_sets"]) == requested
        and int(matcher["frame_reuse_violations"]) == 0
        and int(matcher.get("wait_timeouts", 0)) == 0
        and float(matcher["maximum_absolute_skew_ms"]) <= 2.0 * maximum_skew_ms
        and all(
            float(item["p95"] or 0.0) <= maximum_skew_ms
            for item in matcher["absolute_skew_ms"].values()
        )
    )


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "p50": statistics.median(values),
        "p95": float(np.quantile(values, 0.95)),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def _render_views(points: torch.Tensor, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cloud = points.numpy()
    cloud = cloud[:: max(1, cloud.shape[0] // 40_000)]
    colors = np.clip(cloud[:, 3:6], 0.0, 1.0)
    figure = plt.figure(figsize=(16, 5), dpi=140)
    for index, (elev, azim, title) in enumerate(
        ((90, -90, "top"), (22, -60, "oblique"), (5, 0, "side")), 1
    ):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        axis.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=0.2, c=colors)
        axis.view_init(elev=elev, azim=azim)
        axis.set_title(title)
        axis.set(xlabel="x", ylabel="y", zlabel="z")
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


def _private_output(value: str) -> Path:
    output = Path(value).resolve()
    if not output.is_relative_to((Path.cwd() / ".local").resolve()):
        raise ValueError("live reconstruction outputs must be written under .local/")
    return output


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
