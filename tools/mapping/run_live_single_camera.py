#!/usr/bin/env python3
"""Run and validate one live CameraRig-to-workspace pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any

import psutil
import torch
import yaml

from camera_rig.api import load_camera_config, load_provisioned_camera_bundle
from pointcloud_builder.config import CropConfig, SamplingConfig, load_config
from pointcloud_builder.integrations.camera_rig import create_ffs_builder, create_native_builder
from pointcloud_builder.live import CameraRigLiveSource, LiveSingleCameraWorkspacePipeline
from pointcloud_builder.visualization import save_ascii_ply
from pointcloud_builder.workspace import (
    ExpectedPlaneRegion,
    SingleCameraWorkspacePipeline,
    evaluate_expected_plane,
    select_expected_plane_points,
)
from run_single_camera_replay import _render_open3d_acceptance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-config", required=True)
    parser.add_argument("--provision", required=True)
    parser.add_argument("--mapping-config", required=True)
    parser.add_argument("--ffs-config")
    parser.add_argument("--depth-source", choices=("native", "ffs_stereo"), required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--reopen-frames", type=int, default=60)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--memory-sample-every", type=int, default=0)
    parser.add_argument("--memory-samples")
    args = parser.parse_args()

    if args.frames <= 0 or args.reopen_frames < 0:
        raise ValueError("frame counts must be positive (reopen may be zero)")
    if args.memory_sample_every < 0:
        raise ValueError("--memory-sample-every must be non-negative")
    if bool(args.memory_samples) != bool(args.memory_sample_every):
        raise ValueError("--memory-samples and --memory-sample-every must be used together")
    if args.depth_source == "ffs_stereo" and not args.ffs_config:
        raise ValueError("--ffs-config is required for --depth-source=ffs_stereo")
    output = Path(args.output)
    report_path = Path(args.report)
    output.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    camera_config = load_camera_config(args.camera_config)
    bundle = load_provisioned_camera_bundle(args.provision)
    preflight = _hardware_preflight(camera_config, bundle)
    mapping = _load_yaml(Path(args.mapping_config))
    sampling = _sampling(mapping.get("sampling"))
    if args.depth_source == "native":
        context = create_native_builder(bundle, device="cuda", sampling=sampling)
    else:
        loaded = load_config(args.ffs_config)
        if loaded.depth_source.ffs is None:
            raise ValueError("FFS config must declare depth_source.mode=ffs_stereo")
        context = create_ffs_builder(
            bundle,
            ffs_config=loaded.depth_source.ffs,
            device=loaded.device,
            sampling=sampling,
        )
    workspace_crop = _crop(mapping.get("workspace_crop"), context.workspace_frame)
    workspace_pipeline = SingleCameraWorkspacePipeline(context, workspace_crop=workspace_crop)
    source = CameraRigLiveSource(camera_config)
    live = LiveSingleCameraWorkspacePipeline(source, workspace_pipeline)
    plane = _plane(mapping["expected_plane"], context.workspace_frame)

    process = psutil.Process()
    memory_samples: list[dict[str, int | None]] = []
    memory_start = process.memory_info().rss
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if args.memory_sample_every:
        memory_samples.append(_memory_sample(0, process))
    runs = [
        _run_once(
            live,
            args.frames,
            plane,
            output,
            "primary",
            context,
            required_fresh_streams=("depth",)
            if args.depth_source == "native"
            else ("ir_left", "ir_right"),
            memory_sample_every=args.memory_sample_every,
            memory_samples=memory_samples,
            process=process,
        ),
    ]
    if args.reopen_frames:
        runs.append(
            _run_once(
                live,
                args.reopen_frames,
                plane,
                output,
                "reopen",
                context,
                required_fresh_streams=("depth",)
                if args.depth_source == "native"
                else ("ir_left", "ir_right"),
                memory_sample_every=0,
                memory_samples=memory_samples,
                process=process,
            )
        )
    memory_end = process.memory_info().rss
    gpu_memory = {
        "allocated_end_bytes": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
        "reserved_end_bytes": int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
    }
    primary = runs[0]
    geometry_pass = bool(
        primary["plane"]["median_abs_z_m"] <= 0.020
        and primary["plane"]["p95_abs_z_m"] <= 0.040
    )
    reopen_pass = len(runs) == 1 or bool(
        runs[1]["received_frames"] == args.reopen_frames
        and runs[1]["timeouts"] == 0
    )
    report = {
        "schema_version": "pointcloud-builder.live-single-camera.v1",
        "depth_source": args.depth_source,
        "source_frame": context.source_frame,
        "workspace_frame": context.workspace_frame,
        "preflight": preflight,
        "runs": runs,
        "lifecycle": {
            "open_count": source.open_count,
            "close_count": source.close_count,
            "balanced": source.open_count == source.close_count,
            "reopen_passed": reopen_pass,
        },
        "memory": {
            "rss_start_bytes": memory_start,
            "rss_end_bytes": memory_end,
            "rss_growth_bytes": memory_end - memory_start,
            **gpu_memory,
        },
        "passed": bool(
            preflight["passed"]
            and geometry_pass
            and reopen_pass
            and source.open_count == source.close_count
            and primary["received_frames"] == args.frames
            and primary["timeouts"] == 0
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.memory_samples:
        Path(args.memory_samples).parent.mkdir(parents=True, exist_ok=True)
        Path(args.memory_samples).write_text(
            json.dumps({"samples": memory_samples}, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("live single-camera acceptance failed")


def _hardware_preflight(camera_config: Any, bundle: Any) -> dict[str, Any]:
    import pyrealsense2 as rs

    devices = list(rs.context().query_devices())
    matches = [
        device
        for device in devices
        if device.get_info(rs.camera_info.serial_number) == camera_config.camera.serial
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one connected device matching the private camera config"
        )
    device = matches[0]
    model = device.get_info(rs.camera_info.name)
    serial = device.get_info(rs.camera_info.serial_number)
    usb = device.get_info(rs.camera_info.usb_type_descriptor)
    model_match = camera_config.camera.expected_model.casefold() in model.casefold()
    identity_match = (
        serial == camera_config.camera.serial
        and serial == bundle.device.serial
        and camera_config.camera.name == bundle.device.camera_name
    )
    usb_ok = not usb.strip().startswith("2")
    if not model_match or not identity_match or not usb_ok:
        raise RuntimeError("live hardware model/identity/USB preflight failed")
    return {
        "connected_device_count": len(devices),
        "matching_device_count": 1,
        "model_match": model_match,
        "identity_match": identity_match,
        "usb_type": usb,
        "usb_ok": usb_ok,
        "provision_status": bundle.status,
        "passed": True,
    }


def _run_once(
    live: LiveSingleCameraWorkspacePipeline,
    frames: int,
    plane: ExpectedPlaneRegion,
    output: Path,
    label: str,
    context: Any,
    required_fresh_streams: tuple[str, ...],
    memory_sample_every: int,
    memory_samples: list[dict[str, int | None]],
    process: psutil.Process,
) -> dict[str, Any]:
    timings: list[dict[str, float]] = []
    plane_records: list[dict[str, Any]] = []
    previous_numbers: dict[str, int] = {}
    previous_host_timestamp: int | None = None
    per_stream_discontinuities: dict[str, int] = {}
    per_stream_stale_or_duplicate: dict[str, int] = {}
    stale_or_duplicate_bundles = 0
    selected = None
    rss_peak = psutil.Process().memory_info().rss
    start = time.perf_counter()
    with live:
        for index in range(frames):
            result = live.capture_next()
            timings.append(result.timing_ms)
            metrics = evaluate_expected_plane(result.stages.workspace_raw, plane)
            plane_records.append(metrics.to_dict())
            numbers = result.stages.metadata["frame_numbers"]
            if previous_numbers:
                for stream, number in numbers.items():
                    previous = previous_numbers.get(stream)
                    if previous is not None and number != previous + 1:
                        per_stream_discontinuities[stream] = (
                            per_stream_discontinuities.get(stream, 0) + 1
                        )
                    if previous is not None and number <= previous:
                        per_stream_stale_or_duplicate[stream] = (
                            per_stream_stale_or_duplicate.get(stream, 0) + 1
                        )
            host_timestamp = int(result.stages.metadata["host_receive_timestamp_ns"])
            if previous_host_timestamp is not None and host_timestamp <= previous_host_timestamp:
                stale_or_duplicate_bundles += 1
            previous_numbers = {str(key): int(value) for key, value in numbers.items()}
            previous_host_timestamp = host_timestamp
            rss_peak = max(rss_peak, psutil.Process().memory_info().rss)
            completed = index + 1
            if memory_sample_every and completed % memory_sample_every == 0:
                memory_samples.append(_memory_sample(completed, process))
            if index == frames // 2:
                selected = result.stages
    if selected is None:
        raise RuntimeError("live run did not retain a selected evidence frame")
    board = select_expected_plane_points(selected.workspace_raw, plane)
    save_ascii_ply(selected.workspace_cropped.points, output / f"{label}_workspace.ply")
    _render_open3d_acceptance(
        selected.workspace_cropped.points,
        board,
        output / f"{label}_workspace_3d.png",
        title=label,
        region=plane,
        T_workspace_from_camera=context.T_workspace_from_source.matrix,
        intrinsics=context.calibration.intrinsics[
            "depth" if context.depth_mode == "native" else "ir_left"
        ],
    )
    _render_camera_workspace_comparison(
        selected.camera_raw.points,
        selected.workspace_cropped.points,
        output / f"{label}_camera_workspace.png",
        label,
    )
    return {
        "requested_frames": frames,
        "received_frames": len(timings),
        "timeouts": 0,
        "frame_discontinuities": sum(per_stream_discontinuities.values()),
        "per_stream_discontinuities": per_stream_discontinuities,
        "stale_or_duplicate_frames": stale_or_duplicate_bundles
        + sum(per_stream_stale_or_duplicate.get(name, 0) for name in required_fresh_streams),
        "stale_or_duplicate_bundles": stale_or_duplicate_bundles,
        "per_stream_stale_or_duplicate_frames": per_stream_stale_or_duplicate,
        "required_fresh_streams": list(required_fresh_streams),
        "duration_s": time.perf_counter() - start,
        "timing_ms": _timing_summary(timings),
        "plane": {
            "minimum_point_count": min(int(item["point_count"]) for item in plane_records),
            "median_abs_z_m": statistics.median(
                float(item["median_abs_z_m"]) for item in plane_records
            ),
            "p95_abs_z_m": _quantile(
                [float(item["p95_abs_z_m"]) for item in plane_records], 0.95
            ),
            "maximum_normal_angle_deg": max(
                float(item["normal_angle_to_expected_deg"]) for item in plane_records
            ),
        },
        "rss_peak_bytes": rss_peak,
    }


def _timing_summary(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted(set().union(*(record.keys() for record in records)))
    return {
        key: {
            "p50": statistics.median(float(record.get(key, 0.0)) for record in records),
            "p95": _quantile([float(record.get(key, 0.0)) for record in records], 0.95),
            "mean": statistics.mean(float(record.get(key, 0.0)) for record in records),
            "max": max(float(record.get(key, 0.0)) for record in records),
        }
        for key in keys
    }


def _render_camera_workspace_comparison(
    camera_points: torch.Tensor,
    workspace_points: torch.Tensor,
    path: Path,
    label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    camera = camera_points[:, :3].detach().cpu().numpy()
    workspace = workspace_points[:, :3].detach().cpu().numpy()
    camera = camera[:: max(1, len(camera) // 30_000)]
    workspace = workspace[:: max(1, len(workspace) // 30_000)]
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=140)
    axes[0].scatter(camera[:, 0], camera[:, 2], s=0.2)
    axes[0].set(xlabel="camera x (m)", ylabel="camera z (m)", title=f"{label}: camera XZ")
    axes[1].scatter(workspace[:, 0], workspace[:, 2], s=0.2)
    axes[1].set(xlabel="workspace x (m)", ylabel="workspace z (m)", title=f"{label}: workspace XZ")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


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


def _memory_sample(frame_index: int, process: psutil.Process) -> dict[str, int | None]:
    status = Path(f"/proc/{process.pid}/status").read_text(encoding="utf-8")
    vmhwm_kib = 0
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            vmhwm_kib = int(line.split()[1])
            break
    cuda = torch.cuda.is_available()
    return {
        "frame_index": frame_index,
        "rss_bytes": int(process.memory_info().rss),
        "vmhwm_bytes": vmhwm_kib * 1024,
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated()) if cuda else 0,
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved()) if cuda else 0,
        "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated()) if cuda else 0,
        "cuda_max_reserved_bytes": int(torch.cuda.max_memory_reserved()) if cuda else 0,
        "gpu_process_memory_bytes": _gpu_process_memory_bytes(process.pid),
    }


def _gpu_process_memory_bytes(pid: int) -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    total_mib = 0
    for line in completed.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 2 and fields[0] == str(pid):
            total_mib += int(fields[1])
    return total_mib * 1024 * 1024


if __name__ == "__main__":
    main()
