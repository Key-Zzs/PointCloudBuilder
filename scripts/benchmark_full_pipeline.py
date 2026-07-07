"""Benchmark deprojection, crop, sampling, and full RGB-D pipeline latency."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from time import perf_counter
from typing import Any

import torch

from pointcloud_builder import PointCloudBuilder
from pointcloud_builder.crop import crop_point_cloud
from pointcloud_builder.deprojection import deproject_depth
from pointcloud_builder.sampling import sample_point_cloud
from pointcloud_builder.utils import normalize_color, pack_point_cloud


def synchronize_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA work when benchmarking on GPU."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], q: float) -> float:
    """Return a percentile from a non-empty list."""

    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def benchmark_stage(
    name: str,
    fn: Callable[[], Any],
    *,
    device: torch.device,
    iters: int,
    warmup: int,
) -> None:
    """Benchmark one callable stage."""

    for _ in range(warmup):
        fn()
    synchronize_if_cuda(device)

    latencies_ms: list[float] = []
    for _ in range(iters):
        start = perf_counter()
        fn()
        synchronize_if_cuda(device)
        latencies_ms.append((perf_counter() - start) * 1000.0)

    mean_ms = sum(latencies_ms) / max(len(latencies_ms), 1)
    print(f"{name}_latency_ms_mean: {mean_ms:.4f}")
    print(f"{name}_latency_ms_p50: {percentile(latencies_ms, 0.50):.4f}")
    print(f"{name}_latency_ms_p95: {percentile(latencies_ms, 0.95):.4f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/example_train_voxel_random.yaml")
    parser.add_argument("--iters", "--iterations", dest="iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    builder = PointCloudBuilder.from_yaml(args.config)
    height = builder.camera.height
    width = builder.camera.width
    depth = torch.randint(1, 2000, (height, width), dtype=torch.int32, device=builder.device)
    rgb = torch.randint(0, 255, (height, width, 3), dtype=torch.uint8, device=builder.device)
    frame = {"depth": depth, "rgb": rgb}

    intrinsics = builder.camera.active_intrinsics
    points, valid_mask = deproject_depth(depth, intrinsics, builder.camera.depth_scale, flatten=True)
    colors = None
    if builder.camera.aligned_depth_to_color and builder.config.pointcloud.use_rgb:
        colors = normalize_color(rgb).reshape(-1, 3)[valid_mask]
    raw_point_cloud = pack_point_cloud(points, colors)
    cropped_point_cloud, _ = crop_point_cloud(raw_point_cloud, builder.config.crop)

    print(f"device: {builder.device}")
    print(f"resolution: {width}x{height}")
    print(f"raw_points: {raw_point_cloud.shape[0]}")
    print(f"cropped_points: {cropped_point_cloud.shape[0]}")
    print(f"target_points: {builder.config.sampling.num_points}")
    print(f"sampling_mode: {builder.config.sampling.mode}")

    benchmark_stage(
        "deprojection",
        lambda: deproject_depth(depth, intrinsics, builder.camera.depth_scale, flatten=True),
        device=builder.device,
        iters=args.iters,
        warmup=args.warmup,
    )
    benchmark_stage(
        "crop",
        lambda: crop_point_cloud(raw_point_cloud, builder.config.crop),
        device=builder.device,
        iters=args.iters,
        warmup=args.warmup,
    )
    benchmark_stage(
        "sampling",
        lambda: sample_point_cloud(cropped_point_cloud, builder.config.sampling),
        device=builder.device,
        iters=args.iters,
        warmup=args.warmup,
    )
    benchmark_stage(
        "full_pipeline",
        lambda: builder.from_live_frame(frame),
        device=builder.device,
        iters=args.iters,
        warmup=args.warmup,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
