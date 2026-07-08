"""Benchmark raw RGB-D deprojection."""

from __future__ import annotations

import argparse
from time import perf_counter

import torch

from pointcloud_builder import PointCloudBuilder
from pointcloud_builder.deprojection import deproject_depth


def synchronize_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA work when the benchmark runs on GPU."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], q: float) -> float:
    """Return a percentile from a non-empty list."""

    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/example_head_aligned.yaml")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()

    builder = PointCloudBuilder.from_yaml(args.config)
    height = builder.camera.height
    width = builder.camera.width
    depth = torch.randint(1, 2000, (height, width), dtype=torch.int32, device=builder.device)
    intrinsics = builder.camera.active_intrinsics

    for _ in range(args.warmup):
        deproject_depth(depth, intrinsics, builder.camera.depth_scale, flatten=True)
    synchronize_if_cuda(builder.device)

    latencies_ms: list[float] = []
    point_count = 0
    for _ in range(args.iters):
        start = perf_counter()
        points, _ = deproject_depth(depth, intrinsics, builder.camera.depth_scale, flatten=True)
        synchronize_if_cuda(builder.device)
        latencies_ms.append((perf_counter() - start) * 1000.0)
        point_count = int(points.shape[0])

    mean_ms = sum(latencies_ms) / max(len(latencies_ms), 1)
    print(f"device: {builder.device}")
    print(f"resolution: {width}x{height}")
    print(f"points: {point_count}")
    print(f"iters: {args.iters}")
    print(f"warmup: {args.warmup}")
    print(f"latency_ms_mean: {mean_ms:.4f}")
    print(f"latency_ms_p50: {percentile(latencies_ms, 0.50):.4f}")
    print(f"latency_ms_p95: {percentile(latencies_ms, 0.95):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
