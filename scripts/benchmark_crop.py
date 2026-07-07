"""Benchmark workspace crop on XYZ and XYZRGB point clouds."""

from __future__ import annotations

import argparse
from time import perf_counter

import torch

from pointcloud_builder.config import load_config
from pointcloud_builder.crop import crop_point_cloud
from pointcloud_builder.utils import resolve_device


def synchronize_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA work when the benchmark runs on GPU."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], q: float) -> float:
    """Return a percentile from a non-empty list."""

    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def make_point_cloud(num_points: int, with_rgb: bool, device: torch.device) -> torch.Tensor:
    """Generate a random camera-frame point cloud for crop benchmarking."""

    xyz = torch.empty((num_points, 3), dtype=torch.float32, device=device)
    xyz[:, 0].uniform_(-1.0, 1.0)
    xyz[:, 1].uniform_(-1.0, 1.0)
    xyz[:, 2].uniform_(0.0, 2.0)
    if not with_rgb:
        return xyz
    rgb = torch.rand((num_points, 3), dtype=torch.float32, device=device)
    return torch.cat([xyz, rgb], dim=-1)


def run_case(
    point_cloud: torch.Tensor,
    *,
    iters: int,
    warmup: int,
    config_path: str,
) -> None:
    """Benchmark one point-cloud layout."""

    config = load_config(config_path)
    device = point_cloud.device
    for _ in range(warmup):
        crop_point_cloud(point_cloud, config.crop)
    synchronize_if_cuda(device)

    latencies_ms: list[float] = []
    cropped_count = 0
    for _ in range(iters):
        start = perf_counter()
        cropped, _ = crop_point_cloud(point_cloud, config.crop)
        synchronize_if_cuda(device)
        latencies_ms.append((perf_counter() - start) * 1000.0)
        cropped_count = int(cropped.shape[0])

    mean_ms = sum(latencies_ms) / max(len(latencies_ms), 1)
    print(f"device: {device}")
    print(f"input_points: {point_cloud.shape[0]}")
    print(f"cropped_points: {cropped_count}")
    print(f"with_rgb: {point_cloud.shape[1] == 6}")
    print(f"iters: {iters}")
    print(f"warmup: {warmup}")
    print(f"latency_ms_mean: {mean_ms:.4f}")
    print(f"latency_ms_p50: {percentile(latencies_ms, 0.50):.4f}")
    print(f"latency_ms_p95: {percentile(latencies_ms, 0.95):.4f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/example_head_aligned.yaml")
    parser.add_argument("--num-points", "--points", dest="num_points", type=int, default=307200)
    parser.add_argument("--iters", "--iterations", dest="iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.device)
    for with_rgb in (False, True):
        point_cloud = make_point_cloud(args.num_points, with_rgb=with_rgb, device=device)
        run_case(point_cloud, iters=args.iters, warmup=args.warmup, config_path=args.config)
        if not with_rgb:
            print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
