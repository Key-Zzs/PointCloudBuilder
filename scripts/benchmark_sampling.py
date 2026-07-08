"""Benchmark fixed-size point-cloud sampling modes."""

from __future__ import annotations

import argparse
from time import perf_counter

import torch

from pointcloud_builder.config import SamplingConfig
from pointcloud_builder.sampling import sample_point_cloud
from pointcloud_builder.utils import resolve_device

MODES = ("fps", "stride", "random", "voxel", "voxel_random", "voxel_fps")
FPS_MODES = {"fps", "voxel_fps"}
CPU_FPS_MAX_INPUT_POINTS = 20_000
CPU_FPS_MAX_ITERS = 3
CPU_FPS_MAX_WARMUP = 1


def synchronize_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA work when benchmarking on GPU."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], q: float) -> float:
    """Return a percentile from a non-empty list."""

    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def make_point_cloud(num_points: int, with_rgb: bool, device: torch.device) -> torch.Tensor:
    """Generate a random XYZ or XYZRGB point cloud."""

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
    mode: str,
    target_num_points: int,
    iters: int,
    warmup: int,
) -> None:
    """Benchmark one sampler/memory-layout pair."""

    requested_input_points = int(point_cloud.shape[0])
    requested_iters = iters
    requested_warmup = warmup
    if point_cloud.device.type == "cpu" and mode in FPS_MODES and requested_input_points > CPU_FPS_MAX_INPUT_POINTS:
        point_cloud = point_cloud[:CPU_FPS_MAX_INPUT_POINTS].contiguous()
        iters = min(iters, CPU_FPS_MAX_ITERS)
        warmup = min(warmup, CPU_FPS_MAX_WARMUP)
        print(
            "cpu_fallback: FPS is quadratic in target count; "
            f"using effective_input_points={point_cloud.shape[0]}, "
            f"effective_iters={iters}, effective_warmup={warmup}"
        )

    config = SamplingConfig(
        enabled=True,
        mode=mode,  # type: ignore[arg-type]
        num_points=target_num_points,
        stride=2,
        voxel_size=0.01,
        seed=42,
        deterministic=True,
        pad_mode="repeat",
    )
    for _ in range(warmup):
        sample_point_cloud(point_cloud, config)
    synchronize_if_cuda(point_cloud.device)

    latencies_ms: list[float] = []
    output_points = 0
    for _ in range(iters):
        start = perf_counter()
        sampled, _ = sample_point_cloud(point_cloud, config)
        synchronize_if_cuda(point_cloud.device)
        latencies_ms.append((perf_counter() - start) * 1000.0)
        output_points = int(sampled.shape[0])

    mean_ms = sum(latencies_ms) / max(len(latencies_ms), 1)
    print(f"device: {point_cloud.device}")
    print(f"mode: {mode}")
    print(f"with_rgb: {point_cloud.shape[1] == 6}")
    print(f"requested_input_points: {requested_input_points}")
    print(f"requested_iters: {requested_iters}")
    print(f"requested_warmup: {requested_warmup}")
    print(f"input_points: {point_cloud.shape[0]}")
    print(f"output_points: {output_points}")
    print(f"iters: {iters}")
    print(f"warmup: {warmup}")
    print(f"latency_ms_mean: {mean_ms:.4f}")
    print(f"latency_ms_p50: {percentile(latencies_ms, 0.50):.4f}")
    print(f"latency_ms_p95: {percentile(latencies_ms, 0.95):.4f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-points", "--points", dest="num_points", type=int, default=50000)
    parser.add_argument("--target-num-points", type=int, default=1024)
    parser.add_argument("--iters", "--iterations", dest="iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    device = resolve_device("cuda")
    for mode in MODES:
        for with_rgb in (False, True):
            point_cloud = make_point_cloud(args.num_points, with_rgb=with_rgb, device=device)
            run_case(
                point_cloud,
                mode=mode,
                target_num_points=args.target_num_points,
                iters=args.iters,
                warmup=args.warmup,
            )
            print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
