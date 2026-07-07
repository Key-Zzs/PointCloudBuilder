"""Benchmark crop operation."""

from __future__ import annotations

import argparse

import torch

from pointcloud_builder.benchmark import benchmark_callable
from pointcloud_builder.config import load_config
from pointcloud_builder.crop import crop_points
from pointcloud_builder.utils import resolve_device


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/example_head_depth_raw.yaml")
    parser.add_argument("--points", type=int, default=307200)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.device)
    points = torch.rand((args.points, 3), dtype=torch.float32, device=device)
    result = benchmark_callable(lambda: crop_points(points, config.crop), iterations=args.iterations)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
