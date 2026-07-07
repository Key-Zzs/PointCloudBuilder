"""Benchmark the full RGB-D to point-cloud pipeline."""

from __future__ import annotations

import argparse

import torch

from pointcloud_builder import PointCloudBuilder
from pointcloud_builder.benchmark import benchmark_callable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/example_head_aligned.yaml")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    builder = PointCloudBuilder.from_yaml(args.config)
    height = builder.camera.height
    width = builder.camera.width
    frame = {
        "depth": torch.ones((height, width), dtype=torch.float32, device=builder.device),
        "rgb": torch.ones((height, width, 3), dtype=torch.float32, device=builder.device),
    }
    result = benchmark_callable(lambda: builder.from_live_frame(frame), iterations=args.iterations)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
