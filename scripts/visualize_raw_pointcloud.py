"""Offline raw point-cloud visualization from an NPZ RGB-D frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pointcloud_builder import PointCloudBuilder
from pointcloud_builder.visualization import save_ascii_ply, show_open3d


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--frame", required=True, help="NPZ with depth and optional color arrays.")
    parser.add_argument("--output", default=None, help="Optional ASCII PLY output path.")
    parser.add_argument("--show", action="store_true", help="Show with Open3D.")
    args = parser.parse_args()

    builder = PointCloudBuilder.from_yaml(args.config)
    frame = dict(np.load(args.frame))
    stages, _ = builder.build_stages(frame)
    point_cloud = stages["raw"]
    if args.output:
        save_ascii_ply(point_cloud, Path(args.output))
    if args.show:
        show_open3d(point_cloud)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
