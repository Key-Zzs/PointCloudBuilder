"""Offline sampled point-cloud visualization from an NPZ RGB-D frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pointcloud_builder import PointCloudBuilder
from pointcloud_builder.visualization import save_ascii_ply, show_open3d


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    builder = PointCloudBuilder.from_yaml(args.config)
    frame = dict(np.load(args.frame))
    point_cloud, _ = builder.from_recorded_frame(frame)
    if args.output:
        save_ascii_ply(point_cloud, Path(args.output))
    if args.show:
        show_open3d(point_cloud)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
