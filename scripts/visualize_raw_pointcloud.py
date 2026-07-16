"""Offline raw point-cloud visualization from an NPZ or NPY RGB-D frame."""

from __future__ import annotations

import argparse
from pathlib import Path
from pointcloud_builder import PointCloudBuilder
from pointcloud_builder.frame_io import load_frame
from pointcloud_builder.visualization import save_ascii_ply, show_open3d


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="NPZ/NPY with depth and optional rgb arrays.")
    parser.add_argument("--output", default=None, help="Optional ASCII PLY output path.")
    parser.add_argument("--no-show", action="store_true", help="Disable Open3D window.")
    args = parser.parse_args()

    builder = PointCloudBuilder.from_yaml(args.config)
    frame = load_frame(args.input)
    stages, meta = builder.build_stages(frame)
    point_cloud = stages["raw"]
    print(meta)
    if args.output:
        save_ascii_ply(point_cloud, Path(args.output))
    if not args.no_show:
        show_open3d(point_cloud)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
