"""Offline raw point-cloud visualization from an NPZ or NPY RGB-D frame."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from pointcloud_builder import PointCloudBuilder
from pointcloud_builder.visualization import save_ascii_ply, show_open3d


def load_frame(path: str | Path) -> dict[str, Any]:
    """Load depth and optional RGB arrays from an NPZ or NPY file."""

    input_path = Path(path)
    if input_path.suffix == ".npz":
        data = np.load(input_path)
        frame: dict[str, Any] = {"depth": data["depth"]}
        if "rgb" in data:
            frame["rgb"] = data["rgb"]
        elif "color" in data:
            frame["rgb"] = data["color"]
        return frame
    if input_path.suffix == ".npy":
        data = np.load(input_path, allow_pickle=True)
        if data.shape == () and isinstance(data.item(), dict):
            raw = data.item()
            if "color" in raw and "rgb" not in raw:
                raw["rgb"] = raw["color"]
            return raw
        return {"depth": data}
    raise ValueError(f"Unsupported input file extension: {input_path.suffix}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="NPZ/NPY with depth and optional rgb arrays.")
    parser.add_argument("--output", default=None, help="Optional ASCII PLY output path.")
    parser.add_argument("--no-show", action="store_true", help="Disable Open3D window.")
    args = parser.parse_args()

    builder = PointCloudBuilder.from_yaml(args.config)
    frame = load_frame(args.input)
    point_cloud, meta = builder.from_recorded_frame(frame)
    print(meta)
    if args.output:
        save_ascii_ply(point_cloud, Path(args.output))
    if not args.no_show:
        show_open3d(point_cloud)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
