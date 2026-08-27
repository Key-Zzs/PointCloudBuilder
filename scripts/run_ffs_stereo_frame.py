#!/usr/bin/env python3
"""Run one generic stereo-IR frame through FFS and PointCloudBuilder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visualize_ffs_stereo_pipeline import save_pipeline_artifacts

from pointcloud_builder import PointCloudBuilder
from pointcloud_builder.frame_io import load_frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    builder = PointCloudBuilder.from_yaml(args.config)
    frame = load_frame(args.input)
    ffs = builder.config.depth_source.ffs
    if ffs is None:
        raise ValueError("FFS smoke config must use depth_source.mode=ffs_stereo")
    missing = [key for key in (ffs.left_key, ffs.right_key) if key not in frame]
    if missing:
        raise KeyError(f"Stereo FFS input is missing configured keys: {missing}")
    perception, meta = builder.build_perception_stages(frame)
    print(json.dumps(save_pipeline_artifacts(perception, meta, Path(args.output_dir), no_show=args.no_show), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
