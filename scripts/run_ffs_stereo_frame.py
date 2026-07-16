#!/usr/bin/env python3
"""Run one generic stereo-IR frame through FFS and PointCloudBuilder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pointcloud_builder import PointCloudBuilder
from pointcloud_builder.frame_io import load_frame
from visualize_ffs_stereo_pipeline import save_pipeline_artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    builder = PointCloudBuilder.from_yaml(args.config)
    frame = load_frame(args.input)
    if "left_ir" not in frame or "right_ir" not in frame:
        raise KeyError("Stereo FFS input must contain left_ir and right_ir")
    perception, meta = builder.build_perception_stages(frame)
    print(json.dumps(save_pipeline_artifacts(perception, meta, Path(args.output_dir), no_show=args.no_show), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
