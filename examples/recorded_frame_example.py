"""Minimal recorded-frame conversion example."""

from __future__ import annotations

import torch

from pointcloud_builder import PointCloudBuilder


def main() -> int:
    builder = PointCloudBuilder.from_yaml("configs/example_head_depth_raw.yaml")
    frame = {
        "depth": torch.ones((builder.camera.height, builder.camera.width), dtype=torch.float32),
    }
    pc, meta = builder.from_recorded_frame(frame)
    print(pc.shape)
    print(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
