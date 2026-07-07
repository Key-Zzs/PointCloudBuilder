"""Minimal live-frame conversion example."""

from __future__ import annotations

import torch

from pointcloud_builder import PointCloudBuilder


def main() -> int:
    builder = PointCloudBuilder.from_yaml("configs/example_head_aligned.yaml")
    frame = {
        "depth": torch.ones((builder.camera.height, builder.camera.width), dtype=torch.float32),
        "color": torch.ones((builder.camera.height, builder.camera.width, 3), dtype=torch.float32),
    }
    pc, meta = builder.from_live_frame(frame)
    print(pc.shape)
    print(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
