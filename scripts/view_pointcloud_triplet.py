#!/usr/bin/env python3
"""Open raw, cropped, and sampled PLY point clouds in three windows."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any


STAGES = ("raw", "cropped", "sampled")


def _stage_paths(input_dir: Path) -> dict[str, Path]:
    paths = {stage: input_dir / f"{stage}.ply" for stage in STAGES}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing point-cloud stage files: {missing}")
    return paths


def show_triplet(input_dir: str | Path, *, point_size: float = 2.0) -> None:
    """Keep three Open3D visualizer windows responsive until they are closed."""

    import open3d as o3d  # type: ignore[import-not-found]

    directory = Path(input_dir).expanduser().resolve()
    paths = _stage_paths(directory)
    viewers: list[Any | None] = []
    try:
        for index, stage in enumerate(STAGES):
            cloud = o3d.io.read_point_cloud(str(paths[stage]))
            if cloud.is_empty():
                raise ValueError(f"Point cloud is empty: {paths[stage]}")
            viewer = o3d.visualization.Visualizer()
            created = viewer.create_window(
                window_name=f"PointCloudBuilder: {stage}",
                width=620,
                height=520,
                left=40 + index * 640,
                top=80,
            )
            if not created:
                raise RuntimeError(f"Could not create Open3D window for {stage}")
            viewer.add_geometry(cloud)
            render = viewer.get_render_option()
            if render is not None:
                render.point_size = float(point_size)
            viewers.append(viewer)

        while any(viewer is not None for viewer in viewers):
            for index, viewer in enumerate(viewers):
                if viewer is None:
                    continue
                if viewer.poll_events():
                    viewer.update_renderer()
                else:
                    viewer.destroy_window()
                    viewers[index] = None
            time.sleep(0.01)
    finally:
        for viewer in viewers:
            if viewer is not None:
                viewer.destroy_window()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing raw.ply, cropped.ply, and sampled.ply")
    parser.add_argument("--point-size", type=float, default=2.0)
    args = parser.parse_args()
    if args.point_size <= 0.0:
        parser.error("--point-size must be positive")
    show_triplet(args.input_dir, point_size=args.point_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
