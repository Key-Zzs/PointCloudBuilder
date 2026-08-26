"""Native VoxelBlockGrid persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_volume(volume: Any, path: str | Path) -> None:
    output = Path(path)
    if output.suffix.lower() != ".npz":
        raise ValueError("Open3D TSDF volume path must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    volume.save(str(output))


def load_volume(o3d: Any, path: str | Path) -> Any:
    source = Path(path)
    if source.suffix.lower() != ".npz" or not source.is_file():
        raise ValueError("Open3D TSDF volume must be an existing .npz file")
    return o3d.t.geometry.VoxelBlockGrid.load(str(source))
