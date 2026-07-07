"""Offline visualization helpers.

This module is intentionally separate from the real-time builder path. Open3D is
imported lazily inside functions so deployment code can use PointCloudBuilder
without visualization dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from pointcloud_builder.types import Tensor


def detach_to_cpu(point_cloud: Tensor) -> Tensor:
    """Detach a point cloud tensor and move it to CPU."""

    return point_cloud.detach().to(device=torch.device("cpu"))


def save_ascii_ply(point_cloud: Tensor, path: str | Path) -> None:
    """Save XYZ or XYZRGB point cloud data as an ASCII PLY file."""

    pc = detach_to_cpu(point_cloud)
    if pc.ndim != 2 or pc.shape[-1] not in {3, 6}:
        raise ValueError("point_cloud must have shape N x 3 or N x 6")
    output_path = Path(path)
    has_rgb = pc.shape[-1] == 6
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {pc.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_rgb:
        header.extend(["property uchar red", "property uchar green", "property uchar blue"])
    header.append("end_header")
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(header))
        f.write("\n")
        for row in pc:
            xyz = row[:3].tolist()
            if has_rgb:
                rgb = torch.clamp(row[3:6] * 255.0, 0.0, 255.0).to(torch.uint8).tolist()
                f.write(f"{xyz[0]} {xyz[1]} {xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]}\n")
            else:
                f.write(f"{xyz[0]} {xyz[1]} {xyz[2]}\n")


def show_open3d(point_cloud: Tensor) -> None:
    """Display a point cloud with Open3D for offline debugging."""

    import open3d as o3d  # type: ignore[import-not-found]

    pc = detach_to_cpu(point_cloud)
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(pc[:, :3].numpy())
    if pc.shape[-1] == 6:
        geometry.colors = o3d.utility.Vector3dVector(pc[:, 3:6].clamp(0.0, 1.0).numpy())
    o3d.visualization.draw_geometries([geometry])


def summarize_point_cloud(point_cloud: Tensor) -> dict[str, Any]:
    """Return basic statistics for offline tests and logs."""

    pc = detach_to_cpu(point_cloud)
    return {
        "shape": tuple(int(v) for v in pc.shape),
        "min_xyz": pc[:, :3].amin(dim=0).tolist() if pc.numel() else [0.0, 0.0, 0.0],
        "max_xyz": pc[:, :3].amax(dim=0).tolist() if pc.numel() else [0.0, 0.0, 0.0],
    }
