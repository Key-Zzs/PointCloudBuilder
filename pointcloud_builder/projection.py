"""Shared metric-3D to RGB-pixel projection primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pointcloud_builder.camera_model import CameraExtrinsics, CameraIntrinsics
from pointcloud_builder.types import Tensor


@dataclass(frozen=True)
class ProjectionMap:
    """Nearest-pixel correspondence for a single source frame.

    ``uv`` is always allocated so downstream mask lifting can retain point
    order; callers must honour ``valid`` before indexing an RGB mask.
    """

    uv: Tensor
    valid: Tensor
    color_xyz: Tensor | None = None
    source_frame: str = "depth"


def project_points_to_color_image(
    points: Tensor,
    *,
    extrinsics: CameraExtrinsics | None,
    intrinsics: CameraIntrinsics,
    source_frame: str = "depth",
) -> ProjectionMap:
    """Project XYZ points into an RGB image without attaching RGB values.

    ``extrinsics=None`` is an explicit identity transform for aligned
    depth/color coordinates.  Raw-depth callers should provide the calibrated
    depth-to-color transform instead of relying on this convenience.
    """

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape N x 3 (or XYZ-prefixed channels)")
    xyz = points[:, :3]
    if extrinsics is None:
        color_xyz = xyz
    else:
        rotation = torch.as_tensor(extrinsics.rotation, dtype=xyz.dtype, device=xyz.device)
        translation = torch.as_tensor(extrinsics.translation, dtype=xyz.dtype, device=xyz.device)
        color_xyz = xyz @ rotation.T + translation
    z = color_xyz[:, 2]
    finite = torch.isfinite(color_xyz).all(dim=1) & (z > 0.0)
    safe_z = torch.where(finite, z, torch.ones_like(z))
    u = color_xyz[:, 0] * float(intrinsics.fx) / safe_z + float(intrinsics.cx)
    v = color_xyz[:, 1] * float(intrinsics.fy) / safe_z + float(intrinsics.cy)
    uv = torch.stack((torch.round(u).to(torch.long), torch.round(v).to(torch.long)), dim=1)
    valid = (
        finite
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < int(intrinsics.width))
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < int(intrinsics.height))
    )
    return ProjectionMap(uv=uv, valid=valid, color_xyz=color_xyz, source_frame=source_frame)


def lift_binary_mask(points: Tensor, projection: ProjectionMap, binary_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Lift one RGB binary mask into the visible source-frame point subset."""

    if binary_mask.ndim != 2:
        raise ValueError("binary_mask must have shape H x W")
    if len(points) != len(projection.uv):
        raise ValueError("projection cardinality must match points")
    mask = binary_mask.to(device=points.device, dtype=torch.bool)
    selected = torch.zeros(len(points), dtype=torch.bool, device=points.device)
    valid = projection.valid
    if bool(valid.any()):
        uv = projection.uv[valid]
        selected[valid] = mask[uv[:, 1], uv[:, 0]]
    return points[selected], selected
