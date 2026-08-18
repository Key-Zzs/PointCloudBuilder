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
    color_image_size: tuple[int, int] | None = None
    source_frame: str = "depth"


@dataclass(frozen=True)
class ColorViewVisibilityFilter:
    """Configurable RGB-view z-buffer for projected metric points.

    The filter operates on the calibrated color-camera ``Z`` coordinate after
    the existing depth-to-color transform.  ``epsilon_z`` is in metres and
    keeps points within a configured tolerance of the nearest point in each
    rounded RGB pixel.  Disabled mode deliberately returns the legacy
    projection-valid mask without requiring color-frame XYZ values.
    """

    enabled: bool = True
    epsilon_z: float = 0.005

    def __post_init__(self) -> None:
        if self.epsilon_z < 0.0:
            raise ValueError("visibility_filter.epsilon_z must be non-negative")

    def visible_mask(self, projection: ProjectionMap) -> Tensor:
        """Return source-point visibility under the RGB-camera viewpoint."""

        valid = projection.valid
        if not self.enabled or not bool(valid.any()):
            return valid.clone()
        if projection.color_xyz is None or projection.color_image_size is None:
            raise ValueError("RGB-view visibility requires color-frame XYZ and image size")

        width, height = projection.color_image_size
        if width <= 0 or height <= 0:
            raise ValueError("RGB projection image size must be positive")
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        uv = projection.uv[valid_indices]
        linear_pixels = uv[:, 1] * int(width) + uv[:, 0]
        z_color = projection.color_xyz[valid_indices, 2]
        z_min = torch.full(
            (int(width) * int(height),),
            float("inf"),
            dtype=z_color.dtype,
            device=z_color.device,
        )
        z_min.scatter_reduce_(0, linear_pixels, z_color, reduce="amin", include_self=True)
        visible = torch.zeros_like(valid)
        visible[valid_indices] = z_color <= z_min[linear_pixels] + float(self.epsilon_z)
        return visible


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
    return ProjectionMap(
        uv=uv,
        valid=valid,
        color_xyz=color_xyz,
        color_image_size=(int(intrinsics.width), int(intrinsics.height)),
        source_frame=source_frame,
    )


def lift_binary_mask(
    points: Tensor,
    projection: ProjectionMap,
    binary_mask: Tensor,
    *,
    visibility_filter: ColorViewVisibilityFilter | None = None,
) -> tuple[Tensor, Tensor]:
    """Lift one RGB binary mask into the visible source-frame point subset."""

    if binary_mask.ndim != 2:
        raise ValueError("binary_mask must have shape H x W")
    if len(points) != len(projection.uv):
        raise ValueError("projection cardinality must match points")
    mask = binary_mask.to(device=points.device, dtype=torch.bool)
    selected = torch.zeros(len(points), dtype=torch.bool, device=points.device)
    valid = (
        projection.valid
        if visibility_filter is None
        else visibility_filter.visible_mask(projection)
    )
    if bool(valid.any()):
        uv = projection.uv[valid]
        selected[valid] = mask[uv[:, 1], uv[:, 0]]
    return points[selected], selected
