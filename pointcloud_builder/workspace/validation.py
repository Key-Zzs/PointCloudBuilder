"""Target-agnostic expected-plane validation in workspace coordinates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from pointcloud_builder.workspace.types import WorkspacePointCloud


@dataclass(frozen=True)
class ExpectedPlaneRegion:
    frame: str
    x: tuple[float, float]
    y: tuple[float, float]
    expected_z_m: float
    z_search_range_m: tuple[float, float]

    def __post_init__(self) -> None:
        if not self.frame.strip():
            raise ValueError("plane frame must be non-empty")
        for name, bounds in (("x", self.x), ("y", self.y), ("z_search_range_m", self.z_search_range_m)):
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError(f"{name} must be an ordered two-element range")
        if not math.isfinite(self.expected_z_m):
            raise ValueError("expected_z_m must be finite")


@dataclass(frozen=True)
class PlaneMetrics:
    point_count: int
    signed_z_bias_m: float
    median_abs_z_m: float
    p95_abs_z_m: float
    rmse_m: float
    fitted_plane_normal: tuple[float, float, float]
    normal_angle_to_expected_deg: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def select_expected_plane_points(
    cloud: WorkspacePointCloud,
    region: ExpectedPlaneRegion,
) -> torch.Tensor:
    if cloud.frame != region.frame:
        raise ValueError(
            f"plane region frame {region.frame!r} does not match cloud frame {cloud.frame!r}"
        )
    xyz = cloud.points[:, :3]
    relative_z = xyz[:, 2] - region.expected_z_m
    mask = (
        (xyz[:, 0] >= region.x[0])
        & (xyz[:, 0] <= region.x[1])
        & (xyz[:, 1] >= region.y[0])
        & (xyz[:, 1] <= region.y[1])
        & (relative_z >= region.z_search_range_m[0])
        & (relative_z <= region.z_search_range_m[1])
    )
    return cloud.points[mask]


def evaluate_expected_plane(
    cloud: WorkspacePointCloud,
    region: ExpectedPlaneRegion,
) -> PlaneMetrics:
    points = select_expected_plane_points(cloud, region)[:, :3]
    count = int(points.shape[0])
    if count < 3:
        raise ValueError(f"expected-plane region has too few points: {count}")
    z_error = points[:, 2] - region.expected_z_m
    absolute = torch.abs(z_error)
    centered = points - points.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / float(count)
    _, eigenvectors = torch.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    if normal[2] < 0:
        normal = -normal
    cosine = torch.clamp(normal[2], -1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cosine))
    return PlaneMetrics(
        point_count=count,
        signed_z_bias_m=float(torch.median(z_error).item()),
        median_abs_z_m=float(torch.median(absolute).item()),
        p95_abs_z_m=float(torch.quantile(absolute, 0.95).item()),
        rmse_m=float(torch.sqrt(torch.mean(z_error * z_error)).item()),
        fitted_plane_normal=tuple(float(value) for value in normal.tolist()),
        normal_angle_to_expected_deg=float(angle.item()),
    )
