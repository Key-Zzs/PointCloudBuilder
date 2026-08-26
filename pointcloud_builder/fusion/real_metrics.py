"""Real-scene snapshot metrics for multi-camera workspace fusion acceptance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np
import torch

from pointcloud_builder.workspace import ExpectedPlaneRegion, evaluate_expected_plane
from pointcloud_builder.workspace.types import WorkspacePointCloud


@dataclass(frozen=True)
class CubeMetrics:
    point_count: int
    voxel_count: int
    # Axis semantics are stable: (workspace-XY length, workspace-XY width, height).
    # Only the two XY dimensions are normalized so length >= width.
    dimensions_m: tuple[float, float, float]
    mean_absolute_dimension_error_m: float
    maximum_dimension_error_m: float
    support_plane_gap_m: float
    center_workspace_m: tuple[float, float, float]
    yaw_deg: float
    score: float
    candidate_count: int
    ambiguous: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def board_surface_metrics(
    cloud: WorkspacePointCloud, region: ExpectedPlaneRegion
) -> dict[str, Any]:
    base = evaluate_expected_plane(cloud, region).to_dict()
    points = _plane_points_xy_only(cloud, region)
    z_error = points[:, 2] - region.expected_z_m
    if z_error.numel() < 3:
        raise ValueError("board region has too few points")
    base.update(
        {
            "surface_thickness_m": float(
                (torch.quantile(z_error, 0.95) - torch.quantile(z_error, 0.05)).item()
            ),
            "outlier_ratio": float((torch.abs(z_error) > 0.040).float().mean().item()),
        }
    )
    return base


def symmetric_overlap_metrics(
    camera_a: torch.Tensor,
    camera_b: torch.Tensor,
    *,
    roi: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    voxel_size_m: float = 0.005,
    chunk_size: int = 2048,
) -> dict[str, Any]:
    a = voxel_centroids(_select_roi(camera_a[:, :3], roi), voxel_size_m=voxel_size_m)
    b = voxel_centroids(_select_roi(camera_b[:, :3], roi), voxel_size_m=voxel_size_m)
    if a.shape[0] < 3 or b.shape[0] < 3:
        raise ValueError("overlap ROI has too few points from one or both cameras")
    a_to_b = _nearest_distances(a, b, chunk_size=chunk_size)
    b_to_a = _nearest_distances(b, a, chunk_size=chunk_size)
    symmetric = torch.cat((a_to_b, b_to_a))
    return {
        "voxel_size_m": voxel_size_m,
        "camera_a_voxel_count": int(a.shape[0]),
        "camera_b_voxel_count": int(b.shape[0]),
        "a_to_b": _distance_summary(a_to_b),
        "b_to_a": _distance_summary(b_to_a),
        "symmetric": _distance_summary(symmetric),
        "target_passed": bool(
            torch.median(symmetric).item() <= 0.015
            and torch.quantile(symmetric, 0.95).item() <= 0.030
        ),
        "gross_passed": bool(
            torch.median(symmetric).item() < 0.030
            and torch.quantile(symmetric, 0.95).item() < 0.050
        ),
    }


def detect_cube(
    points: torch.Tensor,
    *,
    board_p95_abs_z_m: float,
    nominal_side_m: float = 0.070,
    voxel_size_m: float = 0.005,
    minimum_component_voxels: int = 20,
) -> CubeMetrics:
    xyz = points[:, :3].detach().cpu().to(torch.float64)
    plane_threshold = max(0.010, 2.0 * float(board_p95_abs_z_m))
    selected = xyz[
        (torch.abs(xyz[:, 2]) > plane_threshold)
        & (xyz[:, 2] >= 0.010)
        & (xyz[:, 2] <= 0.120)
    ]
    if selected.shape[0] < minimum_component_voxels:
        raise ValueError("cube extraction has too few above-plane points")
    keys = torch.floor(selected / voxel_size_m).to(torch.int64)
    unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
    components = _connected_components(unique_keys)
    candidates: list[tuple[float, CubeMetrics]] = []
    for component in components:
        if len(component) < minimum_component_voxels:
            continue
        component_mask = torch.zeros(unique_keys.shape[0], dtype=torch.bool)
        component_mask[torch.tensor(component, dtype=torch.long)] = True
        cluster = selected[component_mask[inverse]]
        # Candidate extraction is explicitly voxel-first.  Use one centroid per
        # occupied 5 mm voxel for robust extents so varying per-pixel density does
        # not bias p01/p99 or support-gap estimates.
        cluster_voxels = voxel_centroids(cluster, voxel_size_m=voxel_size_m)
        metrics = _cube_candidate(
            cluster_voxels,
            point_count=int(cluster.shape[0]),
            voxel_count=len(component),
            nominal_side_m=nominal_side_m,
        )
        if all(0.040 <= value <= 0.100 for value in metrics.dimensions_m):
            candidates.append((metrics.score, metrics))
    if not candidates:
        raise ValueError("no cube-like connected component passed candidate bounds")
    candidates.sort(key=lambda item: (item[0], item[1].center_workspace_m))
    best = candidates[0][1]
    ambiguous = bool(
        len(candidates) > 1
        and candidates[1][0] <= candidates[0][0] + 0.010
    )
    return CubeMetrics(
        **{
            **best.to_dict(),
            "candidate_count": len(candidates),
            "ambiguous": ambiguous,
            # An unresolved close-score candidate is not an automatic PASS.  The
            # formal tool still writes visual evidence before failing closed.
            "passed": bool(best.passed and not ambiguous),
        }
    )


def cube_box_voxel_count(
    points: torch.Tensor,
    cube: Mapping[str, Any],
    *,
    minimum_z_m: float,
    voxel_size_m: float = 0.005,
    padding_m: float = 0.015,
) -> int:
    """Count above-plane occupied voxels inside an oriented cube evidence box."""
    xyz = points[:, :3].detach().cpu().numpy()
    center = np.asarray(cube["center_workspace_m"], dtype=np.float64)
    length, width, height = (float(value) for value in cube["dimensions_m"])
    yaw = np.deg2rad(float(cube["yaw_deg"]))
    axes = np.array(
        ((np.cos(yaw), np.sin(yaw)), (-np.sin(yaw), np.cos(yaw)))
    )
    local_xy = (xyz[:, :2] - center[:2]) @ axes.T
    mask = (
        (np.abs(local_xy[:, 0]) <= length / 2.0 + padding_m)
        & (np.abs(local_xy[:, 1]) <= width / 2.0 + padding_m)
        & (xyz[:, 2] > minimum_z_m)
        & (xyz[:, 2] >= center[2] - height / 2.0 - padding_m)
        & (xyz[:, 2] <= center[2] + height / 2.0 + padding_m)
    )
    selected = torch.from_numpy(xyz[mask])
    return int(voxel_centroids(selected, voxel_size_m=voxel_size_m).shape[0])


def voxel_centroids(
    points: torch.Tensor,
    *,
    voxel_size_m: float,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    if not math.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be finite and positive")
    if points.shape[0] == 0:
        return points[:, :3].clone()
    xyz = points[:, :3]
    origin_tensor = torch.tensor(origin, dtype=xyz.dtype, device=xyz.device)
    keys = torch.floor((xyz - origin_tensor) / voxel_size_m).to(torch.int64)
    unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
    sums = torch.zeros((unique.shape[0], 3), dtype=xyz.dtype, device=xyz.device)
    sums.index_add_(0, inverse, xyz)
    counts = torch.bincount(inverse, minlength=unique.shape[0]).to(xyz.dtype)
    return sums / counts[:, None]


def fusion_geometry_metrics(
    concatenated: WorkspacePointCloud,
    fused: WorkspacePointCloud,
    *,
    board_region: ExpectedPlaneRegion,
    voxel_size_m: float,
) -> dict[str, Any]:
    before = board_surface_metrics(concatenated, board_region)
    after = board_surface_metrics(fused, board_region)
    geometry_roi = (board_region.x, board_region.y, (-0.05, 0.120))
    before_voxels = voxel_centroids(
        _select_roi(concatenated.points[:, :3], geometry_roi),
        voxel_size_m=voxel_size_m,
    )
    after_voxels = voxel_centroids(
        _select_roi(fused.points[:, :3], geometry_roi),
        voxel_size_m=voxel_size_m,
    )
    before_to_after = _nearest_distances(before_voxels, after_voxels)
    return {
        "concatenated_board": before,
        "fused_board": after,
        "point_count_before": int(concatenated.points.shape[0]),
        "voxel_count_after": int(fused.points.shape[0]),
        "occupancy_reduction": 1.0
        - float(fused.points.shape[0]) / max(1, int(concatenated.points.shape[0])),
        "completeness_at_1_5_voxels": float(
            (before_to_after <= 1.5 * voxel_size_m).float().mean().item()
        ),
        "thickness_gate_passed": bool(
            after["surface_thickness_m"] <= before["surface_thickness_m"] + 0.001
        ),
        "board_shift_gate_passed": bool(
            after["median_abs_z_m"] <= before["median_abs_z_m"] + 0.005
            and after["p95_abs_z_m"] <= before["p95_abs_z_m"] + 0.005
        ),
    }


def contribution_metrics(provenance: Any) -> dict[str, Any]:
    output_count = int(provenance.output_voxel_count)
    total_points = int(provenance.input_point_count)
    per_camera = {}
    for name in sorted(provenance.per_camera_input_count):
        point_count = int(provenance.per_camera_input_count[name])
        voxel_count = int(provenance.per_camera_unique_voxel_contribution[name])
        per_camera[name] = {
            "workspace_cropped_point_count": point_count,
            "point_contribution_ratio": point_count / max(1, total_points),
            "touched_voxel_count": voxel_count,
            "touched_voxel_ratio": voxel_count / max(1, output_count),
        }
    return {
        "input_point_count": total_points,
        "output_voxel_count": output_count,
        "multi_camera_voxel_count": int(provenance.multi_camera_voxel_count),
        "per_camera": per_camera,
        "passed": all(
            item["point_contribution_ratio"] >= 0.05
            and item["touched_voxel_ratio"] >= 0.05
            for item in per_camera.values()
        ),
    }


def _cube_candidate(
    points: torch.Tensor,
    *,
    point_count: int,
    voxel_count: int,
    nominal_side_m: float,
) -> CubeMetrics:
    xy = points[:, :2].numpy()
    best: tuple[float, float, float, float, np.ndarray, np.ndarray] | None = None
    for yaw_deg in np.linspace(0.0, 89.0, 90):
        angle = math.radians(float(yaw_deg))
        rotation = np.array(
            ((math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle)))
        )
        rotated = xy @ rotation.T
        # XY visibility is strongly view-dependent on a real surface-only cube;
        # a 5--95% rectangle is robust to grazing-ray and voxel-boundary outliers.
        low = np.quantile(rotated, 0.05, axis=0)
        high = np.quantile(rotated, 0.95, axis=0)
        extents = high - low
        area = float(extents[0] * extents[1])
        candidate = (
            area,
            float(extents[0]),
            float(extents[1]),
            float(yaw_deg),
            low,
            high,
        )
        if best is None or candidate[:4] < best[:4]:
            best = candidate
    assert best is not None
    z = points[:, 2].numpy()
    robust_min_z = float(np.quantile(z, 0.01))
    robust_max_z = float(np.quantile(z, 0.99))
    # Frozen definition: support is measured from the workspace plane (z=0),
    # while height is the observed robust z extent.  The plane-removal threshold
    # must never be subtracted from either quantity.
    gap = max(0.0, robust_min_z)
    height = robust_max_z - robust_min_z
    if best[1] >= best[2]:
        length, width, yaw_deg = best[1], best[2], best[3]
    else:
        length, width, yaw_deg = best[2], best[1], (best[3] + 90.0) % 180.0
    dimensions = (length, width, height)
    errors = tuple(abs(value - nominal_side_m) for value in dimensions)
    support_penalty = max(0.0, gap - 0.015)
    score = sum(errors) + support_penalty
    angle = math.radians(best[3])
    rotation = np.array(
        ((math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle)))
    )
    center_xy = ((best[4] + best[5]) / 2.0) @ rotation
    center = (float(center_xy[0]), float(center_xy[1]), (robust_min_z + robust_max_z) / 2.0)
    passed = bool(
        all(0.055 <= value <= 0.085 for value in dimensions)
        and sum(errors) / 3.0 <= 0.010
        and max(errors) <= 0.015
        and gap <= 0.015
    )
    return CubeMetrics(
        point_count=point_count,
        voxel_count=voxel_count,
        dimensions_m=dimensions,
        mean_absolute_dimension_error_m=sum(errors) / 3.0,
        maximum_dimension_error_m=max(errors),
        support_plane_gap_m=gap,
        center_workspace_m=center,
        yaw_deg=yaw_deg,
        score=score,
        candidate_count=1,
        ambiguous=False,
        passed=passed,
    )


def _connected_components(keys: torch.Tensor) -> list[list[int]]:
    rows = [tuple(int(value) for value in row) for row in keys.tolist()]
    lookup = {row: index for index, row in enumerate(rows)}
    remaining = set(range(len(rows)))
    components: list[list[int]] = []
    neighbors = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        stack = [seed]
        component = []
        while stack:
            index = stack.pop()
            component.append(index)
            x, y, z = rows[index]
            for dx, dy, dz in neighbors:
                neighbor = lookup.get((x + dx, y + dy, z + dz))
                if neighbor is not None and neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _plane_points_xy_only(
    cloud: WorkspacePointCloud, region: ExpectedPlaneRegion
) -> torch.Tensor:
    if cloud.frame != region.frame:
        raise ValueError("board region and cloud frames differ")
    xyz = cloud.points[:, :3]
    return xyz[
        (xyz[:, 0] >= region.x[0])
        & (xyz[:, 0] <= region.x[1])
        & (xyz[:, 1] >= region.y[0])
        & (xyz[:, 1] <= region.y[1])
    ]


def _select_roi(
    xyz: torch.Tensor,
    roi: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> torch.Tensor:
    return xyz[
        (xyz[:, 0] >= roi[0][0])
        & (xyz[:, 0] <= roi[0][1])
        & (xyz[:, 1] >= roi[1][0])
        & (xyz[:, 1] <= roi[1][1])
        & (xyz[:, 2] >= roi[2][0])
        & (xyz[:, 2] <= roi[2][1])
    ]


def _nearest_distances(
    query: torch.Tensor, reference: torch.Tensor, *, chunk_size: int = 2048
) -> torch.Tensor:
    if query.shape[0] == 0 or reference.shape[0] == 0:
        raise ValueError("nearest-neighbor inputs must be non-empty")
    return torch.cat(
        [
            torch.cdist(query[start : start + chunk_size], reference).min(dim=1).values
            for start in range(0, query.shape[0], chunk_size)
        ]
    )


def _distance_summary(values: torch.Tensor) -> dict[str, float | int]:
    return {
        "count": int(values.numel()),
        "median_m": float(torch.median(values).item()),
        "p95_m": float(torch.quantile(values, 0.95).item()),
        "rmse_m": float(torch.sqrt(torch.mean(values * values)).item()),
        "mean_m": float(torch.mean(values).item()),
        "max_m": float(torch.max(values).item()),
    }
