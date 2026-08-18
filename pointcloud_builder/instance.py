"""Generic SAM-mask lifting for instance-dense and instance-sparse diagnostics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from pointcloud_builder.projection import ProjectionMap, lift_binary_mask
from pointcloud_builder.sampling import sample_point_cloud
from pointcloud_builder.segmentation.types import InstanceMask
from pointcloud_builder.support_plane import SupportPlane, filter_support_plane
from pointcloud_builder.types import Tensor


@dataclass(frozen=True)
class InstanceCloud:
    """Only currently observed object geometry; no unobserved surface is added."""

    mask: InstanceMask
    points_before_cleanup: Tensor
    clean_visible_points: Tensor
    sampled_points: Tensor
    source_dense_point_count: int
    sampled_unique_point_count: int
    selection_mask: Tensor
    # These diagnostics deliberately describe the pre-padding evidence.  A
    # fixed-size sampled cloud alone cannot distinguish a 256-point
    # observation from six source points repeated to 256 rows.
    source_stage_point_count: int
    object_selected_count: int
    object_selected_ratio: float
    padded_count: int


@dataclass(frozen=True)
class InstanceFrameResult:
    mode: str
    instances: tuple[InstanceCloud, ...]
    expected_instance_violations: tuple[str, ...]


def _validated_masks(
    masks: Iterable[InstanceMask],
    expected_instances: dict[str, int] | None,
) -> tuple[list[InstanceMask], tuple[str, ...]]:
    items = [item for item in masks if item.valid]
    violations: list[str] = []
    if expected_instances:
        counts = Counter(item.concept_id for item in items)
        for concept, expected in sorted(expected_instances.items()):
            actual = counts.get(concept, 0)
            if actual != expected:
                violations.append(f"concept={concept!r} expected_instances={expected}, observed={actual}")
    return items, tuple(violations)


def _cleanup(points: Tensor, plane: SupportPlane | None) -> Tensor:
    finite = torch.isfinite(points[:, :3]).all(dim=1)
    clean = points[finite]
    if plane is not None and len(clean):
        clean, _ = filter_support_plane(clean, plane)
    return clean


def _sampling_diagnostics(sampled: Tensor, sampling_meta: object) -> tuple[int, int]:
    """Return (unique, padded) counts without inferring support from padding."""

    unique = int(torch.unique(sampled[:, :3], dim=0).shape[0]) if len(sampled) else 0
    candidate_count = int(getattr(sampling_meta, "get", lambda _key, default=0: default)("candidate_count", 0))
    # ``candidate_count`` is after an optional voxel reduction, which is the
    # actual set consumed by the sampler.  It is therefore the only count from
    # which repeat/zero padding can be stated exactly.
    padded = max(0, int(len(sampled)) - min(int(len(sampled)), candidate_count))
    return unique, padded


def build_instance_dense(
    *,
    raw_dense_points: Tensor,
    projection: ProjectionMap,
    masks: Iterable[InstanceMask],
    sampling_config: object,
    support_plane: SupportPlane | None,
    expected_instances: dict[str, int] | None = None,
) -> InstanceFrameResult:
    """Lift masks from raw dense metric geometry, then clean and sample."""

    selected_masks, violations = _validated_masks(masks, expected_instances)
    instances: list[InstanceCloud] = []
    for item in selected_masks:
        binary = torch.from_numpy(np.asarray(item.binary_mask, dtype=bool)).to(raw_dense_points.device)
        visible, selected = lift_binary_mask(raw_dense_points, projection, binary)
        clean = _cleanup(visible, support_plane)
        sampled, sampling_meta = sample_point_cloud(clean, sampling_config)  # type: ignore[arg-type]
        unique, padded = _sampling_diagnostics(sampled, sampling_meta)
        selected_count = int(len(visible))
        source_count = int(len(raw_dense_points))
        instances.append(InstanceCloud(
            mask=item,
            points_before_cleanup=visible,
            clean_visible_points=clean,
            sampled_points=sampled,
            source_dense_point_count=int(len(clean)),
            sampled_unique_point_count=unique,
            selection_mask=selected,
            source_stage_point_count=source_count,
            object_selected_count=selected_count,
            object_selected_ratio=float(selected_count / source_count) if source_count else 0.0,
            padded_count=padded,
        ))
    return InstanceFrameResult("instance_dense", tuple(instances), violations)


def build_instance_sparse(
    *,
    workspace_sampled_points: Tensor,
    projection: ProjectionMap,
    masks: Iterable[InstanceMask],
    sampling_config: object,
    expected_instances: dict[str, int] | None = None,
) -> InstanceFrameResult:
    """Select SAM-covered points only after ordinary workspace sampling."""

    selected_masks, violations = _validated_masks(masks, expected_instances)
    instances: list[InstanceCloud] = []
    for item in selected_masks:
        binary = torch.from_numpy(np.asarray(item.binary_mask, dtype=bool)).to(workspace_sampled_points.device)
        selected_points, selected = lift_binary_mask(workspace_sampled_points, projection, binary)
        sampled, sampling_meta = sample_point_cloud(selected_points, sampling_config)  # type: ignore[arg-type]
        unique, padded = _sampling_diagnostics(sampled, sampling_meta)
        selected_count = int(len(selected_points))
        source_count = int(len(workspace_sampled_points))
        instances.append(InstanceCloud(
            mask=item,
            points_before_cleanup=selected_points,
            clean_visible_points=selected_points,
            sampled_points=sampled,
            source_dense_point_count=selected_count,
            sampled_unique_point_count=unique,
            selection_mask=selected,
            source_stage_point_count=source_count,
            object_selected_count=selected_count,
            object_selected_ratio=float(selected_count / source_count) if source_count else 0.0,
            padded_count=padded,
        ))
    return InstanceFrameResult("instance_sparse", tuple(instances), violations)
