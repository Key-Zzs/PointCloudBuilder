"""Per-camera fixed-pixel consistency state for guarded continuous TSDF updates."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from pointcloud_builder.mapping.config import TsdfDynamicConfig
from pointcloud_builder.mapping.types import DynamicMaskReport, RigDepthObservation


@dataclass(frozen=True)
class GuardedDepthDecision:
    observation: RigDepthObservation
    dynamic_mask: np.ndarray
    persistent_mask: np.ndarray
    report: DynamicMaskReport


class GuardedDepthFilter:
    """Keep transient/moving pixels out until a fixed surface persists."""

    def __init__(self, config: TsdfDynamicConfig) -> None:
        self.config = config
        self._candidate_depth: dict[str, np.ndarray] = {}
        self._candidate_count: dict[str, np.ndarray] = {}

    def apply(
        self,
        observation: RigDepthObservation,
        predicted_depth_m: np.ndarray,
    ) -> GuardedDepthDecision:
        observed = observation.metric_depth
        predicted = np.asarray(predicted_depth_m, dtype=np.float32)
        if predicted.shape != observed.shape or not np.isfinite(predicted).all():
            raise ValueError(
                "predicted depth must be finite with the observation shape"
            )
        name = observation.camera_name
        candidate_depth = self._candidate_depth.get(name)
        candidate_count = self._candidate_count.get(name)
        if candidate_depth is None:
            candidate_depth = np.zeros_like(observed, dtype=np.float32)
            candidate_count = np.zeros_like(observed, dtype=np.uint16)
        assert candidate_count is not None
        observed_valid = observed > 0
        predicted_valid = predicted > 0
        residual = observed - predicted
        comparable = observed_valid & predicted_valid
        background = comparable & (np.abs(residual) <= self.config.residual_threshold_m)
        candidate = observed_valid & ~background
        consistent = (
            candidate
            & (candidate_count > 0)
            & (
                np.abs(observed - candidate_depth)
                <= self.config.consistency_tolerance_m
            )
        )
        new_candidate = candidate & ~consistent
        candidate_count = np.where(
            consistent,
            np.minimum(candidate_count.astype(np.uint32) + 1, np.iinfo(np.uint16).max),
            np.where(new_candidate, 1, 0),
        ).astype(np.uint16)
        # Measure a run against the depth that started it. Comparing only to
        # the preceding frame would let a slowly drifting object become
        # "persistent" through small pairwise steps and leave a ghost trail.
        candidate_depth = np.where(
            consistent, candidate_depth, np.where(new_candidate, observed, 0.0)
        ).astype(np.float32)
        persistent = candidate & (candidate_count >= self.config.persistence_frames)
        integrate = np.zeros_like(observed_valid)
        if self.config.integrate_background_consistent:
            integrate |= background
        if self.config.integrate_persistent_new_surface:
            integrate |= persistent
        filtered = observation.depth.copy()
        filtered[~integrate] = 0
        self._candidate_depth[name] = candidate_depth
        self._candidate_count[name] = candidate_count
        transient = candidate & ~persistent
        residual_values = np.abs(residual[comparable])
        report = DynamicMaskReport(
            camera_name=name,
            total_pixels=int(observed.size),
            background_consistent_pixels=int(background.sum()),
            transient_dynamic_pixels=int(transient.sum()),
            persistent_candidate_pixels=int(persistent.sum()),
            newly_persistent_pixels=int(
                (candidate & (candidate_count == self.config.persistence_frames)).sum()
            ),
            integrated_pixels=int(integrate.sum()),
            residual_median_m=(
                None if not residual_values.size else float(np.median(residual_values))
            ),
            residual_p95_m=(
                None
                if not residual_values.size
                else float(np.quantile(residual_values, 0.95))
            ),
            metrics={
                "background_consistent_ratio": float(background.mean()),
                "transient_dynamic_ratio": float(transient.mean()),
                "persistent_candidate_ratio": float(persistent.mean()),
                "integrated_pixel_ratio": float(integrate.mean()),
            },
        )
        filtered_observation = replace(
            observation,
            depth=filtered,
            valid_mask=np.isfinite(filtered) & (filtered > 0),
        )
        dynamic_mask = transient.copy()
        dynamic_mask.setflags(write=False)
        persistent = persistent.copy()
        persistent.setflags(write=False)
        return GuardedDepthDecision(
            observation=filtered_observation,
            dynamic_mask=dynamic_mask,
            persistent_mask=persistent,
            report=report,
        )

    def reset(self) -> None:
        self._candidate_depth.clear()
        self._candidate_count.clear()
