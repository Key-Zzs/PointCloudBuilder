"""Deterministic offline rig orchestration with optional snapshot voxel fusion."""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any

from pointcloud_builder.rig.config import RigConfig
from pointcloud_builder.rig.frame_matcher import (
    match_exact_index,
    match_nearest_host_timestamp,
)
from pointcloud_builder.rig.processor import RigFrameProcessor
from pointcloud_builder.rig.types import RigBuildResult
from pointcloud_builder.rig.validation import validate_rig_runtimes


@dataclass(frozen=True)
class RigCameraRuntime:
    source: Any
    pipeline: Any
    provenance: dict[str, Any]


class OfflineRigPipeline:
    """Match, independently deproject, and canonically concatenate rig frames."""

    def __init__(
        self, config: RigConfig, runtimes: dict[str, RigCameraRuntime]
    ) -> None:
        validate_rig_runtimes(config, runtimes)
        self.config = config
        self.runtimes = dict(runtimes)
        self.canonical_order = tuple(sorted(self.runtimes))
        self.processor = RigFrameProcessor(config, self.runtimes)

    def build(self, index: int) -> RigBuildResult:
        total_start = time.perf_counter()
        sources = {name: runtime.source for name, runtime in self.runtimes.items()}
        reference = self.config.timing.reference_camera or self.canonical_order[0]
        match_start = time.perf_counter()
        if self.config.timing.mode == "exact_index":
            frame_set = match_exact_index(sources, index, reference_camera=reference)
        else:
            frame_set = match_nearest_host_timestamp(
                sources,
                index,
                reference_camera=reference,
                maximum_skew_ms=self.config.timing.maximum_skew_ms,
            )
        match_ms = (time.perf_counter() - match_start) * 1000.0
        frame_set = replace(
            frame_set,
            match_sequence_index=index,
            matching_policy=self.config.timing.mode,
            metadata={
                **frame_set.metadata,
                "source": "offline",
                "reference_index": index,
            },
        )
        return self.processor.process_frame_set(
            frame_set,
            frame_match_ms=match_ms,
            total_start_s=total_start,
        )
