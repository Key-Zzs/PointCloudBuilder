from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from pointcloud_builder.reconstruction_benchmark import (
    benchmark_reconstruction_scenarios,
    benchmark_timing_overhead,
)
from pointcloud_builder.reconstruction_timing import (
    ReconstructionTiming,
    aggregate_reconstruction_timings,
)


def _sample(scale: float = 1.0) -> ReconstructionTiming:
    return ReconstructionTiming(
        path="current_snapshot",
        stages_ms={
            "concatenate_ms": 1.0 * scale,
            "workspace_crop_ms": 2.0 * scale,
            "voxel_fusion_ms": 3.0 * scale,
            "global_sampling_ms": 4.0 * scale,
            "raw_to_world_concatenated_ms": 10.0 * scale,
            "raw_to_world_fused_ms": 15.0 * scale,
            "raw_to_world_cropped_ms": 12.0 * scale,
            "raw_to_world_sampled_ms": 19.0 * scale,
        },
        per_camera_ms={
            "camera_a": {
                "frame_adapter_ms": 0.1 * scale,
                "depth_resolution_ms": 5.0 * scale,
                "deprojection_ms": 1.0 * scale,
                "local_crop_ms": 0.2 * scale,
                "workspace_transform_ms": 0.3 * scale,
                "raw_to_workspace_per_camera_ms": 7.0 * scale,
            }
        },
        frame_match_ms=0.5 * scale,
        capture_inclusive_total_ms=20.0 * scale,
    )


def test_reconstruction_timing_round_trip_and_summary_are_json_safe() -> None:
    first = _sample()
    assert ReconstructionTiming.from_dict(first.to_dict()) == first
    summary = aggregate_reconstruction_timings([first, _sample(2.0)])
    fusion = summary["processing_only"]["stages_ms"]["raw_to_world_fused_ms"]
    assert fusion == {"p50": 22.5, "p95": 29.25, "mean": 22.5, "max": 30.0}
    assert summary["capture_inclusive"]["frame_match_ms"]["p50"] == 0.75


def test_reconstruction_timing_rejects_unknown_and_async_invalid_values() -> None:
    with pytest.raises(ValueError, match="unknown"):
        ReconstructionTiming(path="current_snapshot", stages_ms={"total": 1.0})
    with pytest.raises(ValueError, match="finite"):
        ReconstructionTiming(
            path="persistent_tsdf_update",
            stages_ms={"map_update_total_ms": float("nan")},
        )


def test_processing_only_summary_omits_capture_section() -> None:
    sample = ReconstructionTiming(
        path="tsdf_extraction", stages_ms={"extract_point_cloud_ms": 1.0}
    )
    assert "capture_inclusive" not in aggregate_reconstruction_timings([sample])


def test_summary_rejects_inconsistent_stage_sets() -> None:
    first = _sample()
    second = ReconstructionTiming(
        path="current_snapshot",
        stages_ms={
            name: value
            for name, value in first.stages_ms.items()
            if name != "global_sampling_ms"
        },
        per_camera_ms=first.per_camera_ms,
    )
    with pytest.raises(ValueError, match="inconsistent stage keys"):
        aggregate_reconstruction_timings([first, second])


class _FakePipeline:
    def __init__(self, multiplier: float) -> None:
        self.multiplier = multiplier
        self.indices: list[int] = []
        self.processor = SimpleNamespace(timing_enabled=True)

    def build(self, index: int) -> SimpleNamespace:
        self.indices.append(index)
        timing = _sample(self.multiplier).to_dict()
        points = torch.zeros((index + 1, 3))
        return SimpleNamespace(
            timing_report_ms={"reconstruction": timing},
            concatenated=SimpleNamespace(points=points),
            workspace_cropped=SimpleNamespace(points=points),
            fused=SimpleNamespace(points=points),
            sampled=SimpleNamespace(points=points),
        )


def test_world_reconstruction_benchmark_uses_same_inputs_and_three_scenarios() -> None:
    pipelines: dict[str, _FakePipeline] = {}

    def factory(name: str, multiplier: float):
        def build() -> _FakePipeline:
            pipeline = _FakePipeline(multiplier)
            pipelines[name] = pipeline
            return pipeline

        return build

    report = benchmark_reconstruction_scenarios(
        {
            "reconstruction_only": factory("reconstruction_only", 1.0),
            "reconstruction_crop": factory("reconstruction_crop", 2.0),
            "reconstruction_crop_sampling": factory(
                "reconstruction_crop_sampling", 3.0
            ),
        },
        frame_indices=(2, 3),
        warmup=1,
    )
    assert report["same_inputs"]
    assert set(report["scenarios"]) == {
        "reconstruction_only",
        "reconstruction_crop",
        "reconstruction_crop_sampling",
    }
    assert all(pipeline.indices == [2, 2, 3] for pipeline in pipelines.values())


def test_timing_overhead_report_compares_enabled_and_disabled_paths() -> None:
    report = benchmark_timing_overhead(
        lambda: _FakePipeline(1.0),
        lambda: _FakePipeline(1.0),
        frame_indices=(1, 2, 3),
        warmup=1,
    )
    assert report["maximum_percent"] == 5.0
    assert set(report["timing_enabled_ms"]) == {"p50", "p95", "mean", "max"}
    assert isinstance(report["passed"], bool)
