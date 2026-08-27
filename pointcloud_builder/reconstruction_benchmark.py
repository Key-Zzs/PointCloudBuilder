"""Scenario runner shared by replay tools and synthetic timing tests."""

from __future__ import annotations

from collections.abc import Callable
import statistics
import time
from typing import Any

from pointcloud_builder.reconstruction_timing import (
    ReconstructionTiming,
    aggregate_reconstruction_timings,
    summarize_ms,
)


def benchmark_reconstruction_scenarios(
    pipeline_factories: dict[str, Callable[[], Any]],
    *,
    frame_indices: tuple[int, ...],
    warmup: int,
) -> dict[str, Any]:
    """Benchmark identical frame indices under each independently built scenario."""

    if set(pipeline_factories) != {
        "reconstruction_only",
        "reconstruction_crop",
        "reconstruction_crop_sampling",
    }:
        raise ValueError("benchmark requires the three canonical scenarios")
    if not frame_indices or warmup < 0:
        raise ValueError("benchmark frame indices must be non-empty and warmup non-negative")
    scenarios: dict[str, Any] = {}
    for name, factory in pipeline_factories.items():
        pipeline = factory()
        for index in frame_indices[:warmup]:
            pipeline.build(index)
        samples = []
        point_counts = []
        for index in frame_indices:
            result = pipeline.build(index)
            timing = result.timing_report_ms["reconstruction"]
            samples.append(ReconstructionTiming.from_dict(timing))
            point_counts.append(
                {
                    "concatenated": int(result.concatenated.points.shape[0]),
                    "cropped": int(result.workspace_cropped.points.shape[0]),
                    "fused": int(result.fused.points.shape[0]),
                    "sampled": int(result.sampled.points.shape[0]),
                }
            )
        scenarios[name] = {
            "timing": aggregate_reconstruction_timings(samples),
            "point_counts": point_counts,
        }
    return {
        "schema_version": "pointcloud-builder.world-reconstruction-benchmark.v1",
        "frame_indices": list(frame_indices),
        "warmup": warmup,
        "same_inputs": True,
        "scenarios": scenarios,
    }


def benchmark_timing_overhead(
    enabled_factory: Callable[[], Any],
    disabled_factory: Callable[[], Any],
    *,
    frame_indices: tuple[int, ...],
    warmup: int,
) -> dict[str, Any]:
    """Measure incremental unified-instrumentation overhead on identical inputs."""

    if not frame_indices or warmup < 0:
        raise ValueError("overhead frame indices must be non-empty")

    def prepare(factory: Callable[[], Any], enabled: bool) -> Any:
        pipeline = factory()
        pipeline.processor.timing_enabled = enabled
        for index in frame_indices[:warmup]:
            result = pipeline.build(index)
            _synchronize_result(result)
        return pipeline

    def measure(pipeline: Any, index: int) -> float:
        started = time.perf_counter()
        result = pipeline.build(index)
        _synchronize_result(result)
        return (time.perf_counter() - started) * 1000.0

    enabled_pipeline = prepare(enabled_factory, True)
    disabled_pipeline = prepare(disabled_factory, False)
    enabled_values = []
    disabled_values = []
    for position, index in enumerate(frame_indices):
        if position % 2 == 0:
            enabled_values.append(measure(enabled_pipeline, index))
            disabled_values.append(measure(disabled_pipeline, index))
        else:
            disabled_values.append(measure(disabled_pipeline, index))
            enabled_values.append(measure(enabled_pipeline, index))
    enabled_mean = statistics.mean(enabled_values)
    disabled_mean = statistics.mean(disabled_values)
    degradation_percent = (
        (enabled_mean / disabled_mean - 1.0) * 100.0 if disabled_mean else float("inf")
    )
    return {
        "timing_enabled_ms": summarize_ms(enabled_values),
        "timing_disabled_ms": summarize_ms(disabled_values),
        "throughput_degradation_percent": degradation_percent,
        "maximum_percent": 5.0,
        "execution_order": "alternating_enabled_disabled",
        "disabled_scope": "builder_single_camera_and_rig_timing_disabled",
        "passed": degradation_percent <= 5.0,
    }


def _synchronize_result(result: Any) -> None:
    points = result.sampled.points
    if getattr(points, "is_cuda", False):
        import torch

        torch.cuda.current_stream(points.device).synchronize()
