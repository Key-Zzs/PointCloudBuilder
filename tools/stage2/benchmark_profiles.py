#!/usr/bin/env python3
"""Benchmark Stage-2 geometry profiles without conflating SAM sidecar cost.

This tool is deliberately an offline, bounded reader.  It reads at most the
requested frames per episode from immutable LeRobot sidecars, reuses one
precomputed episode plane for the table-filtered timing, and reports SAM as a
separate cached-sidecar prerequisite rather than including model inference in
PCB timings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PCB_ROOT = Path(__file__).resolve().parents[2]
OUTER_ROOT = PCB_ROOT.parent
for candidate in (PCB_ROOT, OUTER_ROOT / "tools"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import export_lerobot_to_dp3_zarr as exporter  # noqa: E402
import lerobot_rgbd_source as rgbd_source  # noqa: E402
from pointcloud_builder import PointCloudBuilder  # noqa: E402
from pointcloud_builder.config import PipelineConfig, SupportPlaneConfig  # noqa: E402
from pointcloud_builder.instance import build_instance_dense, build_instance_sparse  # noqa: E402
from pointcloud_builder.segmentation import ManifestPromptProvider, decode_rle  # noqa: E402
from pointcloud_builder.segmentation.types import InstanceMask  # noqa: E402
from pointcloud_builder.support_plane import estimate_episode_support_plane  # noqa: E402


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean_ms": float("nan"), "median_ms": float("nan"), "p95_ms": float("nan"), "p99_ms": float("nan")}
    return {
        "count": int(len(array)), "mean_ms": float(array.mean()), "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)), "p99_ms": float(np.percentile(array, 99)),
    }


def _online_gate(legacy: Iterable[float], table: Iterable[float]) -> dict[str, Any]:
    legacy_array, table_array = np.asarray(list(legacy), dtype=np.float64), np.asarray(list(table), dtype=np.float64)
    count = min(len(legacy_array), len(table_array))
    if not count:
        return {"status": "NOT_RUN", "reason": "no matched legacy/table timings"}
    delta = table_array[:count] - legacy_array[:count]
    legacy_mean = float(legacy_array[:count].mean())
    degradation = float((table_array[:count].mean() / legacy_mean) - 1.0) if legacy_mean > 0.0 else float("inf")
    passed = float(np.percentile(delta, 95)) <= 3.0 + 1e-12 and degradation <= 0.10 + 1e-12
    return {
        "status": "ONLINE_ELIGIBLE" if passed else "OFFLINE_ONLY",
        "incremental": _summary(delta),
        "throughput_degradation_fraction": degradation,
        "thresholds": {"p95_incremental_ms_max": 3.0, "throughput_degradation_fraction_max": 0.10},
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.incomplete-", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _frame_from_source(source_frame: Any) -> dict[str, Any]:
    return exporter._builder_frame_from_source_frame(  # type: ignore[attr-defined]
        source_frame,
        camera="head",
        depth_source="ffs_stereo",
        timestamp_column=rgbd_source.CAMERA_SPECS["head"]["timestamp_column"],
        rgb=None,
    )


def _episode_ranges(ends: np.ndarray) -> list[tuple[int, int]]:
    start = 0
    result = []
    for end in ends.tolist():
        result.append((start, int(end)))
        start = int(end)
    return result


def _selected_indices(start: int, end: int, maximum: int) -> list[int]:
    if maximum <= 0:
        raise ValueError("--max-frames-per-episode must be positive")
    return sorted(set(int(value) for value in np.linspace(start, end - 1, min(maximum, end - start))))


def _records_for_selected_frames(sidecar_path: Path, frame_indices: set[int]) -> dict[int, list[InstanceMask]]:
    """Decode only benchmarked RLE masks, never a whole episode into RAM."""

    import zarr

    root = zarr.open_group(str(sidecar_path), mode="r")
    entries = json.loads(root.attrs["entries_json"])
    runs = root["rle_runs"]
    result: dict[int, list[InstanceMask]] = defaultdict(list)
    for entry in entries:
        frame_index = int(entry["frame_index"])
        if frame_index not in frame_indices:
            continue
        offset, length = int(entry["rle_offset"]), int(entry["rle_length"])
        mask = decode_rle(
            np.asarray(runs[offset : offset + length]),
            shape=tuple(entry["shape"]),
            starts_with_one=bool(entry["starts_with_one"]),
        )
        result[frame_index].append(InstanceMask(
            frame_index=frame_index,
            episode_index=int(entry["episode_index"]),
            track_id=str(entry["track_id"]),
            concept_id=str(entry["concept_id"]),
            binary_mask=mask,
            bbox=tuple(int(value) for value in entry["bbox"]),
            score=float(entry["score"]),
            prompt_type=str(entry["prompt_type"]),  # type: ignore[arg-type]
            prompt_value=str(entry["prompt_value"]),
            valid=bool(entry["valid"]),
        ))
    missing = sorted(frame_indices - set(result))
    if missing:
        raise ValueError(f"SAM sidecar lacks explicit records for benchmark frames: {missing[:8]}")
    return dict(result)


def _timing(meta: dict[str, Any]) -> float:
    timing = meta.get("ffs", {}).get("timing_ms", {})
    value = timing.get("total_builder_pipeline")
    if not isinstance(value, (float, int)):
        raise ValueError("FFS builder metadata lacks total_builder_pipeline timing")
    return float(value)


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.lerobot_path).resolve()
    builder_path = Path(args.builder_config).resolve()
    info = exporter._read_json(source_root / "meta" / "info.json")  # type: ignore[attr-defined]
    data_paths = exporter._data_parquet_paths(source_root)  # type: ignore[attr-defined]
    rows = exporter._count_parquet_rows(data_paths)  # type: ignore[attr-defined]
    source = rgbd_source.open_rgbd_sidecar_source(source_root, source="auto", info=info, parquet_row_count=rows)
    source.validate_join(data_paths, camera="head")
    if not hasattr(source, "root"):
        raise RuntimeError("Stage-2 benchmark requires the authoritative raw RGB-D Zarr sidecar")
    ends = np.asarray(source.root["meta/episode_ends"][:], dtype=np.int64)
    ranges = _episode_ranges(ends)
    episodes = tuple(int(value) for value in args.episodes)
    if not episodes or any(value < 0 or value >= len(ranges) for value in episodes):
        raise ValueError(f"--episodes must be within 0..{len(ranges) - 1}")
    selected_by_episode = {episode: _selected_indices(*ranges[episode], args.max_frames_per_episode) for episode in episodes}
    all_indices = {frame for values in selected_by_episode.values() for frame in values}
    columns = ["episode_index", "frame_index", "global_frame_index", rgbd_source.CAMERA_SPECS["head"]["timestamp_column"], rgbd_source.CAMERA_SPECS["head"]["reused_column"]]
    frames = {
        index: _frame_from_source(source.read_frame_at(data_paths, camera="head", row_index=index, columns=columns, include_ir=True))
        for index in sorted(all_indices)
    }
    base = PointCloudBuilder.from_yaml(builder_path)
    plane_config = replace(base.config, pipeline=PipelineConfig(profile="table_filtered"), support_plane=replace(base.config.support_plane, enabled=True))
    legacy_config = replace(base.config, pipeline=PipelineConfig(profile="legacy"), support_plane=SupportPlaneConfig(enabled=False))
    legacy, table = PointCloudBuilder(legacy_config), PointCloudBuilder(plane_config)
    timings: dict[str, list[float]] = defaultdict(list)
    planes: dict[int, dict[str, Any]] = {}
    plane_models: dict[int, Any] = {}
    for episode, indices in selected_by_episode.items():
        representative = []
        plane_start = time.perf_counter()
        for index in indices:
            stages, _ = table.build_unfiltered_perception_stages(frames[index])
            representative.append((index, stages["cropped"]))
        plane = estimate_episode_support_plane(
            representative,
            distance_threshold_m=base.config.support_plane.ransac_threshold_m,
            ransac_iterations=base.config.support_plane.ransac_iterations,
        )
        timings["plane_estimation_ms"].append((time.perf_counter() - plane_start) * 1000.0)
        table.set_support_plane(plane)
        planes[episode] = plane.to_dict()
        plane_models[episode] = plane
        for index in indices:
            _, legacy_meta = legacy.build_perception_stages(frames[index])
            _, table_meta = table.build_perception_stages(frames[index])
            timings["legacy_total_ms"].append(_timing(legacy_meta))
            timings["table_filtered_total_ms"].append(_timing(table_meta))
    report: dict[str, Any] = {
        "schema_version": "paper_a_stage2_profile_benchmark_v1",
        "source_lerobot_path": str(source_root),
        "builder_config": str(builder_path),
        "episodes": list(episodes),
        "frames_per_episode": {str(key): len(value) for key, value in selected_by_episode.items()},
        "planes": planes,
        "profiles": {
            "legacy": {"status": "PASS", "total": _summary(timings["legacy_total_ms"])},
            "table_filtered": {"status": "PASS", "total": _summary(timings["table_filtered_total_ms"]), "plane_estimation": _summary(timings["plane_estimation_ms"])},
            "instance_dense": {"status": "NOT_RUN_SIDECAR_REQUIRED"},
            "instance_sparse": {"status": "NOT_RUN_SIDECAR_REQUIRED"},
        },
        "table_filtered_online_gate_precomputed_plane": _online_gate(timings["legacy_total_ms"], timings["table_filtered_total_ms"]),
        "sam3_sidecar_timing": {"status": "NOT_RUN_SIDECAR_REQUIRED", "separate_from_pcb": True},
    }
    if args.mask_sidecar_root:
        sidecar_root = Path(args.mask_sidecar_root).resolve()
        manifest = json.loads((sidecar_root / "manifest.json").read_text(encoding="utf-8"))
        prompt_path = Path(args.prompt_config).resolve()
        prompts = ManifestPromptProvider.from_yaml(prompt_path)
        expected = {prompt.concept_id: prompt.expected_instances for prompt in prompts.prompts if prompt.expected_instances is not None}
        dense_builder, sparse_builder = PointCloudBuilder(plane_config), PointCloudBuilder(plane_config)
        for episode, indices in selected_by_episode.items():
            dense_builder.set_support_plane(plane_models[episode])
            sparse_builder.set_support_plane(plane_models[episode])
            entry = manifest.get("episodes", {}).get(str(episode))
            if not isinstance(entry, dict):
                raise FileNotFoundError(f"SAM sidecar missing benchmark episode {episode}")
            masks = _records_for_selected_frames(Path(entry["path"]), set(indices))
            for index in indices:
                dense_start = time.perf_counter()
                dense_stages, _ = dense_builder.build_perception_stages(frames[index])
                build_instance_dense(
                    raw_dense_points=dense_stages["raw"],
                    projection=dense_builder.project_points_to_color_image(dense_stages["raw"], source_frame="raw_dense"),
                    masks=masks[index], sampling_config=dense_builder.config.instance_sampling.as_sampling_config(),
                    support_plane=dense_builder.support_plane, expected_instances=expected,
                )
                timings["instance_dense_total_ms"].append((time.perf_counter() - dense_start) * 1000.0)
                sparse_start = time.perf_counter()
                sparse_stages, _ = sparse_builder.build_perception_stages(frames[index])
                build_instance_sparse(
                    workspace_sampled_points=sparse_stages["sampled"],
                    projection=sparse_builder.project_points_to_color_image(sparse_stages["sampled"], source_frame="workspace_sampled"),
                    masks=masks[index], sampling_config=sparse_builder.config.instance_sampling.as_sampling_config(), expected_instances=expected,
                )
                timings["instance_sparse_total_ms"].append((time.perf_counter() - sparse_start) * 1000.0)
        report["profiles"]["instance_dense"] = {"status": "PASS", "total": _summary(timings["instance_dense_total_ms"])}
        report["profiles"]["instance_sparse"] = {"status": "PASS", "total": _summary(timings["instance_sparse_total_ms"])}
        performance = [entry.get("performance") for entry in manifest.get("episodes", {}).values() if isinstance(entry, dict) and isinstance(entry.get("performance"), dict)]
        if performance:
            report["sam3_sidecar_timing"] = {
                "status": "RECORDED", "separate_from_pcb": True,
                "episode_ms_per_source_frame": _summary(float(item["episode_ms_per_source_frame"]) for item in performance),
                "episode_wall_seconds": _summary(float(item["episode_wall_seconds"]) for item in performance),
                "cuda_peak_memory_allocated_mib": _summary(float(item["cuda_peak_memory_allocated_mib"]) for item in performance),
            }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-path", required=True)
    parser.add_argument("--builder-config", required=True)
    parser.add_argument("--episodes", nargs="+", type=int, default=[0])
    parser.add_argument("--max-frames-per-episode", type=int, default=8)
    parser.add_argument("--mask-sidecar-root")
    parser.add_argument("--prompt-config", default=str(PCB_ROOT / "configs/paper_a_stage2_prompts_v01.yaml"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    output = Path(args.output).resolve()
    _atomic_json(output, report)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
