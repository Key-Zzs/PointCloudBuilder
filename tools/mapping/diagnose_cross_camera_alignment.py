#!/usr/bin/env python3
"""Capture and analyze diagnostic-only cross-camera alignment evidence.

This tool is deliberately outside the production reconstruction path.  It
requires FFS TensorRT-plugin FP16, RGB, host-clock matching, fusion OFF, and
sampling OFF.  Its fitted transform is serialized only as diagnostic evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from camera_rig.api import load_provisioned_camera_bundle

from pointcloud_builder.diagnostics.cross_camera_alignment import (
    apply_transform,
    binned_residuals,
    correlation_summary,
    distance_summary_mm,
    frozen_overlap_mask,
    normalized_image_coordinates,
    point_to_plane_observability,
    robust_point_to_point_icp,
)
from pointcloud_builder.fusion import board_surface_metrics, detect_cube
from pointcloud_builder.integrations.camera_rig.calibration_adapter import (
    resolve_bundle_transform,
    rigid_transform_to_frame_explicit,
)
from pointcloud_builder.mapping.depth_packet import provision_identity_sha256
from pointcloud_builder.mapping.provenance import rig_backend_provenance
from pointcloud_builder.mapping.recording import (
    RigDepthRecordingWriter,
    iter_rig_depth_recording,
    validate_rig_depth_recording,
)
from pointcloud_builder.projection import project_points
from pointcloud_builder.rig import build_live_rig, load_rig_config
from pointcloud_builder.rig_calibration.artifact import (
    load_solution,
    solution_fingerprint,
)
from pointcloud_builder.rig_calibration.diagnostics import (
    candidate_diagnostic_contract,
    candidate_T_workspace_from_geometry_source,
)
from pointcloud_builder.workspace import ExpectedPlaneRegion
from pointcloud_builder.workspace.types import WorkspacePointCloud

CAMERA_A = "camera_a"
CAMERA_B = "camera_b"
SCHEMA = "pointcloud-builder.cross-camera-alignment-diagnostic.v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--rig-config", required=True)
    capture.add_argument("--matched-sets", type=int, default=300)
    capture.add_argument("--output", required=True)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--mapping-config", required=True)
    analyze.add_argument("--analysis-output")
    analyze.add_argument("--candidate-solution")
    analyze.add_argument("--candidate-validation")
    analyze.add_argument(
        "--allow-missing-cube",
        action="store_true",
        help=(
            "explicit overlap-only mode: skip fixed-cube detection and keep "
            "cube acceptance NOT_RUN"
        ),
    )
    analyze.add_argument("--viewer-point-budget", type=int, default=150_000)
    run = subparsers.add_parser("run")
    run.add_argument("--rig-config", required=True)
    run.add_argument("--mapping-config", required=True)
    run.add_argument("--matched-sets", type=int, default=300)
    run.add_argument("--output", required=True)
    run.add_argument("--analysis-output")
    run.add_argument("--candidate-solution")
    run.add_argument("--candidate-validation")
    run.add_argument("--allow-missing-cube", action="store_true")
    run.add_argument("--viewer-point-budget", type=int, default=150_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    candidate_solution = getattr(args, "candidate_solution", None)
    candidate_validation = getattr(args, "candidate_validation", None)
    if bool(candidate_solution) != bool(candidate_validation):
        raise ValueError(
            "--candidate-solution and --candidate-validation must be provided together"
        )
    if candidate_solution and not getattr(args, "analysis_output", None):
        raise ValueError("candidate analysis requires a separate --analysis-output")
    if args.command in {"capture", "run"}:
        capture(Path(args.rig_config), args.matched_sets, _private_root(args.output))
    if args.command in {"analyze", "run"}:
        root = _private_root(args.input if args.command == "analyze" else args.output)
        analysis_output = getattr(args, "analysis_output", None)
        analyze(
            root,
            Path(args.mapping_config),
            args.viewer_point_budget,
            analysis_output=None
            if analysis_output is None
            else _private_root(analysis_output),
            candidate_solution_path=None
            if candidate_solution is None
            else _private_root(candidate_solution),
            candidate_validation_path=None
            if candidate_validation is None
            else _private_root(candidate_validation),
            allow_missing_cube=bool(getattr(args, "allow_missing_cube", False)),
        )


def capture(rig_config_path: Path, matched_sets: int, output: Path) -> None:
    if matched_sets < 5:
        raise ValueError("at least five matched sets are required")
    if output.exists():
        raise FileExistsError(f"diagnostic output already exists: {output}")
    config = load_rig_config(rig_config_path)
    _validate_diagnostic_config(config)
    output.mkdir(parents=True)
    (output / "selected_tensors").mkdir()
    (output / "raw_rgb").mkdir()
    (output / "screenshots").mkdir()
    (output / "rerun").mkdir()
    selected_indices = sorted(
        {round(index * (matched_sets - 1) / 4) for index in range(5)}
    )
    pipeline = build_live_rig(config, device="cuda")
    writer = RigDepthRecordingWriter(
        output / "depth_recording",
        depth_source="ffs_stereo",
        backend_provenance=rig_backend_provenance(config),
    )
    stage_records: dict[str, list[dict[str, Any]]] = {CAMERA_A: [], CAMERA_B: []}
    timing_records: list[dict[str, Any]] = []
    latencies = []
    started = time.perf_counter()
    try:
        pipeline.acquisition.start()
        for index in range(matched_sets):
            built = pipeline.capture_next()
            result = built.result
            writer.append(result.depth_frame_set)
            latencies.append(float(built.total_ms))
            for name, record in result.per_camera_stage_statistics.items():
                stage_records[name].append(dict(record))
            timing_records.append(_timing_record(result))
            _save_rgb_frames(result, output / "raw_rgb", index)
            if index in selected_indices:
                _save_selected_tensor(
                    result,
                    output / "selected_tensors",
                    index,
                    config,
                    pipeline.processor.runtimes,
                )
        pipeline.acquisition.stop()
        acquisition = pipeline.acquisition.report()
        duration = time.perf_counter() - started
        writer.finalize(
            report={
                "schema_version": f"{SCHEMA}.capture",
                "matched_sets": matched_sets,
                "duration_s": duration,
                "capture_fps": matched_sets / duration,
                "fusion_enabled": False,
                "sampling_enabled": False,
                "acquisition": acquisition,
            }
        )
    except BaseException:
        try:
            pipeline.acquisition.stop()
        finally:
            writer.abort()
        raise
    backend_values = sorted(
        {
            str(item["ffs_backend"])
            for records in stage_records.values()
            for item in records
            if item.get("ffs_backend") is not None
        }
    )
    capture_manifest = {
        "schema_version": f"{SCHEMA}.capture-manifest",
        "created_unix_s": time.time(),
        "matched_sets": matched_sets,
        "selected_indices": selected_indices,
        "rig_config": str(rig_config_path.resolve()),
        "rig_config_sha256": _sha256(rig_config_path),
        "depth_backend": backend_values,
        "precision": "fp16",
        "fusion": "OFF",
        "sampling": "OFF",
        "rgb_enabled": True,
        "viewer_used_for_quantitative_analysis": False,
        "latency_ms": _summary(latencies),
        "stage": {
            name: {
                "valid_depth_ratio": _summary(
                    [float(item["valid_depth_ratio"]) for item in records]
                ),
                "valid_disparity_ratio": _summary(
                    [
                        float(item["valid_disparity_ratio"])
                        for item in records
                        if item["valid_disparity_ratio"] is not None
                    ]
                ),
            }
            for name, records in stage_records.items()
        },
        "acquisition": acquisition,
        "privacy": "all device identities and real data remain under .local",
    }
    _write_json(output / "capture_manifest.json", capture_manifest)
    _write_json(
        output / "timing.json",
        {
            "schema_version": f"{SCHEMA}.timing",
            "records": timing_records,
            "note": "host receive times are host completion proxies, not exposure timestamps",
        },
    )


def analyze(
    root: Path,
    mapping_config_path: Path,
    viewer_point_budget: int,
    *,
    analysis_output: Path | None = None,
    candidate_solution_path: Path | None = None,
    candidate_validation_path: Path | None = None,
    allow_missing_cube: bool = False,
) -> None:
    if viewer_point_budget < 1:
        raise ValueError("viewer point budget must be positive")
    output = root if analysis_output is None else analysis_output
    if output != root and output.exists():
        raise FileExistsError(f"analysis output already exists: {output}")
    capture_manifest = _load_json(root / "capture_manifest.json")
    if _sha256(capture_manifest["rig_config"]) != capture_manifest["rig_config_sha256"]:
        raise ValueError("captured diagnostic rig config hash no longer matches")
    recording_manifest = validate_rig_depth_recording(root / "depth_recording")
    mapping = yaml.safe_load(mapping_config_path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or not isinstance(
        mapping.get("expected_plane"), dict
    ):
        raise TypeError("mapping config must contain expected_plane")
    board = _board(mapping["expected_plane"])
    selected = _load_selected(
        root / "selected_tensors", capture_manifest["selected_indices"]
    )
    candidate_contract = None
    transform_overrides: dict[str, np.ndarray] = {}
    if candidate_solution_path is not None:
        assert candidate_validation_path is not None
        transform_overrides, candidate_contract = _candidate_geometry_overrides(
            root,
            capture_manifest,
            recording_manifest,
            candidate_solution_path,
            candidate_validation_path,
        )
        selected = _reframe_selected_tensors(
            selected,
            transform_overrides,
            _recording_geometry_transforms(root, recording_manifest),
        )
    if allow_missing_cube:
        cube = None
        cube_status: dict[str, Any] = {
            "status": "NOT_RUN",
            "reason": "OVERLAP_ONLY_MODE_SKIPPED_CUBE_DETECTION",
            "formal_3d_acceptance": False,
        }
    else:
        cube = _detect_fixed_cube(selected, board)
        cube_status = {"status": "AVAILABLE"}

    fit_anchor, fit_moving = _aggregate_fit_points(selected)
    point_to_plane = _point_to_plane_icp(fit_anchor, fit_moving)
    reverse_point_to_plane = _point_to_plane_icp(fit_moving, fit_anchor)
    point_to_point = robust_point_to_point_icp(
        fit_anchor,
        fit_moving,
        maximum_correspondence_distance_m=0.030,
        trim_fraction=0.80,
    )
    correction = np.asarray(point_to_plane["T_anchor_from_moving_residual"])
    transform_report = _transform_report(
        correction, point_to_plane, point_to_point, reverse_point_to_plane
    )
    if output != root:
        output.mkdir(parents=True)
        (output / "screenshots").mkdir()
        (output / "rerun").mkdir()
    _write_json(output / "residual_transform.json", transform_report)

    frame_records: list[dict[str, Any]] = []
    image_before = _directional_accumulator(("u", "v", "r", "e", "p"))
    image_after = _directional_accumulator(("u", "v", "r", "e", "p"))
    depth_before = _directional_accumulator(("z", "e", "p", "edge"))
    depth_after = _directional_accumulator(("z", "e", "p", "edge"))
    roi_names = ("board", "background", "full_overlap")
    if cube is not None:
        roi_names = ("board", "cube", "cube_edge", "background", "full_overlap")
    roi_accumulator: dict[str, dict[str, list[np.ndarray]]] = {
        roi: {"before": [], "after": []}
        for roi in roi_names
    }
    window_transforms: list[np.ndarray] = []
    frames = list(iter_rig_depth_recording(root / "depth_recording"))
    window_edges = np.array_split(np.arange(len(frames)), 5)
    for window in window_edges:
        anchor_parts, moving_parts = [], []
        for index in window[:: max(1, len(window) // 3)]:
            geometries = {
                item.camera_name: _geometry(
                    item,
                    _workspace_crop(capture_manifest),
                    T_workspace_from_camera=transform_overrides.get(item.camera_name),
                )
                for item in frames[int(index)].observations
            }
            anchor_parts.append(_voxel_downsample(geometries[CAMERA_A]["xyz"], 0.006))
            moving_parts.append(_voxel_downsample(geometries[CAMERA_B]["xyz"], 0.006))
        window_transforms.append(
            np.asarray(
                _point_to_plane_icp(
                    np.concatenate(anchor_parts), np.concatenate(moving_parts)
                )["T_anchor_from_moving_residual"]
            )
        )

    for frame in frames:
        geometries = {
            item.camera_name: _geometry(
                item,
                _workspace_crop(capture_manifest),
                T_workspace_from_camera=transform_overrides.get(item.camera_name),
            )
            for item in frame.observations
        }
        a, b = geometries[CAMERA_A], geometries[CAMERA_B]
        before = _correspondences(a, b, np.eye(4))
        after = _correspondences(a, b, correction)
        frozen_overlap = {
            CAMERA_A: frozen_overlap_mask(before["a_to_b_distance_m"]),
            CAMERA_B: frozen_overlap_mask(before["b_to_a_distance_m"]),
        }
        frame_record = {
            "matched_set_index": int(frame.matched_set_index),
            "maximum_skew_ms": float(frame.maximum_skew_ms),
            "actual_point_count": {
                CAMERA_A: int(a["xyz"].shape[0]),
                CAMERA_B: int(b["xyz"].shape[0]),
            },
            "before": _frame_metric_projection(
                before, a, b, board, cube, frozen_overlap
            ),
            "after": _frame_metric_projection(after, a, b, board, cube, frozen_overlap),
            "single_camera": {
                CAMERA_A: _single_camera_metrics(a, board, cube),
                CAMERA_B: _single_camera_metrics(b, board, cube),
            },
        }
        frame_records.append(frame_record)
        if int(frame.matched_set_index) in set(capture_manifest["selected_indices"]):
            for name, geometry, direction in (
                (CAMERA_A, a, "a_to_b"),
                (CAMERA_B, b, "b_to_a"),
            ):
                coords = normalized_image_coordinates(
                    geometry["uv"],
                    width=geometry["intrinsics"].width,
                    height=geometry["intrinsics"].height,
                    fx=geometry["intrinsics"].fx,
                    fy=geometry["intrinsics"].fy,
                    cx=geometry["intrinsics"].cx,
                    cy=geometry["intrinsics"].cy,
                )
                overlap = frozen_overlap[name]
                for target, corr in ((image_before, before), (image_after, after)):
                    target[name]["u"].append(coords["u_over_w"][overlap])
                    target[name]["v"].append(coords["v_over_h"][overlap])
                    target[name]["r"].append(coords["radius"][overlap])
                    target[name]["e"].append(corr[f"{direction}_distance_m"][overlap])
                    target[name]["p"].append(
                        corr[f"{direction}_signed_plane_m"][overlap]
                    )
                for target, corr in ((depth_before, before), (depth_after, after)):
                    target[name]["z"].append(geometry["depth_m"][overlap])
                    target[name]["e"].append(corr[f"{direction}_distance_m"][overlap])
                    target[name]["p"].append(
                        corr[f"{direction}_signed_plane_m"][overlap]
                    )
                    target[name]["edge"].append(geometry["edge"][overlap])
        roi_masks = _roi_masks(
            b["xyz"], b["edge"], board, cube, frozen_overlap[CAMERA_B]
        )
        for name, mask in roi_masks.items():
            if not mask.any():
                continue
            roi_accumulator[name]["before"].append(
                distance_summary_mm(before["b_to_a_distance_m"][mask])
            )
            roi_accumulator[name]["after"].append(
                distance_summary_mm(after["b_to_a_distance_m"][mask])
            )

    raw_metrics = _aggregate_raw_metrics(
        frame_records, roi_accumulator, window_transforms
    )
    image_analysis = _image_analysis(image_before, image_after)
    depth_analysis = _depth_analysis(depth_before, depth_after)
    timing = _load_json(root / "timing.json")
    skew_analysis = _skew_analysis(frame_records, timing["records"])
    rgb_analysis = _rgb_analysis(selected, correction)
    _write_json(output / "raw_metrics.json", raw_metrics)
    _write_json(output / "image_position_analysis.json", image_analysis)
    _write_json(output / "depth_analysis.json", depth_analysis)
    _write_json(output / "skew_analysis.json", skew_analysis)
    _write_json(output / "rgb_analysis.json", rgb_analysis)
    _render_all(output, selected, correction, board, cube, viewer_point_budget)
    _write_rerun(output, selected, correction, viewer_point_budget)
    manifest = {
        "schema_version": SCHEMA,
        "analysis_calibration": "candidate" if candidate_contract else "production",
        "analysis_scope": "overlap_only" if allow_missing_cube else "board_and_cube",
        "capture_root": str(root.resolve()),
        "candidate_contract": candidate_contract,
        "capture_manifest_sha256": _sha256(root / "capture_manifest.json"),
        "recording_manifest": {
            "schema_version": recording_manifest["schema_version"],
            "matched_set_count": recording_manifest["matched_set_count"],
            "camera_names": recording_manifest["camera_names"],
            "depth_source": recording_manifest["depth_source"],
        },
        "mapping_config": str(mapping_config_path.resolve()),
        "mapping_config_sha256": _sha256(mapping_config_path),
        "scene_rois": {
            "board": asdict(board),
            "cube": cube,
            "cube_status": cube_status,
            "background": "full overlap excluding fixed board and cube ROIs",
        },
        "quantitative_input": "all full per-camera reconstruction-equivalent tensors reconstructed from same-pass depth observations",
        "viewer_point_budget": viewer_point_budget,
        "actual_selected_point_count": {
            str(index): {
                name: int(value["points"].shape[0]) for name, value in cameras.items()
            }
            for index, cameras in selected.items()
        },
        "fitted_transform_applied_to_production": False,
        "production_calibration_modified": False,
    }
    _write_json(output / "manifest.json", manifest)


def _candidate_geometry_overrides(
    root: Path,
    capture_manifest: dict[str, Any],
    recording_manifest: dict[str, Any],
    solution_path: Path,
    validation_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    solution = load_solution(solution_path)
    validation = _load_json(validation_path)
    fingerprint = solution_fingerprint(solution)
    if not solution.passed:
        raise ValueError("candidate diagnostic requires a passed calibration solution")
    if (
        validation.get("passed") is not True
        or validation.get("status") != "PASS"
        or validation.get("solution_fingerprint") != fingerprint
    ):
        raise ValueError(
            "candidate validation must pass and bind to the exact solution fingerprint"
        )
    holdout = validation.get("holdout")
    if not isinstance(holdout, dict) or holdout.get("status") != "PASS":
        raise ValueError("candidate diagnostic requires passed multicamera holdout")

    camera_names = tuple(recording_manifest["camera_names"])
    if set(solution.T_workspace_from_camera) != set(camera_names):
        raise ValueError("candidate camera set differs from diagnostic recording")
    if solution.workspace_frame != recording_manifest["workspace_frame"]:
        raise ValueError("candidate workspace frame differs from diagnostic recording")

    rig = load_rig_config(capture_manifest["rig_config"])
    rig_cameras = {camera.name: camera for camera in rig.enabled_cameras}
    if set(rig_cameras) != set(camera_names):
        raise ValueError("capture rig camera set differs from diagnostic recording")

    overrides: dict[str, np.ndarray] = {}
    geometry_contract: dict[str, Any] = {}
    for camera_name in camera_names:
        source = rig_cameras[camera_name].source
        bundle_path = Path(source.provision_artifact)
        bundle = load_provisioned_camera_bundle(bundle_path)
        calibration = _load_json(
            root / "depth_recording" / "calibration" / f"{camera_name}.json"
        )
        if calibration.get("workspace_frame") != solution.workspace_frame:
            raise ValueError(f"{camera_name}: recording workspace frame mismatch")
        if calibration.get("bundle_identity") != str(bundle.bundle_id):
            raise ValueError(f"{camera_name}: recording CameraBundle identity mismatch")
        if calibration.get("provision_sha256") != provision_identity_sha256(bundle_path):
            raise ValueError(f"{camera_name}: recording provision identity mismatch")
        if bundle.device.to_dict() != solution.camera_identities[camera_name]:
            raise ValueError(f"{camera_name}: candidate camera identity mismatch")
        if _camera_bundle_sha256(bundle_path) != solution.camera_bundle_hashes[camera_name]:
            raise ValueError(f"{camera_name}: candidate CameraBundle hash mismatch")
        geometry_source_frame = str(calibration.get("source_frame", ""))
        internal = rigid_transform_to_frame_explicit(
            resolve_bundle_transform(
                bundle,
                geometry_source_frame,
                solution.camera_frames[camera_name],
            )
        )
        candidate_transform = candidate_T_workspace_from_geometry_source(
            solution,
            camera_name,
            geometry_source_frame=geometry_source_frame,
            internal_transform=internal,
        )
        overrides[camera_name] = candidate_transform
        geometry_contract[camera_name] = {
            "source_frame": geometry_source_frame,
            "target_frame": solution.workspace_frame,
            "T_workspace_from_geometry_source": candidate_transform.tolist(),
        }

    contract = dict(candidate_diagnostic_contract(solution))
    contract.update(
        {
            "solution_path": str(solution_path.resolve()),
            "solution_sha256": _sha256(solution_path),
            "validation_path": str(validation_path.resolve()),
            "validation_sha256": _sha256(validation_path),
            "holdout": holdout,
            "geometry_source_overrides": geometry_contract,
        }
    )
    return overrides, contract


def _recording_geometry_transforms(
    root: Path, recording_manifest: dict[str, Any]
) -> dict[str, np.ndarray]:
    return {
        camera_name: np.asarray(
            _load_json(
                root / "depth_recording" / "calibration" / f"{camera_name}.json"
            )["T_workspace_from_camera"],
            dtype=np.float64,
        )
        for camera_name in recording_manifest["camera_names"]
    }


def _reframe_selected_tensors(
    selected: dict[int, dict[str, dict[str, np.ndarray]]],
    candidate_transforms: dict[str, np.ndarray],
    production_transforms: dict[str, np.ndarray],
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    output: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for index, cameras in selected.items():
        output[index] = {}
        for camera_name, values in cameras.items():
            reframed = dict(values)
            points = np.asarray(values["points"]).copy()
            delta = candidate_transforms[camera_name] @ np.linalg.inv(
                production_transforms[camera_name]
            )
            points[:, :3] = apply_transform(points[:, :3], delta)
            reframed["points"] = points
            output[index][camera_name] = reframed
    return output


def _camera_bundle_sha256(path: str | Path) -> str:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "camera_bundle.json"
    return _sha256(candidate)


def _validate_diagnostic_config(config: Any) -> None:
    names = [camera.name for camera in config.enabled_cameras]
    if names != [CAMERA_A, CAMERA_B]:
        raise ValueError("diagnostic requires canonical camera_a/camera_b")
    if config.fusion.enabled or config.sampling.enabled:
        raise ValueError("diagnostic requires fusion OFF and sampling OFF")
    if config.timing.mode != "nearest_host_timestamp":
        raise ValueError("diagnostic requires unchanged host-timestamp matcher")
    if any(camera.depth.mode != "ffs_stereo" for camera in config.enabled_cameras):
        raise ValueError("diagnostic requires current FFS depth")
    if any(not camera.pointcloud.use_rgb for camera in config.enabled_cameras):
        raise ValueError("diagnostic requires RGB from both cameras")
    for camera in config.enabled_cameras:
        pipeline = yaml.safe_load(
            Path(camera.pipeline_config).read_text(encoding="utf-8")
        )
        ffs = pipeline["depth_source"]["ffs"]
        if ffs.get("backend") != "tensorrt_plugin" or ffs.get("precision") != "fp16":
            raise ValueError("diagnostic requires current TensorRT plugin FP16")


def _save_rgb_frames(result: Any, output: Path, index: int) -> None:
    from PIL import Image

    for name, envelope in result.frame_match.envelopes.items():
        stream = envelope.frame.streams.get("color")
        if stream is None:
            raise ValueError(f"{name} has no RGB stream")
        Image.fromarray(stream.data, mode="RGB").save(
            output / f"set_{index:06d}_{name}.png", compress_level=3
        )


def _timing_record(result: Any) -> dict[str, Any]:
    frame_match = result.frame_match
    cameras = {}
    for name, envelope in frame_match.envelopes.items():
        cameras[name] = {
            "frame_index": int(envelope.frame_index),
            "host_receive_timestamp_ns": int(envelope.host_receive_timestamp_ns),
            "signed_delta_ms": float(frame_match.per_camera_delta_ms[name]),
            "absolute_delta_ms": float(frame_match.per_camera_absolute_delta_ms[name]),
            "streams": {
                stream_name: {
                    "frame_number": int(stream.frame_number),
                    "sensor_timestamp_ns": stream.sensor_timestamp_ns,
                    "timestamp_domain": stream.timestamp_domain,
                }
                for stream_name, stream in envelope.frame.streams.items()
            },
        }
    return {
        "matched_set_index": int(frame_match.match_sequence_index),
        "match_timestamp_ns": int(frame_match.match_timestamp_ns),
        "maximum_skew_ms": float(frame_match.maximum_skew_ms),
        "matching_policy": frame_match.matching_policy,
        "cameras": cameras,
    }


def _save_selected_tensor(
    result: Any,
    output: Path,
    index: int,
    config: Any,
    runtimes: dict[str, Any],
) -> None:
    observations = {
        item.camera_name: item for item in result.depth_frame_set.observations
    }
    for item in result.per_camera_workspace:
        points = item.cloud.points.detach().cpu().numpy().astype(np.float32, copy=False)
        geometry = _geometry(observations[item.camera_name], config.workspace_crop)
        if points.shape[0] != geometry["xyz"].shape[0] or not np.allclose(
            points[:, :3], geometry["xyz"], atol=2e-5, rtol=0
        ):
            raise RuntimeError(
                "source-pixel provenance does not match reconstruction tensor"
            )
        runtime = runtimes[item.camera_name]
        camera_model = runtime.pipeline.context.builder.camera
        color_uv, color_valid = _color_projection_uv(
            geometry["camera_xyz"], camera_model
        )
        bundle_color = runtime.pipeline.context.calibration.bundle.intrinsics["color"]
        np.savez_compressed(
            output / f"set_{index:06d}_{item.camera_name}.npz",
            points=points,
            uv=geometry["uv"].astype(np.int16),
            depth_m=geometry["depth_m"].astype(np.float32),
            edge=geometry["edge"],
            color_uv=color_uv.astype(np.float32),
            color_projection_valid=color_valid,
            color_intrinsics=np.asarray(
                (
                    camera_model.color_intrinsics.width,
                    camera_model.color_intrinsics.height,
                    camera_model.color_intrinsics.fx,
                    camera_model.color_intrinsics.fy,
                    camera_model.color_intrinsics.cx,
                    camera_model.color_intrinsics.cy,
                ),
                dtype=np.float64,
            ),
            color_distortion_coeffs=np.asarray(
                bundle_color.distortion_coeffs, dtype=np.float64
            ),
        )


def _color_projection_uv(
    points_depth: np.ndarray, camera_model: Any
) -> tuple[np.ndarray, np.ndarray]:
    extrinsics = camera_model.depth_to_color_extrinsics
    if extrinsics is None:
        raise ValueError("RGB diagnostic requires depth-to-color extrinsics")
    rotation = np.asarray(extrinsics.rotation, dtype=np.float64)
    translation = np.asarray(extrinsics.translation, dtype=np.float64)
    color = points_depth @ rotation.T + translation
    intrinsics = camera_model.color_intrinsics
    projection = project_points(torch.from_numpy(color), intrinsics)
    uv = projection.pixels_px.detach().cpu().numpy()
    projection_valid = projection.valid.detach().cpu().numpy()
    valid = (
        projection_valid
        & (np.rint(uv[:, 0]) >= 0)
        & (np.rint(uv[:, 0]) < intrinsics.width)
        & (np.rint(uv[:, 1]) >= 0)
        & (np.rint(uv[:, 1]) < intrinsics.height)
    )
    return uv, valid


def _geometry(
    observation: Any,
    crop: Any,
    *,
    T_workspace_from_camera: np.ndarray | None = None,
) -> dict[str, Any]:
    if observation.depth_source == "ffs_stereo" and (
        not observation.rectified
        or any(abs(value) > 1e-12 for value in observation.distortion_coeffs)
    ):
        raise ValueError(
            "diagnostic pinhole deprojection requires rectified zero-distortion FFS input"
        )
    depth = observation.metric_depth.astype(np.float64)
    intrinsics = observation.intrinsics
    rows, cols = np.indices(depth.shape)
    valid = depth > 0
    z = depth
    x = (cols - intrinsics.cx) * z / intrinsics.fx
    y = (rows - intrinsics.cy) * z / intrinsics.fy
    camera = np.stack((x, y, z), axis=-1)
    transform = (
        observation.T_workspace_from_camera
        if T_workspace_from_camera is None
        else np.asarray(T_workspace_from_camera, dtype=np.float64)
    )
    workspace = camera @ transform[:3, :3].T
    workspace += transform[:3, 3]
    normal_camera = _organized_normals(camera, valid)
    normal_workspace = normal_camera @ transform[:3, :3].T
    edge = _depth_edges(depth, valid)
    keep = valid & _crop_mask(workspace, crop)
    return {
        "xyz": np.ascontiguousarray(workspace[keep], dtype=np.float64),
        "camera_xyz": np.ascontiguousarray(camera[keep], dtype=np.float64),
        "uv": np.column_stack((cols[keep], rows[keep])).astype(np.float64),
        "depth_m": depth[keep],
        "edge": edge[keep],
        "normals": normal_workspace[keep],
        "intrinsics": intrinsics,
    }


def _organized_normals(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    dx = np.zeros_like(points)
    dy = np.zeros_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dy[1:-1] = points[2:] - points[:-2]
    normals = np.cross(dx, dy)
    norm = np.linalg.norm(normals, axis=-1)
    good = valid & np.isfinite(norm) & (norm > 1e-9)
    normals[good] /= norm[good, None]
    normals[~good] = np.nan
    return normals


def _depth_edges(
    depth: np.ndarray, valid: np.ndarray, threshold_m: float = 0.020
) -> np.ndarray:
    gradient = np.zeros(depth.shape, dtype=np.float64)
    gradient[:, 1:] = np.maximum(gradient[:, 1:], np.abs(depth[:, 1:] - depth[:, :-1]))
    gradient[:, :-1] = np.maximum(
        gradient[:, :-1], np.abs(depth[:, 1:] - depth[:, :-1])
    )
    gradient[1:] = np.maximum(gradient[1:], np.abs(depth[1:] - depth[:-1]))
    gradient[:-1] = np.maximum(gradient[:-1], np.abs(depth[1:] - depth[:-1]))
    neighbor_invalid = np.zeros_like(valid)
    neighbor_invalid[:, 1:] |= ~valid[:, :-1]
    neighbor_invalid[:, :-1] |= ~valid[:, 1:]
    neighbor_invalid[1:] |= ~valid[:-1]
    neighbor_invalid[:-1] |= ~valid[1:]
    return valid & ((gradient >= threshold_m) | neighbor_invalid)


def _crop_mask(points: np.ndarray, crop: Any) -> np.ndarray:
    if not crop.enabled:
        return np.ones(points.shape[:2], dtype=bool)
    return (
        (points[..., 0] >= crop.x[0])
        & (points[..., 0] <= crop.x[1])
        & (points[..., 1] >= crop.y[0])
        & (points[..., 1] <= crop.y[1])
        & (points[..., 2] >= crop.z[0])
        & (points[..., 2] <= crop.z[1])
    )


def _workspace_crop(capture_manifest: dict[str, Any]) -> Any:
    config = load_rig_config(capture_manifest["rig_config"])
    return config.workspace_crop


def _load_selected(
    directory: Path, indices: Iterable[int]
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    output = {}
    for index in indices:
        output[int(index)] = {}
        for name in (CAMERA_A, CAMERA_B):
            with np.load(directory / f"set_{int(index):06d}_{name}.npz") as data:
                output[int(index)][name] = {key: data[key] for key in data.files}
    return output


def _detect_fixed_cube(
    selected: dict[int, Any], board: ExpectedPlaneRegion
) -> dict[str, Any]:
    middle = selected[sorted(selected)[len(selected) // 2]]
    combined = np.concatenate((middle[CAMERA_A]["points"], middle[CAMERA_B]["points"]))
    cloud = WorkspacePointCloud(torch.from_numpy(combined), frame="workspace")
    board_metrics = board_surface_metrics(cloud, board)
    cube = detect_cube(
        cloud.points,
        board_p95_abs_z_m=float(board_metrics["p95_abs_z_m"]),
    ).to_dict()
    if cube["ambiguous"]:
        raise RuntimeError("fresh fixed cube ROI is ambiguous")
    cube["roi_source"] = "fresh middle selected tensor; frozen for all frames"
    return cube


def _aggregate_fit_points(selected: dict[int, Any]) -> tuple[np.ndarray, np.ndarray]:
    anchors, moving = [], []
    for cameras in selected.values():
        anchors.append(_voxel_downsample(cameras[CAMERA_A]["points"][:, :3], 0.004))
        moving.append(_voxel_downsample(cameras[CAMERA_B]["points"][:, :3], 0.004))
    return np.concatenate(anchors), np.concatenate(moving)


def _voxel_downsample(points: np.ndarray, voxel_m: float) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float64)
    keys = np.floor(xyz / voxel_m).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return xyz[np.sort(first)]


def _point_to_plane_icp(anchor: np.ndarray, moving: np.ndarray) -> dict[str, Any]:
    import open3d as o3d
    from scipy.spatial import cKDTree

    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(anchor))
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(moving))
    target = target.voxel_down_sample(0.004)
    source = source.voxel_down_sample(0.004)
    normal_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.020, max_nn=40)
    target.estimate_normals(normal_param)
    source.estimate_normals(normal_param)
    target_points = np.asarray(target.points)
    source_points = np.asarray(source.points)
    target_normals = np.asarray(target.normals)
    source_normals = np.asarray(source.normals)
    source_distance, source_index = cKDTree(target_points).query(
        source_points, workers=1
    )
    _, target_index = cKDTree(source_points).query(target_points, workers=1)
    source_rows = np.arange(source_points.shape[0])
    mutual = target_index[source_index] == source_rows
    normal_consistent = np.abs(
        np.einsum("ij,ij->i", source_normals, target_normals[source_index])
    ) >= math.cos(math.radians(30.0))
    accepted = mutual & normal_consistent & (source_distance <= 0.030)
    accepted_rows = np.flatnonzero(accepted)
    if accepted_rows.size < 100:
        raise RuntimeError(
            "point-to-plane ICP has too few mutual normal-consistent correspondences"
        )
    trim_count = max(100, math.floor(accepted_rows.size * 0.80))
    chosen = accepted_rows[
        np.argsort(source_distance[accepted_rows], kind="stable")[:trim_count]
    ]
    fit_source = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(source_points[chosen])
    )
    fit_source.normals = o3d.utility.Vector3dVector(source_normals[chosen])
    target_rows = source_index[chosen]
    fit_target = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(target_points[target_rows])
    )
    fit_target.normals = o3d.utility.Vector3dVector(target_normals[target_rows])
    loss = o3d.pipelines.registration.TukeyLoss(k=0.010)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=80
    )
    result = o3d.pipelines.registration.registration_icp(
        fit_source, fit_target, 0.030, np.eye(4), estimation, criteria
    )
    if result.fitness <= 0 or not np.isfinite(result.transformation).all():
        raise RuntimeError("robust point-to-plane ICP did not produce a valid fit")
    corrected = apply_transform(
        source_points[chosen], np.asarray(result.transformation)
    )
    final_distance, final_index = cKDTree(target_points[target_rows]).query(
        corrected, workers=1
    )
    final_valid = final_distance <= 0.030
    observability = point_to_plane_observability(
        corrected[final_valid], target_normals[target_rows][final_index[final_valid]]
    )
    return {
        "T_anchor_from_moving_residual": np.asarray(result.transformation),
        "fitness": float(result.fitness),
        "inlier_rmse_m": float(result.inlier_rmse),
        "anchor_voxels": len(target.points),
        "moving_voxels": len(source.points),
        "mutual_correspondence_count": int(mutual.sum()),
        "normal_consistent_count": int((mutual & normal_consistent).sum()),
        "pretrim_inlier_count": int(accepted_rows.size),
        "trimmed_fit_count": int(chosen.size),
        "trim_fraction": 0.80,
        "normal_consistency_max_angle_deg": 30.0,
        "observability": observability,
        "maximum_correspondence_distance_m": 0.030,
        "normal_radius_m": 0.020,
        "normal_max_nn": 40,
        "robust_loss": "Tukey",
        "robust_loss_k_m": 0.010,
        "initial_transform": "identity",
        "fit_voxel_m": 0.004,
    }


def _correspondences(
    a: dict[str, Any], b: dict[str, Any], correction: np.ndarray
) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    corrected_b = apply_transform(b["xyz"], correction)
    a_tree = cKDTree(a["xyz"])
    b_tree = cKDTree(corrected_b)
    b_distance, b_index = a_tree.query(corrected_b, workers=1)
    a_distance, a_index = b_tree.query(a["xyz"], workers=1)
    normals = a["normals"][b_index]
    delta = corrected_b - a["xyz"][b_index]
    signed = np.einsum("ij,ij->i", delta, normals)
    corrected_b_normals = b["normals"] @ correction[:3, :3].T
    reverse_normals = corrected_b_normals[a_index]
    reverse_delta = a["xyz"] - corrected_b[a_index]
    reverse_signed = np.einsum("ij,ij->i", reverse_delta, reverse_normals)
    return {
        "corrected_b": corrected_b,
        "b_to_a_index": b_index,
        "b_to_a_distance_m": b_distance,
        "b_to_a_signed_plane_m": signed,
        "a_to_b_index": a_index,
        "a_to_b_distance_m": a_distance,
        "a_to_b_signed_plane_m": reverse_signed,
    }


def _directional_accumulator(
    keys: tuple[str, ...],
) -> dict[str, dict[str, list[np.ndarray]]]:
    return {name: {key: [] for key in keys} for name in (CAMERA_A, CAMERA_B)}


def _frame_metric_projection(
    corr: dict[str, Any],
    a: dict[str, Any],
    b: dict[str, Any],
    board: Any,
    cube: Any,
    overlap: dict[str, np.ndarray],
) -> dict[str, Any]:
    masks = _roi_masks(b["xyz"], b["edge"], board, cube, overlap[CAMERA_B])
    return {
        "overlap_definition": "frozen before-correction nearest-neighbor distance <= 30 mm",
        "overlap_count": {
            CAMERA_A: int(overlap[CAMERA_A].sum()),
            CAMERA_B: int(overlap[CAMERA_B].sum()),
        },
        "a_to_b": distance_summary_mm(corr["a_to_b_distance_m"][overlap[CAMERA_A]]),
        "b_to_a": distance_summary_mm(corr["b_to_a_distance_m"][overlap[CAMERA_B]]),
        "symmetric": distance_summary_mm(
            np.concatenate(
                (
                    corr["a_to_b_distance_m"][overlap[CAMERA_A]],
                    corr["b_to_a_distance_m"][overlap[CAMERA_B]],
                )
            )
        ),
        "point_to_plane": _signed_summary(
            corr["b_to_a_signed_plane_m"][overlap[CAMERA_B]]
        ),
        "roi": {
            name: distance_summary_mm(corr["b_to_a_distance_m"][mask])
            for name, mask in masks.items()
            if int(mask.sum()) > 0
        },
        "board_plane": _cross_plane_metrics(a, b, board, corr["corrected_b"]),
    }


def _single_camera_metrics(
    geometry: dict[str, Any], board: Any, cube: Any
) -> dict[str, Any]:
    board_mask = _board_mask(geometry["xyz"], board)
    plane = _fit_plane(geometry["xyz"][board_mask])
    cube_mask = _cube_mask(geometry["xyz"], cube)
    return {
        "board": plane,
        "cube": (
            {
                "status": "NOT_RUN",
                "reason": "NO_CUBE_LIKE_CONNECTED_COMPONENT",
            }
            if cube is None
            else {
                "status": "AVAILABLE",
                "point_count": int(cube_mask.sum()),
                "extent_m": _extent(geometry["xyz"][cube_mask]),
                "edge_ratio": float(geometry["edge"][cube_mask].mean())
                if cube_mask.any()
                else None,
            }
        ),
        "valid_point_count": int(geometry["xyz"].shape[0]),
        "edge_ratio": float(geometry["edge"].mean()),
    }


def _cross_plane_metrics(
    a: dict[str, Any], b: dict[str, Any], board: Any, corrected_b: np.ndarray
) -> dict[str, Any]:
    pa = _fit_plane(a["xyz"][_board_mask(a["xyz"], board)])
    pb_points = corrected_b[_board_mask(b["xyz"], board)]
    pb = _fit_plane(pb_points)
    normal_a = np.asarray(pa["normal"])
    normal_b = np.asarray(pb["normal"])
    if np.dot(normal_a, normal_b) < 0:
        normal_b *= -1
    angle = math.degrees(math.acos(float(np.clip(np.dot(normal_a, normal_b), -1, 1))))
    anchor_center = -float(pa["offset_m"]) * normal_a
    signed_b_to_anchor = (pb_points - anchor_center) @ normal_a
    return {
        "normal_angular_difference_deg": angle,
        "signed_plane_offset_mm": float(np.median(signed_b_to_anchor) * 1000),
        "plane_thickness_a_mm": pa["thickness_mm"],
        "plane_thickness_b_mm": pb["thickness_mm"],
        "combined_double_layer_thickness_mm": _fit_plane(
            np.concatenate((a["xyz"][_board_mask(a["xyz"], board)], pb_points))
        )["thickness_mm"],
    }


def _fit_plane(points: np.ndarray) -> dict[str, Any]:
    if points.shape[0] < 10:
        raise ValueError("plane fit requires at least ten points")
    center = np.median(points, axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal *= -1
    signed = (points - center) @ normal
    return {
        "point_count": int(points.shape[0]),
        "normal": normal.tolist(),
        "offset_m": float(-np.dot(normal, center)),
        "thickness_mm": float(
            (np.quantile(signed, 0.95) - np.quantile(signed, 0.05)) * 1000
        ),
        "rmse_mm": float(np.sqrt(np.mean(signed * signed)) * 1000),
    }


def _roi_masks(
    points: np.ndarray,
    edge: np.ndarray,
    board: Any,
    cube: Any,
    overlap: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    board_mask = _board_mask(points, board)
    cube_mask = _cube_mask(points, cube)
    overlap_mask = np.ones(points.shape[0], dtype=bool) if overlap is None else overlap
    masks = {
        "board": overlap_mask & board_mask,
        "background": overlap_mask & ~(board_mask | cube_mask),
        "full_overlap": overlap_mask,
    }
    if cube is not None:
        masks.update(
            {
                "cube": overlap_mask & cube_mask,
                "cube_edge": overlap_mask & cube_mask & edge,
            }
        )
    return masks


def _board_mask(points: np.ndarray, board: Any) -> np.ndarray:
    return (
        (points[:, 0] >= board.x[0])
        & (points[:, 0] <= board.x[1])
        & (points[:, 1] >= board.y[0])
        & (points[:, 1] <= board.y[1])
        & (points[:, 2] >= board.z_search_range_m[0])
        & (points[:, 2] <= board.z_search_range_m[1])
    )


def _cube_mask(
    points: np.ndarray, cube: dict[str, Any] | None, padding_m: float = 0.015
) -> np.ndarray:
    if cube is None:
        return np.zeros(points.shape[0], dtype=bool)
    center = np.asarray(cube["center_workspace_m"])
    length, width, height = cube["dimensions_m"]
    yaw = math.radians(float(cube["yaw_deg"]))
    axes = np.array(((math.cos(yaw), math.sin(yaw)), (-math.sin(yaw), math.cos(yaw))))
    local = (points[:, :2] - center[:2]) @ axes.T
    return (
        (np.abs(local[:, 0]) <= length / 2 + padding_m)
        & (np.abs(local[:, 1]) <= width / 2 + padding_m)
        & (points[:, 2] >= center[2] - height / 2 - padding_m)
        & (points[:, 2] <= center[2] + height / 2 + padding_m)
    )


def _extent(points: np.ndarray) -> list[float] | None:
    if points.shape[0] < 10:
        return None
    return (
        np.quantile(points, 0.99, axis=0) - np.quantile(points, 0.01, axis=0)
    ).tolist()


def _aggregate_raw_metrics(
    records: list[dict[str, Any]], rois: Any, windows: list[np.ndarray]
) -> dict[str, Any]:
    def aggregate(stage: str, path: tuple[str, ...]) -> dict[str, float | int]:
        values = []
        for record in records:
            value: Any = record[stage]
            for key in path:
                value = value[key]
            values.append(float(value) / 1000.0)
        return distance_summary_mm(np.asarray(values))

    roi_output = {}
    for name, values in rois.items():
        before = values["before"]
        after = values["after"]
        if not before or not after:
            roi_output[name] = {
                "status": "NOT_RUN",
                "reason": "NO_POINTS_IN_ROI",
                "before": {"frame_count": len(before)},
                "after": {"frame_count": len(after)},
            }
            continue
        b = {
            "frame_count": len(before),
            "median_of_frame_medians_mm": float(
                np.median([item["median_mm"] for item in before])
            ),
            "p95_of_frame_p95_mm": float(
                np.quantile([item["p95_mm"] for item in before], 0.95)
            ),
        }
        a = {
            "frame_count": len(after),
            "median_of_frame_medians_mm": float(
                np.median([item["median_mm"] for item in after])
            ),
            "p95_of_frame_p95_mm": float(
                np.quantile([item["p95_mm"] for item in after], 0.95)
            ),
        }
        roi_output[name] = {
            "status": "AVAILABLE",
            "before": b,
            "after": a,
            "median_improvement": _improvement(
                b["median_of_frame_medians_mm"], a["median_of_frame_medians_mm"]
            ),
            "p95_improvement": _improvement(
                b["p95_of_frame_p95_mm"], a["p95_of_frame_p95_mm"]
            ),
        }
    translations = np.stack([item[:3, 3] for item in windows]) * 1000
    rotations = np.asarray([_rotation_geodesic_deg(item[:3, :3]) for item in windows])
    return {
        "frame_count": len(records),
        "per_frame": records,
        "symmetric_before_frame_medians": aggregate(
            "before", ("symmetric", "median_mm")
        ),
        "symmetric_after_frame_medians": aggregate("after", ("symmetric", "median_mm")),
        "roi": roi_output,
        "temporal_residual_transform": {
            "window_count": len(windows),
            "T_anchor_from_moving_residual_per_window": [
                item.tolist() for item in windows
            ],
            "translation_mean_mm": translations.mean(axis=0).tolist(),
            "translation_std_mm": translations.std(axis=0).tolist(),
            "translation_norm_std_mm": float(
                np.linalg.norm(translations, axis=1).std()
            ),
            "rotation_geodesic_mean_deg": float(rotations.mean()),
            "rotation_geodesic_std_deg": float(rotations.std()),
        },
    }


def _image_analysis(before: Any, after: Any) -> dict[str, Any]:
    output: dict[str, Any] = {"directions_kept_separate": True}
    for label, directional in (("before", before), ("after", after)):
        output[label] = {}
        for name, values in directional.items():
            u, v, radius = (np.concatenate(values[key]) for key in ("u", "v", "r"))
            residual = np.concatenate(values["e"])
            plane = np.concatenate(values["p"])
            radial_high = max(float(radius.max()), np.finfo(float).eps)
            output[label][name] = {
                "horizontal_full_fov_5_bins": binned_residuals(
                    u,
                    residual,
                    signed_point_to_plane_m=plane,
                    edges=np.linspace(0, 1, 6),
                ),
                "vertical_full_fov_5_bins": binned_residuals(
                    v,
                    residual,
                    signed_point_to_plane_m=plane,
                    edges=np.linspace(0, 1, 6),
                ),
                "radial_observed_5_bins": binned_residuals(
                    radius,
                    residual,
                    signed_point_to_plane_m=plane,
                    edges=np.linspace(0, radial_high, 6),
                ),
                "observed_coverage": {
                    "u_over_w_min_max": [float(u.min()), float(u.max())],
                    "v_over_h_min_max": [float(v.min()), float(v.max())],
                    "radius_min_max": [float(radius.min()), float(radius.max())],
                },
                "radial_trend_mm_per_unit": _bounded_correlation(
                    radius, residual * 1000
                ),
                "inference_note": "bin metrics use five complete distributed tensors; regression is deterministic bounded descriptive evidence",
            }
    return output


def _depth_analysis(before: Any, after: Any) -> dict[str, Any]:
    output: dict[str, Any] = {
        "edge_threshold_m": 0.020,
        "directions_kept_separate": True,
    }
    for label, directional in (("before", before), ("after", after)):
        output[label] = {}
        for name, values in directional.items():
            depth = np.concatenate(values["z"])
            residual = np.concatenate(values["e"])
            plane = np.concatenate(values["p"])
            edge = np.concatenate(values["edge"])
            output[label][name] = {
                "depth_5_bins": _quantile_bins(depth, residual, plane, 5),
                "trend_mm_per_m": _bounded_correlation(depth, residual * 1000),
                "edge": distance_summary_mm(residual[edge]),
                "interior": distance_summary_mm(residual[~edge]),
                "edge_to_interior_median_ratio": float(
                    np.median(residual[edge]) / np.median(residual[~edge])
                ),
            }
    return output


def _bounded_correlation(
    x: np.ndarray, y: np.ndarray, limit: int = 2000
) -> dict[str, Any]:
    if x.size > limit:
        indices = np.linspace(0, x.size - 1, limit, dtype=np.int64)
        x, y = x[indices], y[indices]
    return correlation_summary(x, y)


def _skew_analysis(
    records: list[dict[str, Any]], timing_records: list[dict[str, Any]]
) -> dict[str, Any]:
    skew = np.asarray([item["maximum_skew_ms"] for item in records])
    signed_b = np.asarray(
        [item["cameras"][CAMERA_B]["signed_delta_ms"] for item in timing_records]
    )
    before_median = np.asarray(
        [item["before"]["symmetric"]["median_mm"] for item in records]
    )
    before_p95 = np.asarray([item["before"]["symmetric"]["p95_mm"] for item in records])
    after_median = np.asarray(
        [item["after"]["symmetric"]["median_mm"] for item in records]
    )
    fixed_edges = np.asarray(
        (0.0, 5.0, 10.0, 20.0, max(20.000001, float(skew.max()) + 1e-9))
    )
    fixed_counts = np.histogram(skew, bins=fixed_edges)[0]
    use_quantiles = bool((fixed_counts == 0).sum() >= 2)
    bins = (
        _quantile_bins(skew, before_median / 1000, None, 4)
        if use_quantiles
        else _fixed_bins(skew, before_median / 1000, fixed_edges)
    )
    windows = []
    for indices in np.array_split(np.arange(skew.size), 5):
        windows.append(
            {
                "index_start": int(indices[0]),
                "index_end": int(indices[-1]),
                "maximum_skew_ms": _summary(skew[indices].tolist()),
                "signed_camera_b_delta_ms": _summary(signed_b[indices].tolist()),
                "median_residual_mm": _summary(before_median[indices].tolist()),
            }
        )
    return {
        "static_scene_expected_derivative": "approximately zero",
        "skew_ms": _summary(skew.tolist()),
        "signed_camera_b_delta_ms": _summary(signed_b.tolist()),
        "binning": "quantiles_due_to_uneven_fixed_bin_support"
        if use_quantiles
        else "fixed",
        "fixed_edges_ms": fixed_edges.tolist(),
        "fixed_counts": fixed_counts.tolist(),
        "bins": bins,
        "before_median": correlation_summary(skew, before_median),
        "before_p95": correlation_summary(skew, before_p95),
        "after_median": correlation_summary(skew, after_median),
        "signed_before_median": correlation_summary(signed_b, before_median),
        "temporal_windows": windows,
        "interpretation_guard": "restricted skew support, host completion timestamps, or temporal drift preclude causal sync attribution from p-values alone",
    }


def _fixed_bins(
    x: np.ndarray, residual: np.ndarray, edges: np.ndarray
) -> list[dict[str, Any]]:
    assignment = np.digitize(x, edges[1:-1])
    output = []
    for index in range(edges.size - 1):
        mask = assignment == index
        values = residual[mask]
        output.append(
            {
                "bin": index,
                "low": float(edges[index]),
                "high": float(edges[index + 1]),
                "count": int(mask.sum()),
                "median_mm": None
                if not values.size
                else float(np.median(values) * 1000),
                "p95_mm": None
                if not values.size
                else float(np.quantile(values, 0.95) * 1000),
                "signed_point_to_plane_median_mm": None,
            }
        )
    return output


def _rgb_analysis(selected: dict[int, Any], correction: np.ndarray) -> dict[str, Any]:
    accumulators = {
        CAMERA_A: {key: [] for key in ("color", "radius", "depth", "distance")},
        CAMERA_B: {key: [] for key in ("color", "radius", "depth", "distance")},
    }
    from scipy.spatial import cKDTree

    for cameras in selected.values():
        a, b = cameras[CAMERA_A], cameras[CAMERA_B]
        a_xyz = a["points"][:, :3]
        b_xyz = apply_transform(b["points"][:, :3], correction)
        distance_b, index_b = cKDTree(a_xyz).query(b_xyz, workers=1)
        _accumulate_rgb_direction(accumulators[CAMERA_B], b, a, distance_b, index_b)
        distance_a, index_a = cKDTree(b_xyz).query(a_xyz, workers=1)
        _accumulate_rgb_direction(accumulators[CAMERA_A], a, b, distance_a, index_a)
    directional = {}
    for name, values in accumulators.items():
        colors = np.concatenate(values["color"])
        radii = np.concatenate(values["radius"])
        depths = np.concatenate(values["depth"])
        distances = np.concatenate(values["distance"])
        directional[name] = {
            "correspondence_geometry": distance_summary_mm(distances),
            "rgb_l2_unit_interval": {
                "count": int(colors.size),
                "median": float(np.median(colors)),
                "p95": float(np.quantile(colors, 0.95)),
            },
            "rgb_difference_vs_color_image_radius": _bounded_correlation(radii, colors),
            "rgb_difference_vs_depth": _bounded_correlation(depths, colors),
        }
    all_coefficients = [
        value
        for cameras in selected.values()
        for item in cameras.values()
        for value in item["color_distortion_coeffs"].tolist()
    ]
    return {
        "geometry_correspondence_threshold_mm": 5.0,
        "directions": directional,
        "pinhole_vs_distortion_aware_parity": {
            "bundle_coefficients_all_zero": bool(
                all(abs(value) <= 1e-12 for value in all_coefficients)
            ),
            "result": (
                "EXACT_IDENTITY_FOR_PERSISTED_COEFFICIENTS"
                if all(abs(value) <= 1e-12 for value in all_coefficients)
                else "NOT_RUN_REQUIRES_TARGETED_AUDIT"
            ),
        },
        "caveat": "exposure and white balance also affect RGB difference",
    }


def _accumulate_rgb_direction(
    output: dict[str, list[np.ndarray]],
    source: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    distance: np.ndarray,
    index: np.ndarray,
) -> None:
    close = (
        (distance <= 0.005)
        & source["color_projection_valid"]
        & reference["color_projection_valid"][index]
    )
    if not close.any():
        return
    difference = np.linalg.norm(
        source["points"][close, 3:6] - reference["points"][index[close], 3:6],
        axis=1,
    )
    _width, _height, fx, fy, cx, cy = source["color_intrinsics"]
    uv = source["color_uv"][close]
    radius = np.sqrt(((uv[:, 0] - cx) / fx) ** 2 + ((uv[:, 1] - cy) / fy) ** 2)
    output["color"].append(difference)
    output["radius"].append(radius)
    output["depth"].append(source["depth_m"][close])
    output["distance"].append(distance[close])


def _quantile_bins(
    x: np.ndarray, residual: np.ndarray, signed: np.ndarray | None, bins: int
) -> list[dict[str, Any]]:
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    assignment = np.clip(np.digitize(x, edges[1:-1]), 0, bins - 1)
    output = []
    for index in range(bins):
        mask = assignment == index
        output.append(
            {
                "bin": index,
                "low": float(edges[index]),
                "high": float(edges[index + 1]),
                "count": int(mask.sum()),
                "median_mm": float(np.median(residual[mask]) * 1000),
                "p95_mm": float(np.quantile(residual[mask], 0.95) * 1000),
                "signed_point_to_plane_median_mm": None
                if signed is None
                else float(np.nanmedian(signed[mask]) * 1000),
            }
        )
    return output


def _render_all(
    root: Path,
    selected: dict[int, Any],
    correction: np.ndarray,
    board: Any,
    cube: Any,
    budget: int,
) -> None:
    from scipy.spatial import cKDTree

    middle = selected[sorted(selected)[len(selected) // 2]]
    a, b = middle[CAMERA_A]["points"], middle[CAMERA_B]["points"]
    distance_b, _ = cKDTree(a[:, :3]).query(b[:, :3], workers=1)
    distance_a, _ = cKDTree(b[:, :3]).query(a[:, :3], workers=1)
    overlap_a = distance_a <= 0.030
    overlap_b = distance_b <= 0.030
    shots = root / "screenshots"
    _render_pair(
        a[overlap_a],
        b[overlap_b],
        shots / "geometry-overview.png",
        mode="camera",
        budget=budget,
    )
    _render_pair(
        a[_board_mask(a[:, :3], board)],
        b[_board_mask(b[:, :3], board)],
        shots / "geometry-board.png",
        mode="camera",
        budget=budget,
    )
    if cube is not None:
        _render_pair(
            a[_cube_mask(a[:, :3], cube)],
            b[_cube_mask(b[:, :3], cube)],
            shots / "geometry-cube.png",
            mode="camera",
            budget=budget,
        )
    edge_a, edge_b = middle[CAMERA_A]["edge"], middle[CAMERA_B]["edge"]
    _render_pair(
        a[edge_a & overlap_a],
        b[edge_b & overlap_b],
        shots / "geometry-edge.png",
        mode="camera",
        budget=budget,
    )
    _render_pair(
        a[overlap_a],
        b[overlap_b],
        shots / "rgb-overview.png",
        mode="rgb",
        budget=budget,
    )
    _render_pair(
        a[_board_mask(a[:, :3], board)],
        b[_board_mask(b[:, :3], board)],
        shots / "rgb-board.png",
        mode="rgb",
        budget=budget,
    )
    if cube is not None:
        _render_pair(
            a[_cube_mask(a[:, :3], cube)],
            b[_cube_mask(b[:, :3], cube)],
            shots / "rgb-cube.png",
            mode="rgb",
            budget=budget,
        )
    _render_pair(
        a[overlap_a],
        b[overlap_b],
        shots / "before-residual.png",
        mode="camera",
        budget=budget,
    )
    corrected = b.copy()
    corrected[:, :3] = apply_transform(b[:, :3], correction)
    _render_pair(
        a[overlap_a],
        corrected[overlap_b],
        shots / "after-residual.png",
        mode="camera",
        budget=budget,
    )


def _render_pair(
    a: np.ndarray, b: np.ndarray, path: Path, *, mode: str, budget: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per = max(1, budget // 2)
    aa = a[:: max(1, math.ceil(a.shape[0] / per))]
    bb = b[:: max(1, math.ceil(b.shape[0] / per))]
    colors_a = (
        np.tile((1.0, 0.05, 0.05), (aa.shape[0], 1)) if mode == "camera" else aa[:, 3:6]
    )
    colors_b = (
        np.tile((0.05, 0.2, 1.0), (bb.shape[0], 1)) if mode == "camera" else bb[:, 3:6]
    )
    figure = plt.figure(figsize=(16, 8), dpi=160)
    for slot, (elev, azim, title) in enumerate(
        ((90, -90, "top"), (24, -58, "oblique")), 1
    ):
        axis = figure.add_subplot(1, 2, slot, projection="3d")
        axis.scatter(aa[:, 0], aa[:, 1], aa[:, 2], s=0.15, c=colors_a)
        axis.scatter(bb[:, 0], bb[:, 1], bb[:, 2], s=0.15, c=colors_b)
        axis.view_init(elev=elev, azim=azim)
        axis.set_title(
            f"{title}: actual={a.shape[0] + b.shape[0]}, viewer={aa.shape[0] + bb.shape[0]}"
        )
        axis.set(xlabel="workspace x", ylabel="workspace y", zlabel="workspace z")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _write_rerun(
    root: Path, selected: dict[int, Any], correction: np.ndarray, budget: int
) -> None:
    try:
        import rerun as rr
    except ImportError:
        return
    path = root / "rerun" / "cross-camera-alignment.rrd"
    rr.init("cross-camera-alignment-diagnostic", spawn=False)
    rr.save(str(path))
    middle = selected[sorted(selected)[len(selected) // 2]]
    a, b = middle[CAMERA_A]["points"], middle[CAMERA_B]["points"]
    from scipy.spatial import cKDTree

    distance_b, _ = cKDTree(a[:, :3]).query(b[:, :3], workers=1)
    distance_a, _ = cKDTree(b[:, :3]).query(a[:, :3], workers=1)
    a = a[distance_a <= 0.030]
    b = b[distance_b <= 0.030]
    per = max(1, budget // 2)
    aa = a[:: max(1, math.ceil(a.shape[0] / per))]
    bb = b[:: max(1, math.ceil(b.shape[0] / per))]
    rr.log("diagnostics/geometry/camera_a", rr.Points3D(aa[:, :3], colors=[255, 0, 0]))
    rr.log("diagnostics/geometry/camera_b", rr.Points3D(bb[:, :3], colors=[0, 60, 255]))
    rr.log(
        "diagnostics/rgb/camera_a",
        rr.Points3D(
            aa[:, :3], colors=np.clip(aa[:, 3:6] * 255, 0, 255).astype(np.uint8)
        ),
    )
    rr.log(
        "diagnostics/rgb/camera_b",
        rr.Points3D(
            bb[:, :3], colors=np.clip(bb[:, 3:6] * 255, 0, 255).astype(np.uint8)
        ),
    )
    rr.log(
        "diagnostics/rgb/combined",
        rr.Points3D(
            np.concatenate((aa[:, :3], bb[:, :3])),
            colors=np.clip(
                np.concatenate((aa[:, 3:6], bb[:, 3:6])) * 255, 0, 255
            ).astype(np.uint8),
        ),
    )
    corrected = apply_transform(bb[:, :3], correction)
    rr.log(
        "diagnostics/residual/before",
        rr.Points3D(
            np.concatenate((aa[:, :3], bb[:, :3])),
            colors=np.concatenate(
                (
                    np.tile((255, 0, 0), (aa.shape[0], 1)),
                    np.tile((0, 60, 255), (bb.shape[0], 1)),
                )
            ),
        ),
    )
    rr.log(
        "diagnostics/residual/after_se3",
        rr.Points3D(
            np.concatenate((aa[:, :3], corrected)),
            colors=np.concatenate(
                (
                    np.tile((255, 0, 0), (aa.shape[0], 1)),
                    np.tile((0, 60, 255), (bb.shape[0], 1)),
                )
            ),
        ),
    )
    _, index = cKDTree(aa[:, :3]).query(corrected, workers=1)
    selection = np.linspace(
        0, corrected.shape[0] - 1, min(1000, corrected.shape[0]), dtype=int
    )
    rr.log(
        "diagnostics/residual/vectors",
        rr.LineStrips3D(
            np.stack((corrected[selection], aa[index[selection], :3]), axis=1),
            colors=[255, 220, 0],
        ),
    )


def _transform_report(
    transform: np.ndarray,
    point_to_plane: dict[str, Any],
    point_to_point: dict[str, Any],
    reverse_point_to_plane: dict[str, Any],
) -> dict[str, Any]:
    rotation = transform[:3, :3]
    rx = math.atan2(rotation[2, 1], rotation[2, 2])
    ry = math.atan2(
        -rotation[2, 0], math.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2)
    )
    rz = math.atan2(rotation[1, 0], rotation[0, 0])
    translation = transform[:3, 3] * 1000
    reverse_as_forward = np.linalg.inv(
        np.asarray(reverse_point_to_plane["T_anchor_from_moving_residual"])
    )
    disagreement = reverse_as_forward @ np.linalg.inv(transform)
    return {
        "schema_version": f"{SCHEMA}.residual-transform",
        "convention": "Delta_T_workspace_surface_anchor_camera_a_from_camera_b_workspace_surface; p_b_corrected = Delta_T @ p_b_workspace",
        "anchor": CAMERA_A,
        "moving": CAMERA_B,
        "diagnostic_only": True,
        "written_back_to_camera_bundle": False,
        "matrix": transform.tolist(),
        "translation": {
            "dx_mm": float(translation[0]),
            "dy_mm": float(translation[1]),
            "dz_mm": float(translation[2]),
            "norm_mm": float(np.linalg.norm(translation)),
        },
        "rotation": {
            "rx_deg": math.degrees(rx),
            "ry_deg": math.degrees(ry),
            "rz_deg": math.degrees(rz),
            "geodesic_deg": _rotation_geodesic_deg(rotation),
        },
        "primary_estimator": {
            key: _jsonable(value)
            for key, value in point_to_plane.items()
            if key != "T_anchor_from_moving_residual"
        },
        "reverse_estimator": {
            **{
                key: _jsonable(value)
                for key, value in reverse_point_to_plane.items()
                if key != "T_anchor_from_moving_residual"
            },
            "inverse_as_forward_matrix": reverse_as_forward.tolist(),
            "forward_reverse_translation_disagreement_mm": float(
                np.linalg.norm(disagreement[:3, 3]) * 1000
            ),
            "forward_reverse_rotation_disagreement_deg": _rotation_geodesic_deg(
                disagreement[:3, :3]
            ),
        },
        "comparison_point_to_point": {
            **{
                key: _jsonable(value)
                for key, value in point_to_point.items()
                if key != "T_anchor_from_moving_residual"
            },
            "matrix": np.asarray(
                point_to_point["T_anchor_from_moving_residual"]
            ).tolist(),
        },
    }


def _signed_summary(values_m: np.ndarray) -> dict[str, Any]:
    finite = values_m[np.isfinite(values_m)]
    return {
        "count": int(finite.size),
        "signed_median_mm": float(np.median(finite) * 1000),
        "absolute": distance_summary_mm(np.abs(finite)),
    }


def _rotation_geodesic_deg(rotation: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))


def _improvement(before: float, after: float) -> float:
    return float((before - after) / before) if before > 0 else 0.0


def _board(raw: dict[str, Any]) -> ExpectedPlaneRegion:
    return ExpectedPlaneRegion(
        frame=str(raw.get("frame", "workspace")),
        x=tuple(float(x) for x in raw["x"]),
        y=tuple(float(x) for x in raw["y"]),
        expected_z_m=float(raw.get("expected_z_m", 0)),
        z_search_range_m=tuple(float(x) for x in raw["z_search_range_m"]),
    )


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("summary requires values")
    return {
        "min": min(values),
        "p50": statistics.median(values),
        "p95": float(np.quantile(values, 0.95)),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def _private_root(value: str | Path) -> Path:
    path = Path(value).resolve()
    local = (Path.cwd() / ".local").resolve()
    if not path.is_relative_to(local):
        raise ValueError("real diagnostic artifacts must remain under .local/")
    return path


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    main()
