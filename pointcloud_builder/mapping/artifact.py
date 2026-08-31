"""Atomic, native-volume TSDF map artifacts with save/load parity receipts."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pointcloud_builder.mapping.config import TsdfMapConfig, load_tsdf_config
from pointcloud_builder.mapping.open3d import Open3dTsdfMap
from pointcloud_builder.mapping.types import MapExtraction, TsdfMapArtifact
from pointcloud_builder.mapping.validation import (
    load_json,
    validate_checksums,
    write_checksums,
)

_SCHEMA_V1 = "pointcloud-builder.tsdf-map-artifact.v1"
_SCHEMA_V2 = "pointcloud-builder.tsdf-map-artifact.v2"
_SCHEMA = "pointcloud-builder.tsdf-map-artifact.v3"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def write_tsdf_map_artifact(
    output: str | Path,
    *,
    mapper: Open3dTsdfMap,
    source_recording_sha256: str,
    integration_metrics: dict[str, Any],
    rig_calibration_provenance: dict[str, Any] | None = None,
) -> TsdfMapArtifact:
    destination = Path(output)
    if mapper.state.lifecycle != "frozen":
        raise ValueError("only a frozen TSDF map can be published")
    if _SHA256.fullmatch(source_recording_sha256) is None:
        raise ValueError("source recording receipt must be a lowercase SHA-256")
    if destination.exists():
        raise FileExistsError(f"TSDF map output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        (temporary / "screenshots").mkdir()
        config_path = temporary / "config.resolved.yaml"
        config_path.write_text(
            yaml.safe_dump(mapper.config.to_dict(), sort_keys=False), encoding="utf-8"
        )
        volume_path = temporary / "volume.npz"
        mapper.save(volume_path)
        extraction = mapper.extract()
        _write_point_ply(temporary / "point_cloud.ply", extraction.points)
        _write_point_ply(temporary / "point_cloud_raw.ply", extraction.raw_points)
        _write_point_ply(
            temporary / "point_cloud_cropped.ply", extraction.cropped_points
        )
        _write_point_ply(
            temporary / "point_cloud_sampled.ply", extraction.sampled_points
        )
        _write_mesh_ply(
            temporary / "mesh.ply", extraction.vertices, extraction.triangles
        )
        before = mapper.volume_statistics()
        parity = _save_load_parity(
            mapper.config,
            mapper.workspace_frame,
            volume_path,
            before,
            extraction,
        )
        metrics = {
            "schema_version": "pointcloud-builder.tsdf-map-metrics.v1",
            "state": mapper.state.__dict__,
            "volume": before,
            "extraction": {
                "point_count": extraction.point_count,
                "raw_point_count": extraction.raw_point_count,
                "cropped_point_count": extraction.cropped_point_count,
                "sampled_point_count": extraction.sampled_point_count,
                "vertex_count": extraction.vertex_count,
                "triangle_count": extraction.triangle_count,
                "extraction_ms": extraction.extraction_ms,
                "extract_point_cloud_ms": extraction.extract_point_cloud_ms,
                "extract_mesh_ms": extraction.extract_mesh_ms,
                "post_crop_ms": extraction.post_crop_ms,
                "post_sampling_ms": extraction.post_sampling_ms,
                "extract_raw_world_cloud_ms": extraction.extract_raw_world_cloud_ms,
                "extract_cropped_world_cloud_ms": extraction.extract_cropped_world_cloud_ms,
                "extract_sampled_world_cloud_ms": extraction.extract_sampled_world_cloud_ms,
            },
            "integration": integration_metrics,
            "save_load_parity": parity,
        }
        _write_json(temporary / "metrics.json", metrics)
        _write_json(
            temporary / "source_recording.json",
            {
                "schema_version": "pointcloud-builder.tsdf-source-recording.v1",
                "recording_manifest_sha256": source_recording_sha256,
            },
        )
        rig_calibration = _normalize_rig_calibration_provenance(
            rig_calibration_provenance,
            workspace_frame=mapper.workspace_frame,
        )
        _write_json(temporary / "rig_calibration.json", rig_calibration)
        manifest = {
            "schema_version": _SCHEMA,
            "workspace_frame": mapper.workspace_frame,
            "map_revision": mapper.state.map_revision,
            "lifecycle": mapper.state.lifecycle,
            "files": {
                "config": "config.resolved.yaml",
                "volume": "volume.npz",
                "point_cloud": "point_cloud.ply",
                "point_cloud_raw": "point_cloud_raw.ply",
                "point_cloud_cropped": "point_cloud_cropped.ply",
                "point_cloud_sampled": "point_cloud_sampled.ply",
                "mesh": "mesh.ply",
                "metrics": "metrics.json",
                "source_recording": "source_recording.json",
                "rig_calibration": "rig_calibration.json",
            },
            "rig_calibration": {
                "calibration_mode": rig_calibration["calibration_mode"],
                "rig_calibration_fingerprint": rig_calibration[
                    "rig_calibration_fingerprint"
                ],
                "solution_fingerprint": rig_calibration["solution_fingerprint"],
                "camera_set": rig_calibration["camera_set"],
            },
        }
        _write_json(temporary / "manifest.json", manifest)
        members = sorted(
            path.relative_to(temporary).as_posix()
            for path in temporary.rglob("*")
            if path.is_file()
        )
        write_checksums(temporary, members)
        validate_tsdf_map_artifact(temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return TsdfMapArtifact(root=destination, manifest=manifest)


def validate_tsdf_map_artifact(root: str | Path) -> dict[str, Any]:
    artifact = Path(root)
    checksums = validate_checksums(artifact)
    if not (artifact / "screenshots").is_dir():
        raise ValueError("TSDF map artifact is missing screenshots directory")
    manifest = load_json(artifact / "manifest.json")
    schema = manifest.get("schema_version")
    if schema not in {_SCHEMA_V1, _SCHEMA_V2, _SCHEMA}:
        raise ValueError("unsupported TSDF map artifact schema")
    if (
        not isinstance(manifest.get("workspace_frame"), str)
        or not manifest["workspace_frame"].strip()
        or manifest.get("lifecycle") != "frozen"
    ):
        raise ValueError("TSDF map artifact must describe a frozen workspace map")
    if (
        isinstance(manifest.get("map_revision"), bool)
        or not isinstance(manifest.get("map_revision"), int)
        or manifest["map_revision"] < 0
    ):
        raise ValueError("TSDF map revision must be a non-negative integer")
    expected_files = {
        "config": "config.resolved.yaml",
        "volume": "volume.npz",
        "point_cloud": "point_cloud.ply",
        "mesh": "mesh.ply",
        "metrics": "metrics.json",
        "source_recording": "source_recording.json",
    }
    if schema in {_SCHEMA_V2, _SCHEMA}:
        expected_files.update(
            {
                "point_cloud_raw": "point_cloud_raw.ply",
                "point_cloud_cropped": "point_cloud_cropped.ply",
                "point_cloud_sampled": "point_cloud_sampled.ply",
            }
        )
    if schema == _SCHEMA:
        expected_files["rig_calibration"] = "rig_calibration.json"
    if manifest.get("files") != expected_files:
        raise ValueError("TSDF map manifest exact file contract mismatch")
    if not set(expected_files.values()) <= set(checksums):
        raise ValueError("TSDF map manifest member is absent from checksums")
    metrics = load_json(artifact / "metrics.json")
    state = metrics.get("state")
    if (
        metrics.get("schema_version") != "pointcloud-builder.tsdf-map-metrics.v1"
        or not isinstance(state, dict)
        or state.get("workspace_frame") != manifest["workspace_frame"]
        or state.get("lifecycle") != "frozen"
        or state.get("map_revision") != manifest["map_revision"]
    ):
        raise ValueError("TSDF map metrics/manifest identity mismatch")
    parity = metrics.get("save_load_parity")
    if not isinstance(parity, dict) or not parity.get("passed"):
        gates = parity.get("gates") if isinstance(parity, dict) else None
        failed = (
            sorted(name for name, passed in gates.items() if passed is not True)
            if isinstance(gates, dict)
            else []
        )
        detail = "" if not failed else ": " + ", ".join(failed)
        raise ValueError("TSDF map save/load parity did not pass" + detail)
    source = load_json(artifact / "source_recording.json")
    digest = source.get("recording_manifest_sha256")
    if (
        source.get("schema_version") != "pointcloud-builder.tsdf-source-recording.v1"
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
    ):
        raise ValueError("TSDF source recording receipt is invalid")
    if schema == _SCHEMA:
        rig_calibration = load_json(artifact / "rig_calibration.json")
        checked = _validate_map_rig_calibration(
            rig_calibration, workspace_frame=manifest["workspace_frame"]
        )
        summary = manifest.get("rig_calibration")
        expected_summary = {
            "calibration_mode": checked["calibration_mode"],
            "rig_calibration_fingerprint": checked[
                "rig_calibration_fingerprint"
            ],
            "solution_fingerprint": checked["solution_fingerprint"],
            "camera_set": checked["camera_set"],
        }
        if summary != expected_summary:
            raise ValueError("TSDF map rig calibration summary mismatch")
    load_tsdf_config(artifact / "config.resolved.yaml")
    return manifest


def validate_tsdf_map_rig_calibration_compatibility(
    root: str | Path, live_provenance: dict[str, Any]
) -> None:
    """Fail closed when an initial map and live rig use different geometry."""

    artifact = Path(root)
    manifest = validate_tsdf_map_artifact(artifact)
    if manifest["schema_version"] == _SCHEMA:
        map_provenance = load_json(artifact / "rig_calibration.json")
    else:
        map_provenance = {
            "calibration_mode": "fixed_provision",
            "rig_calibration_fingerprint": None,
        }
    mismatches = []
    for name in ("calibration_mode", "rig_calibration_fingerprint"):
        if map_provenance.get(name) != live_provenance.get(name):
            mismatches.append(name)
    if mismatches:
        raise ValueError(
            "initial map/live rig calibration mismatch: " + ", ".join(mismatches)
        )


def load_tsdf_map_artifact(root: str | Path) -> Open3dTsdfMap:
    artifact = Path(root)
    manifest = validate_tsdf_map_artifact(artifact)
    config = load_tsdf_config(artifact / "config.resolved.yaml")
    mapper = Open3dTsdfMap(config, workspace_frame=manifest["workspace_frame"])
    mapper.load(artifact / "volume.npz")
    return mapper


def write_extracted_geometry(output: str | Path, extraction: MapExtraction) -> Path:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"geometry output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        _write_point_ply(temporary / "point_cloud.ply", extraction.points)
        _write_point_ply(temporary / "raw.ply", extraction.raw_points)
        _write_point_ply(temporary / "cropped.ply", extraction.cropped_points)
        _write_point_ply(temporary / "sampled.ply", extraction.sampled_points)
        _write_mesh_ply(
            temporary / "mesh.ply", extraction.vertices, extraction.triangles
        )
        _write_json(
            temporary / "extraction.json",
            {
                "schema_version": "pointcloud-builder.tsdf-extraction.v1",
                "point_count": extraction.point_count,
                "raw_point_count": extraction.raw_point_count,
                "cropped_point_count": extraction.cropped_point_count,
                "sampled_point_count": extraction.sampled_point_count,
                "vertex_count": extraction.vertex_count,
                "triangle_count": extraction.triangle_count,
                "extraction_ms": extraction.extraction_ms,
                "extract_point_cloud_ms": extraction.extract_point_cloud_ms,
                "extract_mesh_ms": extraction.extract_mesh_ms,
                "post_crop_ms": extraction.post_crop_ms,
                "post_sampling_ms": extraction.post_sampling_ms,
            },
        )
        members = sorted(
            path.relative_to(temporary).as_posix()
            for path in temporary.iterdir()
            if path.is_file()
        )
        write_checksums(temporary, members)
        validate_checksums(temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _save_load_parity(
    config: TsdfMapConfig,
    workspace_frame: str,
    volume_path: Path,
    before: dict[str, Any],
    extraction_before: MapExtraction,
) -> dict[str, Any]:
    started = time.perf_counter()
    loaded = Open3dTsdfMap(config, workspace_frame=workspace_frame)
    try:
        loaded.load(volume_path)
        after = loaded.volume_statistics()
        extraction_after = loaded.extract()
    finally:
        loaded.close()
    distance = _sampled_symmetric_distance(
        extraction_before.raw_points,
        extraction_after.raw_points,
        maximum_points=2000,
    )
    cropped_distance = _sampled_symmetric_distance(
        extraction_before.cropped_points,
        extraction_after.cropped_points,
        maximum_points=2000,
    )
    sampled_distance = _sampled_symmetric_distance(
        extraction_before.sampled_points,
        extraction_after.sampled_points,
        maximum_points=2000,
    )
    shapes_equal = all(
        before["attributes"][name]["shape"] == after["attributes"][name]["shape"]
        for name in ("tsdf", "weight", "color")
    )
    statistics_equal = all(
        _volume_statistics_equal(before["attributes"][name], after["attributes"][name])
        for name in ("tsdf", "weight")
    )
    counts_equal = (
        extraction_before.point_count == extraction_after.point_count
        and extraction_before.cropped_point_count
        == extraction_after.cropped_point_count
        and extraction_before.sampled_point_count
        == extraction_after.sampled_point_count
        and extraction_before.vertex_count == extraction_after.vertex_count
        and extraction_before.triangle_count == extraction_after.triangle_count
    )
    gates = {
        "active_blocks_equal": before["active_block_count"]
        == after["active_block_count"],
        "attribute_shapes_equal": shapes_equal,
        "tsdf_weight_statistics_equal": statistics_equal,
        "geometry_counts_equal": counts_equal,
        "raw_geometry_max_distance_le_1um": distance["maximum_m"] <= 1e-6,
        "cropped_geometry_max_distance_le_1um": (cropped_distance["maximum_m"] <= 1e-6),
        "postprocessed_geometry_max_distance_le_1um": (
            sampled_distance["maximum_m"] <= 1e-6
        ),
    }
    passed = bool(all(gates.values()))
    return {
        "active_blocks_equal": gates["active_blocks_equal"],
        "attribute_shapes_equal": shapes_equal,
        "tsdf_weight_statistics_equal": statistics_equal,
        "geometry_counts_equal": counts_equal,
        "sampled_symmetric_distance_m": distance,
        "cropped_symmetric_distance_m": cropped_distance,
        "postprocessed_symmetric_distance_m": sampled_distance,
        "gates": gates,
        "validation_ms": (time.perf_counter() - started) * 1000.0,
        "passed": passed,
    }


def _volume_statistics_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Compare order-invariant summaries at the precision of their source values."""

    exact_fields = ("shape", "dtype", "minimum", "maximum", "nonzero_count")
    if any(left.get(name) != right.get(name) for name in exact_fields):
        return False
    try:
        left_mean = float(left["mean"])
        right_mean = float(right["mean"])
    except (KeyError, TypeError, ValueError):
        return False
    # TSDF means originate from float32 values. Hash-map save/load may reorder
    # blocks, changing only float32 reduction roundoff while all stored values,
    # extrema, occupancy, and extracted geometry remain identical.
    epsilon = float(np.finfo(np.float32).eps)
    return bool(
        np.isfinite((left_mean, right_mean)).all()
        and np.isclose(left_mean, right_mean, rtol=8.0 * epsilon, atol=epsilon)
    )


def _sampled_symmetric_distance(
    left: np.ndarray, right: np.ndarray, *, maximum_points: int
) -> dict[str, float]:
    if not len(left) or not len(right):
        value = 0.0 if len(left) == len(right) else float("inf")
        return {"median_m": value, "p95_m": value, "maximum_m": value}
    left_sample = left[
        np.linspace(0, len(left) - 1, min(len(left), maximum_points))
        .round()
        .astype(int)
    ]
    right_sample = right[
        np.linspace(0, len(right) - 1, min(len(right), maximum_points))
        .round()
        .astype(int)
    ]

    def nearest(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        values = []
        for start in range(0, len(source), 128):
            delta = source[start : start + 128, None, :] - target[None, :, :]
            values.append(np.sqrt(np.square(delta).sum(axis=2)).min(axis=1))
        return np.concatenate(values)

    # Sample query points only. The complete opposite extraction remains the
    # reference so a harmless Open3D ordering change cannot create false error.
    distances = np.concatenate(
        (nearest(left_sample, right), nearest(right_sample, left))
    )
    return {
        "median_m": float(np.median(distances)),
        "p95_m": float(np.quantile(distances, 0.95)),
        "maximum_m": float(distances.max()),
    }


def _write_point_ply(path: Path, points: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n"
        )
        np.savetxt(stream, points, fmt="%.9g %.9g %.9g")


def _write_mesh_ply(path: Path, vertices: np.ndarray, triangles: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(vertices)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            f"element face {len(triangles)}\n"
            "property list uchar int vertex_indices\nend_header\n"
        )
        np.savetxt(stream, vertices, fmt="%.9g %.9g %.9g")
        if len(triangles):
            np.savetxt(stream, triangles, fmt="3 %d %d %d")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _normalize_rig_calibration_provenance(
    value: dict[str, Any] | None, *, workspace_frame: str
) -> dict[str, Any]:
    raw = dict(value or {})
    mode = raw.get("calibration_mode", "fixed_provision")
    normalized = {
        "schema_version": "pointcloud-builder.tsdf-rig-calibration.v1",
        "workspace_frame": workspace_frame,
        "calibration_mode": mode,
        "rig_calibration_schema": raw.get("rig_calibration_schema"),
        "rig_calibration_fingerprint": raw.get("rig_calibration_fingerprint"),
        "solution_fingerprint": raw.get("solution_fingerprint"),
        "camera_set": sorted(raw.get("camera_bundle_hashes", {})),
        "camera_bundle_hashes": dict(sorted(raw.get("camera_bundle_hashes", {}).items())),
        "production_applied": True,
    }
    return _validate_map_rig_calibration(normalized, workspace_frame=workspace_frame)


def _validate_map_rig_calibration(
    value: object, *, workspace_frame: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("TSDF map rig calibration must be a mapping")
    if (
        value.get("schema_version")
        != "pointcloud-builder.tsdf-rig-calibration.v1"
        or value.get("workspace_frame") != workspace_frame
        or value.get("production_applied") is not True
    ):
        raise ValueError("TSDF map rig calibration identity mismatch")
    camera_set = value.get("camera_set")
    hashes = value.get("camera_bundle_hashes")
    if (
        not isinstance(camera_set, list)
        or camera_set != sorted(set(camera_set))
        or not isinstance(hashes, dict)
        or sorted(hashes) != camera_set
    ):
        raise ValueError("TSDF map camera set/hash mismatch")
    mode = value.get("calibration_mode")
    if mode == "fixed_provision":
        if any(
            value.get(name) is not None
            for name in (
                "rig_calibration_schema",
                "rig_calibration_fingerprint",
                "solution_fingerprint",
            )
        ):
            raise ValueError("fixed TSDF map cannot claim deployed calibration")
    elif mode == "validated_multipose_deployment":
        if value.get("rig_calibration_schema") != (
            "pointcloud-builder.rig-calibration-deployment.v1"
        ):
            raise ValueError("TSDF map deployment schema mismatch")
        for name in ("rig_calibration_fingerprint", "solution_fingerprint"):
            digest = value.get(name)
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ValueError(f"TSDF map {name} is invalid")
        if not camera_set:
            raise ValueError("deployed TSDF map requires a camera set")
        if any(
            not isinstance(digest, str) or _SHA256.fullmatch(digest) is None
            for digest in hashes.values()
        ):
            raise ValueError("TSDF map CameraBundle hash is invalid")
    else:
        raise ValueError("TSDF map calibration mode is unsupported")
    return dict(value)
