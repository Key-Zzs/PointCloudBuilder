#!/usr/bin/env python3
"""Aggregate M8 interference controls and two-run repeatability."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pointcloud_builder.fusion import evaluate_interference, evaluate_repeatability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-a-only", required=True)
    parser.add_argument("--native-b-only", required=True)
    parser.add_argument("--native-ab", required=True)
    parser.add_argument("--ffs-a-only", required=True)
    parser.add_argument("--ffs-b-only", required=True)
    parser.add_argument("--ffs-ab", required=True)
    parser.add_argument("--formal-run-1", required=True)
    parser.add_argument("--formal-run-2", required=True)
    parser.add_argument("--m7-ffs-report", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    native_a_path = Path(args.native_a_only)
    native_b_path = Path(args.native_b_only)
    ffs_a_path = Path(args.ffs_a_only)
    ffs_b_path = Path(args.ffs_b_only)
    if native_a_path.resolve() == native_b_path.resolve():
        raise ValueError("native A-only and B-only reports must be distinct")
    if ffs_a_path.resolve() == ffs_b_path.resolve():
        raise ValueError("FFS A-only and B-only reports must be distinct")
    if Path(args.formal_run_1).resolve() == Path(args.formal_run_2).resolve():
        raise ValueError("formal runs must be independent report files")

    native_single = {
        "camera_a": _single(
            native_a_path,
            expected_camera="camera_a",
            expected_depth="native",
            valid_key="valid_depth_ratio",
        ),
        "camera_b": _single(
            native_b_path,
            expected_camera="camera_b",
            expected_depth="native",
            valid_key="valid_depth_ratio",
        ),
    }
    native_ab = _concurrent_live(
        Path(args.native_ab), valid_key="valid_depth_ratio"
    )
    ffs_single_depth = {
        "camera_a": _single(
            ffs_a_path,
            expected_camera="camera_a",
            expected_depth="ffs_stereo",
            valid_key="valid_depth_ratio",
        ),
        "camera_b": _single(
            ffs_b_path,
            expected_camera="camera_b",
            expected_depth="ffs_stereo",
            valid_key="valid_depth_ratio",
        ),
    }
    ffs_single_disparity = {
        "camera_a": _single(
            ffs_a_path,
            expected_camera="camera_a",
            expected_depth="ffs_stereo",
            valid_key="valid_disparity_ratio",
        ),
        "camera_b": _single(
            ffs_b_path,
            expected_camera="camera_b",
            expected_depth="ffs_stereo",
            valid_key="valid_disparity_ratio",
        ),
    }
    ffs_ab_depth = _concurrent_fusion(
        Path(args.ffs_ab), valid_key="valid_depth_ratio_p50"
    )
    ffs_ab_disparity = _concurrent_fusion(
        Path(args.ffs_ab), valid_key="valid_disparity_ratio_p50"
    )
    interference = {
        "native_depth": evaluate_interference(native_single, native_ab),
        "ffs_depth": evaluate_interference(ffs_single_depth, ffs_ab_depth),
        "ffs_disparity": evaluate_interference(
            ffs_single_disparity, ffs_ab_disparity
        ),
    }
    first = _load(Path(args.formal_run_1))
    second = _load(Path(args.formal_run_2))
    _validate_formal(first, Path(args.formal_run_1))
    _validate_formal(second, Path(args.formal_run_2))
    m7_ffs = _validate_m7_ffs(_load(Path(args.m7_ffs_report)), Path(args.m7_ffs_report))
    repeatability = evaluate_repeatability(
        first["repeatability_projection"], second["repeatability_projection"]
    )
    report = {
        "schema_version": "pointcloud-builder.m8-aggregate-acceptance.v1",
        "interference_protocol": {
            "sequence": ["camera_a_only", "camera_b_only", "camera_a_and_b"],
            "device_options_modified": False,
            "emitter_modified": False,
            "laser_power_modified": False,
            "exposure_or_gain_modified": False,
        },
        "interference": interference,
        "repeatability": repeatability,
        "formal_runs_passed": bool(first["passed"] and second["passed"]),
        "m7_ffs_performance": m7_ffs,
        "input_sha256": {
            "native_a_only": _sha256(native_a_path),
            "native_b_only": _sha256(native_b_path),
            "native_ab": _sha256(Path(args.native_ab)),
            "ffs_a_only": _sha256(ffs_a_path),
            "ffs_b_only": _sha256(ffs_b_path),
            "ffs_ab": _sha256(Path(args.ffs_ab)),
            "formal_run_1": _sha256(Path(args.formal_run_1)),
            "formal_run_2": _sha256(Path(args.formal_run_2)),
            "m7_ffs": _sha256(Path(args.m7_ffs_report)),
        },
        "passed": bool(
            all(item["passed"] for item in interference.values())
            and repeatability["passed"]
            and first["passed"]
            and second["passed"]
            and m7_ffs["passed"]
        ),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("M8 aggregate acceptance failed")


def _single(
    path: Path,
    *,
    expected_camera: str,
    expected_depth: str,
    valid_key: str,
) -> dict[str, float]:
    report = _load(path)
    if report.get("schema_version") != "pointcloud-builder.live-single-camera.v1":
        raise ValueError(f"{path} has an unexpected single-camera schema")
    if report.get("camera_name") != expected_camera:
        raise ValueError(f"{path} is not the {expected_camera} control")
    if report.get("depth_source") != expected_depth:
        raise ValueError(f"{path} is not a {expected_depth} control")
    if not report.get("passed"):
        raise ValueError(f"{path} did not pass its acquisition route")
    primary = report["runs"][0]
    if int(primary.get("received_frames", 0)) < 300:
        raise ValueError(f"{path} has fewer than 300 control frames")
    valid = primary["stage"][valid_key]
    if valid is None:
        raise ValueError(f"{path} has no {valid_key}")
    return {
        "valid_ratio": float(valid),
        "board_median_abs_z_m": float(primary["plane"]["median_abs_z_m"]),
        "board_p95_m": float(primary["plane"]["p95_abs_z_m"]),
        "board_rmse_m": float(primary["plane"]["rmse_m"]),
        "outlier_ratio": float(primary["plane"]["outlier_ratio"]),
    }


def _concurrent_live(path: Path, *, valid_key: str) -> dict[str, dict[str, float]]:
    report = _load(path)
    if report.get("schema_version") != "pointcloud-builder.live-rig-acceptance.v1":
        raise ValueError(f"{path} has an unexpected concurrent schema")
    if report.get("depth_modes") != {"camera_a": "native", "camera_b": "native"}:
        raise ValueError(f"{path} is not the native AB control")
    if not report.get("passed"):
        raise ValueError(f"{path} did not pass its concurrent route")
    primary = report["runs"][0]
    if int(primary.get("acquisition", {}).get("matcher", {}).get("matched_sets", 0)) < 300:
        raise ValueError(f"{path} has fewer than 300 matched controls")
    return {
        name: {
            "valid_ratio": float(primary["per_camera_stage"][name][valid_key]["p50"]),
            "board_median_abs_z_m": float(primary["per_camera_plane"][name]["median_abs_z_m"]),
            "board_p95_m": float(primary["per_camera_plane"][name]["p95_abs_z_m"]),
            "board_rmse_m": float(primary["per_camera_plane"][name]["rmse_m"]),
            "outlier_ratio": float(primary["per_camera_plane"][name]["outlier_ratio"]),
        }
        for name in ("camera_a", "camera_b")
    }


def _concurrent_fusion(path: Path, *, valid_key: str) -> dict[str, dict[str, float]]:
    report = _load(path)
    _validate_formal(report, path)
    return {
        name: {
            "valid_ratio": float(report["stage_summary"][name][valid_key]),
            "board_median_abs_z_m": float(report["board_summary"][name]["median_abs_z_m"]),
            "board_p95_m": float(report["board_summary"][name]["p95_abs_z_m"]),
            "board_rmse_m": float(report["board_summary"][name]["rmse_m"]),
            "outlier_ratio": float(report["board_summary"][name]["outlier_ratio"]),
        }
        for name in ("camera_a", "camera_b")
    }


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _validate_formal(report: dict[str, Any], path: Path) -> None:
    if report.get("schema_version") != "pointcloud-builder.real-multicamera-fusion.v1":
        raise ValueError(f"{path} has an unexpected formal fusion schema")
    matched_sets = int(report.get("matched_sets", 0))
    if matched_sets < 300:
        raise ValueError(f"{path} has fewer than 300 matched sets")
    if report.get("snapshot_only") is not True or report.get("persistent_mapping") is not False:
        raise ValueError(f"{path} is not a snapshot-only report")
    if report.get("passed") is not True:
        raise ValueError(f"{path} did not pass its formal gates")
    if set(report.get("stage_summary", {})) != {"camera_a", "camera_b"}:
        raise ValueError(f"{path} does not contain both camera routes")
    if any(
        item.get("depth_mode") != "ffs_stereo"
        for item in report["stage_summary"].values()
    ):
        raise ValueError(f"{path} is not an FFS formal route")
    snapshots = report.get("evidence_snapshots", [])
    expected_indices = sorted({round(index * (matched_sets - 1) / 4) for index in range(5)})
    if report.get("evidence_snapshot_count") != 5 or [
        int(item.get("frame_index", -1)) for item in snapshots
    ] != expected_indices:
        raise ValueError(f"{path} does not contain five distributed evidence snapshots")
    required_gates = {
        "matcher_integrity",
        "per_camera_board",
        "fused_board_shift",
        "cross_camera_overlap",
        "cube_evidence_snapshots",
        "cube_same_physical_candidate",
        "cube_joint_camera_observation",
        "cube_fused_degradation",
        "fusion_contribution",
        "fusion_thickness",
        "global_sampling",
        "no_per_camera_early_sampling",
        "processing_order",
        "deterministic_voxel_fusion",
        "tensorrt_plugin",
        "worker_cleanup",
    }
    gates = report.get("gates", {})
    if set(gates) != required_gates or not all(gates.values()):
        raise ValueError(f"{path} has missing or failed formal gates")


def _validate_m7_ffs(report: dict[str, Any], path: Path) -> dict[str, Any]:
    if report.get("schema_version") != "pointcloud-builder.live-rig-acceptance.v1":
        raise ValueError(f"{path} has an unexpected M7 schema")
    if report.get("depth_modes") != {
        "camera_a": "ffs_stereo",
        "camera_b": "ffs_stereo",
    }:
        raise ValueError(f"{path} is not the M7 dual-camera FFS report")
    gates = report.get("gates", {})
    latency_p95 = float(report["runs"][0]["timing_ms"]["total_ms"]["p95"])
    processed_fps = float(report["processed_fps"])
    passed = bool(
        report.get("passed")
        and gates.get("ffs_processed_fps")
        and gates.get("ffs_end_to_end_p95")
        and processed_fps >= 15.0
        and latency_p95 <= 66.8
    )
    return {
        "report_schema": report["schema_version"],
        "processed_fps": processed_fps,
        "end_to_end_p95_ms": latency_p95,
        "minimum_fps": 15.0,
        "maximum_p95_ms": 66.8,
        "passed": passed,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
