#!/usr/bin/env python3
"""Validate a new target capture against an existing fixed transform.

No pose is solved or refined.  CameraRig v1.0.0 projection/overlay helpers are
used only to evaluate the already-provisioned transform.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import numpy as np
from camera_rig import __version__ as camera_rig_version
from camera_rig.api import RigidTransform, load_provisioned_camera_bundle
from camera_rig.calibration.fixed.overlays import write_fixed_pose_overlay
from camera_rig.calibration.pose import project_points_px
from camera_rig.targets.observation import TargetObservation

from pointcloud_builder.integrations.camera_rig.calibration_adapter import (
    resolve_bundle_transform,
)


def check_sentinel(
    *,
    provision: Path,
    capture: Path,
    target_report: Path,
    output_report: Path,
    overlays: Path,
) -> dict:
    if camera_rig_version != "1.0.0":
        raise RuntimeError("fixed-pose sentinel is pinned to CameraRig v1.0.0")
    bundle = load_provisioned_camera_bundle(provision)
    color = bundle.intrinsics["color"]
    T_color_from_workspace = resolve_bundle_transform(
        bundle, "workspace", color.frame
    )
    raw = json.loads(target_report.read_text(encoding="utf-8"))
    target = json.loads(
        (provision / "target/target_spec.json").read_text(encoding="utf-8")
    )
    per_frame = []
    observations: dict[int, TargetObservation] = {}
    all_residuals = []
    for item in raw["per_frame"]:
        if not item.get("success") or not isinstance(item.get("observation"), dict):
            continue
        index = int(item["frame_index"])
        observation = TargetObservation.from_dict(item["observation"])
        pose = RigidTransform(
            source_frame=observation.target_frame,
            target_frame=color.frame,
            matrix=T_color_from_workspace.matrix,
        )
        projected = project_points_px(observation.object_points_m, pose, color)
        residuals = np.linalg.norm(projected - observation.image_points_px, axis=1)
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        p95 = float(np.percentile(residuals, 95))
        accepted = rmse <= 0.50 and p95 <= 1.00
        per_frame.append(
            {
                "frame_index": index,
                "point_count": len(residuals),
                "rmse_px": rmse,
                "median_px": float(np.median(residuals)),
                "p95_px": p95,
                "maximum_px": float(np.max(residuals)),
                "accepted": accepted,
            }
        )
        observations[index] = observation
        all_residuals.extend(float(value) for value in residuals)
    accepted = [item for item in per_frame if item["accepted"]]
    ratio = len(accepted) / 60.0
    passed = len(per_frame) == 60 and len(accepted) >= 50 and ratio >= 0.90
    ranked = sorted(accepted, key=lambda item: (item["rmse_px"], item["frame_index"]))
    selected = (
        {
            "best": ranked[0],
            "median_quality": ranked[len(ranked) // 2],
            "worst_accepted": ranked[-1],
        }
        if ranked
        else {}
    )
    overlay_paths = {}
    for label, item in selected.items():
        index = int(item["frame_index"])
        with np.load(capture / f"frames/frame_{index:06d}.npz") as arrays:
            image = np.asarray(arrays["color"], dtype=np.uint8)
        output = overlays / f"{label}_frame_{index:06d}.png"
        write_fixed_pose_overlay(
            output,
            image_rgb=image,
            observation=observations[index],
            T_camera_from_target=RigidTransform(
                source_frame=observations[index].target_frame,
                target_frame=color.frame,
                matrix=T_color_from_workspace.matrix,
            ),
            intrinsics=color,
            board_width_m=float(target["board_width_m"]),
            board_height_m=float(target["board_height_m"]),
        )
        overlay_paths[label] = str(output)
    report = {
        "schema_version": "pointcloud-builder.fixed-pose-sentinel.v1",
        "camera_rig_version": camera_rig_version,
        "camera_name": bundle.device.camera_name,
        "frames_evaluated": len(per_frame),
        "accepted_frames": len(accepted),
        "accepted_ratio": ratio,
        "thresholds": {
            "required_frames": 60,
            "minimum_accepted_frames": 50,
            "minimum_accepted_ratio": 0.90,
            "maximum_frame_rmse_px": 0.50,
            "maximum_frame_p95_px": 1.00,
        },
        "aggregate": {
            "median_frame_rmse_px": (
                statistics.median(item["rmse_px"] for item in per_frame)
                if per_frame
                else None
            ),
            "p95_residual_px": (
                float(np.percentile(all_residuals, 95)) if all_residuals else None
            ),
            "maximum_residual_px": max(all_residuals) if all_residuals else None,
        },
        "selected_overlays": overlay_paths,
        "per_frame": per_frame,
        "pose_optimization_performed": False,
        "status": "PASS" if passed else "FAIL",
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provision", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--target-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overlays", type=Path, required=True)
    args = parser.parse_args()
    report = check_sentinel(
        provision=args.provision,
        capture=args.capture,
        target_report=args.target_report,
        output_report=args.report,
        overlays=args.overlays,
    )
    print(
        json.dumps(
            {
                "camera_name": report["camera_name"],
                "frames_evaluated": report["frames_evaluated"],
                "accepted_frames": report["accepted_frames"],
                "accepted_ratio": report["accepted_ratio"],
                "aggregate": report["aggregate"],
                "pose_optimization_performed": False,
                "status": report["status"],
            },
            indent=2,
        )
    )
    if report["status"] != "PASS":
        raise SystemExit("fixed-pose sentinel failed")


if __name__ == "__main__":
    main()
