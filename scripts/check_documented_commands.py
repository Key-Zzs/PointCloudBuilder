#!/usr/bin/env python3
"""Validate documented reconstruction entry points without running hardware."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READMES = (ROOT / "README.md", ROOT / "README_zh-CN.md")
REQUIRED_PATHS = (
    "environment.reconstruction.yml",
    "scripts/bootstrap_reconstruction_env.sh",
    "scripts/doctor_reconstruction_env.py",
    "scripts/prepare_camera_rig_calibration.py",
    "scripts/prepare_ffs_assets.py",
    "scripts/prepare_ffs_pipeline_configs.py",
    "configs/mapping/dense_rgb_reconstruction_example.yaml",
    "configs/mapping/compact_rgb_reconstruction_example.yaml",
    "configs/mapping/raw_rgb_concatenation_example.yaml",
    "configs/mapping/live_rig_three_camera_example.yaml",
    "configs/calibration/charuco_500x700_existing_board.yaml",
    "configs/calibration/ncamera_physical_acceptance_strict_example.yaml",
    "tools/calibration/promote_rig_calibration.py",
    "tools/calibration/evaluate_ncamera_rig_alignment.py",
    "tools/mapping/run_live_reconstruction_profile.py",
    "tools/mapping/run_live_tsdf_mapping.py",
    "tools/mapping/record_live_rig_depth.py",
    "tools/mapping/build_tsdf_offline.py",
    "tools/mapping/extract_tsdf_geometry.py",
    "tools/mapping/validate_tsdf_map.py",
    "tools/mapping/benchmark_fusion_voxels.py",
    "tools/mapping/benchmark_world_reconstruction.py",
)
INTERACTIVE_FLAGS = (
    "--interactive",
    "--rig-config",
    "--tsdf-config",
    "--initial-map",
    "--rerun-connect",
    "--rerun-record",
    "--viewer-point-budget",
)
LIVE_RIG_FLAGS = ("--acceptance-scope",)
PROFILE_FLAGS = ("--profile", "--matched-sets", "--viewer", "--rerun-record")
BENCHMARK_FLAGS = ("--input-mode", "--frames", "--warmup", "--report")
DOCTOR_FLAGS = ("--no-hardware", "--expected-d435i-count", "--asset-root")
USB_FLAGS = ("--identity-map", "--expected-count", "--report")
CAMERA_RIG_PREPARATION_FLAGS = (
    "--identity-map",
    "--target",
    "--asset-root",
    "--expected-camera-count",
    "--runtime-only",
    "--workspace-equals-target",
    "--update-existing",
    "--check",
    "--report",
)
FFS_PIPELINE_CONFIG_FLAGS = (
    "--asset-root",
    "--output-dir",
    "--camera-name",
    "--force",
)
PROMOTION_FLAGS = (
    "--solution",
    "--validation",
    "--physical-acceptance",
    "--output",
)
NCAMERA_FLAGS = (
    "--rig-config",
    "--rig-calibration",
    "--candidate-solution",
    "--candidate-validation",
    "--thresholds",
    "--recording",
    "--mapping-config",
    "--matched-sets",
    "--declared-no-overlap",
    "--output",
)


def main() -> int:
    checks = {}
    for relative in REQUIRED_PATHS:
        exists = (ROOT / relative).is_file()
        checks[f"path:{relative}"] = exists
    help_env = os.environ.copy()
    help_env["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT), str(ROOT / "third_party/CameraRig/src"))
    )
    help_result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/mapping/run_live_tsdf_mapping.py"),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=help_env,
    )
    checks["interactive_help_exit"] = help_result.returncode == 0
    for flag in INTERACTIVE_FLAGS:
        checks[f"interactive_flag:{flag}"] = flag in help_result.stdout
    live_rig_help = subprocess.run(
        [sys.executable, str(ROOT / "tools/mapping/run_live_rig.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=help_env,
    )
    checks["live_rig_help_exit"] = live_rig_help.returncode == 0
    for flag in LIVE_RIG_FLAGS:
        checks[f"live_rig_flag:{flag}"] = flag in live_rig_help.stdout
    for name, relative, flags in (
        (
            "live_reconstruction_profile",
            "tools/mapping/run_live_reconstruction_profile.py",
            PROFILE_FLAGS,
        ),
        (
            "world_reconstruction_benchmark",
            "tools/mapping/benchmark_world_reconstruction.py",
            BENCHMARK_FLAGS,
        ),
        ("doctor", "scripts/doctor_reconstruction_env.py", DOCTOR_FLAGS),
        (
            "camera_rig_preparation",
            "scripts/prepare_camera_rig_calibration.py",
            CAMERA_RIG_PREPARATION_FLAGS,
        ),
        (
            "ffs_pipeline_configs",
            "scripts/prepare_ffs_pipeline_configs.py",
            FFS_PIPELINE_CONFIG_FLAGS,
        ),
        ("usb_topology", "tools/mapping/check_usb_topology.py", USB_FLAGS),
        (
            "rig_calibration_promotion",
            "tools/calibration/promote_rig_calibration.py",
            PROMOTION_FLAGS,
        ),
        (
            "ncamera_acceptance",
            "tools/calibration/evaluate_ncamera_rig_alignment.py",
            NCAMERA_FLAGS,
        ),
    ):
        result = subprocess.run(
            [sys.executable, str(ROOT / relative), "--help"],
            check=False,
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=help_env,
        )
        checks[f"{name}_help_exit"] = result.returncode == 0
        for flag in flags:
            checks[f"{name}_flag:{flag}"] = flag in result.stdout
    for readme in READMES:
        text = readme.read_text(encoding="utf-8")
        checks[f"{readme.name}:policy:pose_validated"] = (
            "target.detection_policy: pose_validated" in text
        )
        for relative in REQUIRED_PATHS:
            checks[f"{readme.name}:reference:{relative}"] = relative in text
        for flag in INTERACTIVE_FLAGS:
            checks[f"{readme.name}:flag:{flag}"] = flag in text
        for flag in LIVE_RIG_FLAGS:
            checks[f"{readme.name}:flag:{flag}"] = flag in text
        for flag in (
            PROFILE_FLAGS
            + BENCHMARK_FLAGS
            + DOCTOR_FLAGS
            + CAMERA_RIG_PREPARATION_FLAGS
            + FFS_PIPELINE_CONFIG_FLAGS
            + USB_FLAGS
            + PROMOTION_FLAGS
            + NCAMERA_FLAGS
        ):
            checks[f"{readme.name}:flag:{flag}"] = flag in text
        checks[f"{readme.name}:board:dictionary"] = "DICT_4X4_50" in text
        checks[f"{readme.name}:board:size"] = "500 x 700" in text
        checks[f"{readme.name}:board:layout"] = "5 x 7" in text
        checks[f"{readme.name}:board:square"] = "100 mm" in text
        checks[f"{readme.name}:board:marker"] = "75 mm" in text
        for command_path in re.findall(
            r"(?:python\s+|\./)([A-Za-z0-9_./-]+\.(?:py|sh))", text
        ):
            checks[f"{readme.name}:command:{command_path}"] = (
                ROOT / command_path
            ).is_file()
    report = {
        "schema_version": "pointcloud-builder.documented-command-check.v1",
        "checks": checks,
        "passed": all(checks.values()),
        "failed": sorted(name for name, passed in checks.items() if not passed),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
