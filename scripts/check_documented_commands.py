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
    "scripts/prepare_ffs_assets.py",
    "configs/mapping/dense_rgb_reconstruction_example.yaml",
    "configs/mapping/compact_rgb_reconstruction_example.yaml",
    "configs/mapping/raw_rgb_concatenation_example.yaml",
    "tools/mapping/run_live_tsdf_mapping.py",
    "tools/mapping/record_live_rig_depth.py",
    "tools/mapping/build_tsdf_offline.py",
    "tools/mapping/benchmark_fusion_voxels.py",
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
