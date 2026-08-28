#!/usr/bin/env python3
"""Fail closed when a static capture is mislabeled as multi-pose calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pointcloud_builder.local_paths import require_repo_local_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-pose-count", type=int, default=6)
    args = parser.parse_args()
    source = require_repo_local_path(
        args.capture_manifest, label="real capture manifest"
    )
    output = require_repo_local_path(args.output, label="real preflight report")
    raw = json.loads(source.read_text(encoding="utf-8"))
    matched_sets = int(raw["matched_sets"])
    if matched_sets <= 0:
        raise ValueError("capture manifest has no matched sets")
    # The source contract says the scene and target stayed fixed.  Repeated
    # frames therefore remain one pose group; frame indices are not poses.
    report = {
        "schema_version": "pointcloud-builder.static-multipose-preflight.v1",
        "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "observation_frame_count": matched_sets,
        "pose_group_count": 1,
        "pose_grouping": "all frames -> static_pose_0",
        "minimum_required_pose_count": args.min_pose_count,
        "MULTIPOSE_PREFLIGHT": "INSUFFICIENT_POSE_DIVERSITY",
        "passed": False,
        "reason": "repeated observations of one stationary board pose are not multi-pose evidence",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("MULTIPOSE_PREFLIGHT=INSUFFICIENT_POSE_DIVERSITY; expected fail-closed result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
