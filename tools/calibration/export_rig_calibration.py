#!/usr/bin/env python3
"""Explicitly export per-camera fixed-mount candidates from a passed solution."""

from __future__ import annotations

import argparse
import json

from pointcloud_builder.local_paths import require_repo_local_path
from pointcloud_builder.rig_calibration import load_solution
from pointcloud_builder.rig_calibration.export import export_fixed_mount_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    solution_path = require_repo_local_path(args.solution, label="candidate rig solution")
    validation_path = require_repo_local_path(
        args.validation, label="rig validation report"
    )
    output_root = require_repo_local_path(
        args.output_root, label="fixed-mount candidate output root"
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    written = export_fixed_mount_candidates(
        load_solution(solution_path), output_root, validation_report=validation
    )
    print(f"EXPORTED_CANDIDATES={len(written)}; production calibration unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
