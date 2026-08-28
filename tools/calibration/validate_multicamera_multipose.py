#!/usr/bin/env python3
"""Validate a candidate rig solution against solve and holdout observations."""

from __future__ import annotations

import argparse
import json

from pointcloud_builder.local_paths import require_repo_local_path
from pointcloud_builder.rig_calibration import load_observations, load_solution
from pointcloud_builder.rig_calibration.validation import (
    validate_rig_calibration_solution,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    solution_path = require_repo_local_path(args.solution, label="candidate rig solution")
    observation_path = require_repo_local_path(
        args.observations, label="real rig observations"
    )
    output = require_repo_local_path(args.output, label="rig validation report")
    report = validate_rig_calibration_solution(
        load_solution(solution_path), load_observations(observation_path)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"RIG_CALIBRATION_VALIDATION={report['status']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
