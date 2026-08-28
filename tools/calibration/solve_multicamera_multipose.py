#!/usr/bin/env python3
"""Solve PCB-owned N-camera, multi-pose candidate calibration."""

from __future__ import annotations

import argparse

from pointcloud_builder.rig_calibration import (
    load_observations,
    solve_rig_calibration,
    write_solution,
)
from pointcloud_builder.rig_calibration.config import load_rig_calibration_config
from pointcloud_builder.local_paths import require_repo_local_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    observation_path = require_repo_local_path(
        args.observations, label="real rig observations"
    )
    output_path = require_repo_local_path(args.output, label="candidate rig solution")
    solution = solve_rig_calibration(
        load_observations(observation_path),
        load_rig_calibration_config(args.config),
    )
    write_solution(solution, output_path)
    print(f"RIG_CALIBRATION={solution.validation['status']}; candidate-only output written")
    return 0 if solution.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
