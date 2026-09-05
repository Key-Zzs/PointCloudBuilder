#!/usr/bin/env python3
"""Evaluate factory K/D health on frozen solve/holdout rig observations."""

from __future__ import annotations

import argparse
import hashlib

from pointcloud_builder.local_paths import require_repo_local_path
from pointcloud_builder.rig_calibration.artifact import load_observations
from pointcloud_builder.rig_calibration.intrinsic_health import (
    evaluate_rig_intrinsic_health,
    write_rig_intrinsic_health,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    observations_path = require_repo_local_path(
        args.observations, label="rig observations"
    )
    observations = load_observations(observations_path)
    output = require_repo_local_path(args.output, label="intrinsic health report")
    if output.exists():
        raise FileExistsError(f"intrinsic-health output already exists: {output}")
    report = evaluate_rig_intrinsic_health(
        observations,
        observations_sha256=hashlib.sha256(observations_path.read_bytes()).hexdigest(),
    )
    write_rig_intrinsic_health(report, output)
    print(
        f"INTRINSIC_HEALTH={report['status']}; "
        f"CAMERAS={','.join(report['camera_set'])}; "
        "FACTORY_INTRINSICS_MUTATED=NO"
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
