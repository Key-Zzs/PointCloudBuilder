#!/usr/bin/env python3
"""Promote a passed candidate and physical acceptance to production."""

from __future__ import annotations

import argparse
import json

from pointcloud_builder.rig_calibration.deployment import promote_rig_calibration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--physical-acceptance", required=True)
    parser.add_argument("--intrinsic-health", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    artifact = promote_rig_calibration(
        args.solution,
        args.validation,
        args.physical_acceptance,
        args.output,
        intrinsic_health_path=args.intrinsic_health,
    )
    print(
        json.dumps(
            {
                "schema_version": artifact["schema_version"],
                "status": artifact["status"],
                "rig_calibration_fingerprint": artifact["rig_calibration_fingerprint"],
                "camera_set": sorted(artifact["cameras"]),
                "production_applied": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
