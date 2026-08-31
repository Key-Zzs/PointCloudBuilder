#!/usr/bin/env python3
"""Bind immutable current A/B diagnostic reports into formal acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pointcloud_builder.rig_calibration.physical_acceptance import (
    summarize_legacy_ab_diagnostic,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--diagnostic-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"physical acceptance output already exists: {output}")
    artifact = summarize_legacy_ab_diagnostic(
        args.solution, args.validation, args.diagnostic_root
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema_version": artifact["schema_version"],
                "status": artifact["status"],
                "solution_fingerprint": artifact["solution_fingerprint"],
                "camera_set": artifact["camera_set"],
                "diagnostic_residual_writeback": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
