#!/usr/bin/env python3
"""Fail-closed validation and native-volume extraction smoke for a TSDF map."""

from __future__ import annotations

import argparse
import json

from pointcloud_builder.mapping.artifact import (
    load_tsdf_map_artifact,
    validate_tsdf_map_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True)
    args = parser.parse_args()
    manifest = validate_tsdf_map_artifact(args.map)
    mapper = load_tsdf_map_artifact(args.map)
    try:
        extraction = mapper.extract()
        result = {
            "schema_version": "pointcloud-builder.tsdf-validation-cli.v1",
            "valid": True,
            "workspace_frame": manifest["workspace_frame"],
            "active_blocks": mapper.state.active_block_count,
            "point_count": extraction.point_count,
            "triangle_count": extraction.triangle_count,
        }
    finally:
        mapper.close()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
