#!/usr/bin/env python3
"""Re-extract point and mesh geometry from a validated TSDF volume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pointcloud_builder.mapping.artifact import (
    load_tsdf_map_artifact,
    write_extracted_geometry,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = _private_output(args.output)
    mapper = load_tsdf_map_artifact(args.map)
    try:
        extraction = mapper.extract()
        write_extracted_geometry(output, extraction)
    finally:
        mapper.close()
    print(
        json.dumps(
            {
                "schema_version": "pointcloud-builder.tsdf-extract-cli.v1",
                "published": True,
                "point_count": extraction.point_count,
                "triangle_count": extraction.triangle_count,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _private_output(value: str) -> Path:
    output = Path(value).resolve()
    if not output.is_relative_to((Path.cwd() / ".local").resolve()):
        raise ValueError("extracted real geometry must be written under .local/")
    return output


if __name__ == "__main__":
    main()
