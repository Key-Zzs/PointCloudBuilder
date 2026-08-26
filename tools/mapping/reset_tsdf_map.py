#!/usr/bin/env python3
"""Publish a recoverable empty revision derived from a validated TSDF map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pointcloud_builder.mapping.artifact import (
    load_tsdf_map_artifact,
    write_tsdf_map_artifact,
)
from pointcloud_builder.mapping.validation import load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = _private_output(args.output)
    source = load_json(Path(args.map) / "source_recording.json")
    mapper = load_tsdf_map_artifact(args.map)
    try:
        mapper.unfreeze()
        mapper.reset()
        state = mapper.freeze()
        write_tsdf_map_artifact(
            output,
            mapper=mapper,
            source_recording_sha256=source["recording_manifest_sha256"],
            integration_metrics={
                "reset_from_validated_map": True,
                "new_map_revision": state.map_revision,
            },
        )
    finally:
        mapper.close()
    print(
        json.dumps(
            {
                "schema_version": "pointcloud-builder.tsdf-reset-cli.v1",
                "published": True,
                "active_blocks": mapper.state.active_block_count,
                "active_weight_voxels": 0,
                "native_empty_sentinel": True,
                "map_revision": state.map_revision,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _private_output(value: str) -> Path:
    output = Path(value).resolve()
    if not output.is_relative_to((Path.cwd() / ".local").resolve()):
        raise ValueError("reset real maps must be written under .local/")
    return output


if __name__ == "__main__":
    main()
