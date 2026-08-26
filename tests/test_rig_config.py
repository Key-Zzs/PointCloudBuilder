from __future__ import annotations

from copy import deepcopy

import pytest

from pointcloud_builder.rig import parse_rig_config


def _raw(names: tuple[str, ...] = ("camera_a", "camera_b")) -> dict:
    return {
        "schema_version": "pointcloud-builder.rig.v1",
        "output_frame": "workspace",
        "cameras": [
            {
                "name": name,
                "enabled": True,
                "source": {
                    "type": "synthetic",
                    "capture_artifact": f"synthetic://{name}",
                    "provision_artifact": f"synthetic://{name}/bundle",
                },
                "depth": {"mode": "native"},
                "pipeline_config": None,
                "local_crop": {"enabled": False},
            }
            for name in names
        ],
        "timing": {"mode": "exact_index", "maximum_skew_ms": 33.4},
        "workspace_crop": {
            "enabled": True,
            "x": [-0.75, 0.75],
            "y": [-0.6, 0.6],
            "z": [-0.01, 0.6],
        },
        "fusion": {"enabled": False},
        "sampling": {
            "enabled": True,
            "mode": "voxel_fps",
            "num_points": 1024,
            "voxel_size": 0.01,
            "deterministic": True,
            "seed": 7,
        },
    }


def test_rig_config_is_strict_and_versioned() -> None:
    config = parse_rig_config(_raw())
    assert config.schema_version == "pointcloud-builder.rig.v1"
    assert [camera.name for camera in config.enabled_cameras] == ["camera_a", "camera_b"]
    assert config.output_frame == config.workspace_crop.frame == "workspace"


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda raw: raw.update({"unknown": 1}), "unknown fields"),
        (lambda raw: raw["cameras"][0].update({"unknown": 1}), "unknown fields"),
        (lambda raw: raw.update({"cameras": []}), "non-empty"),
        (lambda raw: raw["cameras"].append(deepcopy(raw["cameras"][0])), "duplicate"),
        (lambda raw: raw["fusion"].update({"enabled": True}), "M6"),
    ],
)
def test_rig_config_rejects_invalid_contracts(mutator, message: str) -> None:
    raw = _raw()
    mutator(raw)
    with pytest.raises(ValueError, match=message):
        parse_rig_config(raw)
