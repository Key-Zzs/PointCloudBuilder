from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from pointcloud_builder.mapping.config import load_tsdf_config
from pointcloud_builder.rig import load_rig_config

ROOT = Path(__file__).parents[1]
SCRIPT_PATH = ROOT / "scripts/prepare_live_reconstruction_configs.py"
SPEC = importlib.util.spec_from_file_location(
    "prepare_live_reconstruction_configs", SCRIPT_PATH
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _mapping() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/mapping/native_workspace_example.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_load_camera_names_uses_only_logical_identity(tmp_path: Path) -> None:
    path = tmp_path / "identity.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": MODULE.IDENTITY_SCHEMA,
                "camera_c": {"serial": "C"},
                "camera_a": {"serial": "A"},
                "camera_b": {"serial": "B"},
            }
        ),
        encoding="utf-8",
    )
    assert MODULE.load_camera_names(path, 3) == ["camera_a", "camera_b", "camera_c"]


def test_render_configs_creates_fixed_provision_three_camera_profiles(
    tmp_path: Path,
) -> None:
    output = tmp_path / ".local/configs"
    mapping = _mapping()
    configs = MODULE.render_configs(
        ["camera_a", "camera_b", "camera_c"],
        camera_rig_root=tmp_path / ".local/camera_rig",
        ffs_config=output / "ffs_tensorrt_plugin_rgb.yaml",
        output_dir=output,
        workspace_frame="workspace",
        mapping=mapping,
    )

    assert set(configs) == set(MODULE.OUTPUT_NAMES)
    assert configs["mapping_acceptance.yaml"] == mapping
    assert configs["tsdf_frozen_ffs.yaml"]["dynamic"]["mode"] == "frozen_static"
    for name in MODULE.PROFILE_TEMPLATES:
        config = configs[name]
        assert "rig_calibration" not in config
        assert [item["name"] for item in config["cameras"]] == [
            "camera_a",
            "camera_b",
            "camera_c",
        ]
        assert all(item["depth"] == {"mode": "ffs_stereo"} for item in config["cameras"])
        assert all(item["pointcloud"] == {"use_rgb": True} for item in config["cameras"])
        assert config["cameras"][0]["source"]["camera_config"] == (
            "../camera_rig/camera_a/configs/runtime.yaml"
        )
        assert config["cameras"][0]["pipeline_config"] == (
            "ffs_tensorrt_plugin_rgb.yaml"
        )


def test_write_configs_preserves_workspace_mapping_and_refuses_other_drift(
    tmp_path: Path,
) -> None:
    output = tmp_path / ".local/configs"
    configs = MODULE.render_configs(
        ["camera_a", "camera_b", "camera_c"],
        camera_rig_root=tmp_path / ".local/camera_rig",
        ffs_config=output / "ffs_tensorrt_plugin_rgb.yaml",
        output_dir=output,
        workspace_frame="workspace",
        mapping=_mapping(),
    )
    statuses = MODULE.write_configs(configs, output_dir=output, force=False)
    assert all(status == "written" for status in statuses.values())
    for name in MODULE.PROFILE_TEMPLATES:
        assert len(load_rig_config(output / name).cameras) == 3
    assert load_tsdf_config(output / "tsdf_frozen_ffs.yaml").dynamic.mode == (
        "frozen_static"
    )

    custom = yaml.safe_load((output / "mapping.yaml").read_text(encoding="utf-8"))
    custom["workspace_crop"]["x"] = [-0.5, 0.5]
    (output / "mapping.yaml").write_text(yaml.safe_dump(custom), encoding="utf-8")
    statuses = MODULE.write_configs(configs, output_dir=output, force=False)
    assert statuses["mapping.yaml"] == "preserved_existing_workspace_mapping"

    raw = output / "live_rig_ffs_rgb_raw.yaml"
    changed = yaml.safe_load(raw.read_text(encoding="utf-8"))
    changed["timing"]["maximum_skew_ms"] = 99
    raw.write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(FileExistsError, match="different content"):
        MODULE.write_configs(configs, output_dir=output, force=False)
