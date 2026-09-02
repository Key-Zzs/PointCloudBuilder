from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

from pointcloud_builder.config import load_config

ROOT = Path(__file__).parents[1]
SCRIPT_PATH = ROOT / "scripts/prepare_ffs_pipeline_configs.py"
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "prepare_ffs_pipeline_configs", SCRIPT_PATH
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _template() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/mapping/ffs_workspace_example.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_render_configs_binds_every_checked_route(tmp_path: Path) -> None:
    output = tmp_path / ".local/configs"
    asset_root = tmp_path / ".local/ffs"
    configs = MODULE.render_configs(
        _template(),
        asset_root=asset_root,
        output_dir=output,
        camera_name="camera_a",
    )

    assert set(configs) == set(MODULE.OUTPUT_NAMES)
    assert configs["ffs_pytorch.yaml"]["depth_source"]["ffs"]["checkpoint_path"] == (
        "../ffs/artifacts/model_best_bp2_serialize.pth"
    )
    assert configs["ffs_tensorrt_single.yaml"]["depth_source"]["ffs"][
        "engine_path"
    ] == "../ffs/artifacts/tensorrt_single_fp16_o3.engine"
    two_stage = configs["ffs_tensorrt_two_stage.yaml"]["depth_source"]["ffs"]
    assert two_stage["feature_engine_path"].endswith(
        "tensorrt_two_stage_feature_fp16_o3.engine"
    )
    assert two_stage["post_engine_path"].endswith(
        "tensorrt_two_stage_post_fp16_o3.engine"
    )
    plugin = configs["ffs_tensorrt_plugin.yaml"]
    assert plugin["pointcloud"] == {
        "use_rgb": False,
        "output_format": "xyz",
    }
    assert plugin["depth_source"]["ffs"]["plugin_library_path"] == (
        "../ffs/build/libffs_gwc_plugin.so"
    )
    assert configs["ffs_tensorrt_plugin_rgb.yaml"]["pointcloud"] == {
        "use_rgb": True,
        "output_format": "xyzrgb",
    }


def test_write_configs_validates_and_overwrites_existing_files(
    tmp_path: Path,
) -> None:
    output = tmp_path / ".local/configs"
    configs = MODULE.render_configs(
        _template(),
        asset_root=tmp_path / ".local/ffs",
        output_dir=output,
        camera_name="camera_a",
    )

    paths = MODULE.write_configs(configs, output_dir=output)
    assert [path.name for path in paths] == list(MODULE.OUTPUT_NAMES)
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in paths)
    for path in paths:
        assert load_config(path).depth_source.ffs is not None

    changed = yaml.safe_load(paths[0].read_text(encoding="utf-8"))
    changed["camera"]["name"] = "stale_name"
    paths[0].write_text(yaml.safe_dump(changed), encoding="utf-8")
    MODULE.write_configs(configs, output_dir=output)
    assert yaml.safe_load(paths[0].read_text(encoding="utf-8"))["camera"]["name"] == "camera_a"
