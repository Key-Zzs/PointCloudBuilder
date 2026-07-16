from __future__ import annotations

import pytest

from pointcloud_builder.config import load_config, parse_config


def _camera(*, aligned: bool = False) -> dict:
    intrinsics = {"width": 640, "height": 480, "fx": 100.0, "fy": 100.0, "cx": 319.5, "cy": 239.5}
    return {
        "name": "head",
        "depth_scale": 0.001,
        "aligned_depth_to_color": aligned,
        "color_intrinsics": intrinsics,
        "depth_intrinsics": intrinsics,
        "depth_to_color_extrinsics": {
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation": [0, 0, 0],
        },
    }


def test_legacy_config_defaults_to_frame_mode() -> None:
    config = parse_config({"camera": _camera(), "pointcloud": {"use_rgb": False, "output_format": "xyz"}})
    assert config.depth_source.mode == "frame"
    assert config.depth_source.ffs is None


def test_ffs_config_is_strict_and_explicit() -> None:
    config = parse_config(
        {
            "camera": _camera(),
            "pointcloud": {"use_rgb": False, "output_format": "xyz"},
            "depth_source": {
                "mode": "ffs_stereo",
                "ffs": {
                    "backend": "tensorrt_single",
                    "engine_path": "model.engine",
                    "manifest_path": "model.manifest.json",
                    "max_disp": 192,
                    "valid_iters": 8,
                    "precision": "fp16",
                    "baseline_m": 0.05,
                },
            },
        }
    )
    assert config.depth_source.mode == "ffs_stereo"
    assert config.depth_source.ffs is not None
    assert config.depth_source.ffs.backend == "tensorrt_single"
    assert config.depth_source.ffs.max_disp == 192
    assert config.depth_source.ffs.width == 640
    assert config.depth_source.ffs.height == 480


def test_ffs_rejects_non_v05_shape() -> None:
    with pytest.raises(ValueError, match="480.*640"):
        parse_config(
            {
                "camera": _camera(),
                "pointcloud": {"use_rgb": False, "output_format": "xyz"},
                "depth_source": {"mode": "ffs_stereo", "ffs": {"backend": "pytorch", "height": 448, "width": 640, "baseline_m": 0.05}},
            }
        )


def test_ffs_rejects_aligned_depth_camera_mode() -> None:
    with pytest.raises(ValueError, match="aligned_depth_to_color=false"):
        parse_config(
            {
                "camera": _camera(aligned=True),
                "pointcloud": {"use_rgb": False, "output_format": "xyz"},
                "depth_source": {"mode": "ffs_stereo", "ffs": {"backend": "pytorch", "baseline_m": 0.05}},
            }
        )


def test_ffs_builder_precision_and_resource_contract() -> None:
    config = parse_config(
        {
            "camera": _camera(),
            "pointcloud": {"use_rgb": False, "output_format": "xyz"},
            "depth_source": {
                "mode": "ffs_stereo",
                "ffs": {
                    "backend": "tensorrt_single",
                    "precision": "fp32",
                    "builder_optimization_level": 0,
                    "workspace_gib": 2.5,
                    "config_path": "route.yaml",
                    "artifact_id": "fp32_o0",
                    "baseline_m": 0.05,
                },
            },
        }
    )
    ffs = config.depth_source.ffs
    assert ffs is not None
    assert ffs.precision == "fp32"
    assert ffs.builder_optimization_level == 0
    assert ffs.workspace_gib == 2.5
    assert ffs.config_path == "route.yaml"
    assert ffs.artifact_id == "fp32_o0"


@pytest.mark.parametrize("key,value", [("builder_optimization_level", -1), ("builder_optimization_level", 6), ("workspace_gib", 0)])
def test_ffs_rejects_invalid_builder_resource_contract(key: str, value: float) -> None:
    with pytest.raises(ValueError):
        parse_config(
            {
                "camera": _camera(),
                "pointcloud": {"use_rgb": False, "output_format": "xyz"},
                "depth_source": {
                    "mode": "ffs_stereo",
                    "ffs": {"backend": "tensorrt_single", key: value, "baseline_m": 0.05},
                },
            }
        )


def test_ffs_asset_paths_are_relative_to_declaring_yaml(tmp_path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "builder.yaml"
    config_path.write_text(
        """
device: cpu
camera:
  name: head
  depth_scale: 0.001
  aligned_depth_to_color: false
  color_intrinsics: {width: 640, height: 480, fx: 100, fy: 100, cx: 319.5, cy: 239.5}
  depth_intrinsics: {width: 640, height: 480, fx: 100, fy: 100, cx: 319.5, cy: 239.5}
pointcloud: {use_rgb: false, output_format: xyz}
depth_source:
  mode: ffs_stereo
  ffs:
    backend: pytorch
    checkpoint_path: ../artifacts/model.pth
    model_config_path: ../artifacts/cfg.yaml
    calibration_path: ../calibration.json
    baseline_m: 0.05
""",
        encoding="utf-8",
    )
    ffs = load_config(config_path).depth_source.ffs
    assert ffs is not None
    assert ffs.checkpoint_path == str((tmp_path / "artifacts/model.pth").resolve())
    assert ffs.model_config_path == str((tmp_path / "artifacts/cfg.yaml").resolve())
    assert ffs.calibration_path == str((tmp_path / "calibration.json").resolve())
