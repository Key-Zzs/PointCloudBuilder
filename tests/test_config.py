from __future__ import annotations

from pointcloud_builder.config import load_config


def test_yaml_config_parses() -> None:
    config = load_config("configs/example_head_aligned.yaml")
    assert config.camera.color_intrinsics.width == 640
    assert config.camera.depth_intrinsics.height == 480
    assert config.camera.aligned_depth_to_color is True
    assert config.pointcloud.use_rgb is True
    assert config.pointcloud.output_format == "xyzrgb"
