from __future__ import annotations

import torch
import yaml

from pointcloud_builder import PointCloudBuilder


def test_builder_from_yaml_instantiates() -> None:
    builder = PointCloudBuilder.from_yaml("configs/example_head_depth_raw.yaml")
    assert builder.camera.width == 640
    assert builder.camera.active_intrinsics.fx == 600.0
    assert builder.config.pointcloud.use_rgb is False
    assert builder.config.crop.enabled is True
    assert builder.config.sampling.enabled is True


def test_builder_returns_sampled_fixed_size_point_cloud() -> None:
    builder = PointCloudBuilder.from_yaml("configs/example_head_depth_raw.yaml")
    frame = {
        "depth": torch.ones((builder.camera.height, builder.camera.width), dtype=torch.float32),
    }
    pc, meta = builder.from_recorded_frame(frame)
    assert pc.shape == (builder.config.sampling.num_points, 3)
    assert meta["stage"] == "sampled"
    assert meta["num_raw_points"] == builder.camera.height * builder.camera.width
    assert meta["num_cropped_points"] <= meta["num_raw_points"]
    assert meta["num_sampled_points"] == builder.config.sampling.num_points
    assert meta["crop_enabled"] is True
    assert meta["sampling_enabled"] is True
    assert meta["sampling_mode"] == builder.config.sampling.mode
    assert meta["target_num_points"] == builder.config.sampling.num_points
    assert "input_empty" in meta
    assert "padded" in meta
    assert "pad_mode" in meta
    assert "voxel_size" in meta


def test_recorded_and_live_outputs_have_same_shape() -> None:
    builder = PointCloudBuilder.from_yaml("configs/example_head_aligned.yaml")
    frame = {
        "depth": torch.ones((builder.camera.height, builder.camera.width), dtype=torch.float32),
        "rgb": torch.ones((builder.camera.height, builder.camera.width, 3), dtype=torch.uint8),
    }
    recorded, recorded_meta = builder.from_recorded_frame(frame)
    live, live_meta = builder.from_live_frame(frame)
    assert recorded.shape == live.shape
    assert recorded.shape == (builder.config.sampling.num_points, 6)
    assert recorded_meta["num_sampled_points"] == live_meta["num_sampled_points"]


def test_live_with_stages_reuses_sampled_output() -> None:
    builder = PointCloudBuilder.from_yaml("configs/example_head_depth_raw.yaml")
    frame = {
        "depth": torch.ones((builder.camera.height, builder.camera.width), dtype=torch.float32),
    }

    live, meta, stages = builder.from_live_frame_with_stages(frame)

    assert stages["sampled"] is live
    assert set(stages) == {"raw", "cropped", "sampled"}
    assert stages["raw"].shape[0] == meta["num_raw_points"]
    assert stages["cropped"].shape[0] == meta["num_cropped_points"]
    assert stages["sampled"].shape[0] == meta["num_sampled_points"]
    assert meta["mode"] == "live"


def test_cuda_output_does_not_crash_when_available() -> None:
    if not torch.cuda.is_available():
        return
    builder = PointCloudBuilder.from_yaml("configs/example_head_depth_raw.yaml")
    frame = {
        "depth": torch.ones((builder.camera.height, builder.camera.width), dtype=torch.float32),
    }
    pc, meta = builder.from_live_frame(frame)
    assert pc.is_cuda
    assert pc.shape == (builder.config.sampling.num_points, 3)
    assert meta["device"].startswith("cuda")


def test_raw_depth_projects_to_color_for_rgb_points(tmp_path) -> None:
    config_path = tmp_path / "raw_depth_to_color.yaml"
    config = {
        "device": "cpu",
        "camera": {
            "name": "test",
            "aligned_depth_to_color": False,
            "depth_scale": 1.0,
            "depth_intrinsics": {"width": 2, "height": 2, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
            "color_intrinsics": {"width": 2, "height": 2, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
            "depth_to_color_extrinsics": {
                "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "translation": [1.0, 0.0, 0.0],
            },
        },
        "pointcloud": {
            "use_rgb": True,
            "output_format": "xyzrgb",
            "rgb_mapping": "project_depth_to_color",
            "rgb_sampling": "nearest",
            "xyz_frame": "depth",
        },
        "sampling": {"enabled": False, "mode": "voxel_random", "num_points": 4},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    builder = PointCloudBuilder.from_yaml(config_path)
    rgb = torch.tensor(
        [
            [[1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [1, 1, 1]],
        ],
        dtype=torch.uint8,
    )
    pc, meta = builder.from_recorded_frame({"depth": torch.ones((2, 2)), "rgb": rgb})
    assert pc.shape == (4, 6)
    assert torch.allclose(pc[0, 3:], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(pc[1, 3:], torch.zeros(3))
    assert meta["rgb"]["mapping"] == "project_depth_to_color"
    assert meta["rgb"]["valid_projection_count"] == 2
    assert meta["rgb"]["invalid_projection_count"] == 2
