from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pointcloud_builder import PointCloudBuilder
from pointcloud_builder.camera_model import CameraIntrinsics
from pointcloud_builder.config import CameraConfig, PointCloudBuilderConfig, PointCloudConfig
from pointcloud_builder.deprojection import deproject_depth


def make_builder(
    *,
    aligned_depth_to_color: bool,
    use_rgb: bool,
    output_format: str = "xyzrgb",
    device: str = "cpu",
) -> PointCloudBuilder:
    color_intrinsics = CameraIntrinsics(width=3, height=3, fx=1.0, fy=1.0, cx=1.0, cy=1.0)
    depth_intrinsics = CameraIntrinsics(width=3, height=3, fx=1.0, fy=1.0, cx=0.0, cy=0.0)
    config = PointCloudBuilderConfig(
        device=device,
        camera=CameraConfig(
            name="test",
            depth_scale=1.0,
            aligned_depth_to_color=aligned_depth_to_color,
            color_intrinsics=color_intrinsics,
            depth_intrinsics=depth_intrinsics,
        ),
        pointcloud=PointCloudConfig(use_rgb=use_rgb, output_format=output_format),
    )
    return PointCloudBuilder(config)


def test_constant_depth_plane_z_equals_depth_m() -> None:
    intrinsics = CameraIntrinsics(width=3, height=3, fx=1.0, fy=1.0, cx=1.0, cy=1.0)
    depth = torch.full((3, 3), 1000, dtype=torch.uint16)
    points, mask = deproject_depth(depth, intrinsics, depth_scale=0.001)
    assert points.shape == (9, 3)
    assert mask.shape == (9,)
    assert torch.allclose(points[:, 2], torch.ones(9))


def test_center_pixel_projects_to_zero_xy() -> None:
    intrinsics = CameraIntrinsics(width=3, height=3, fx=1.0, fy=1.0, cx=1.0, cy=1.0)
    points, _ = deproject_depth(torch.ones((3, 3)), intrinsics, depth_scale=1.0)
    center = points[4]
    assert center[0].item() == pytest.approx(0.0)
    assert center[1].item() == pytest.approx(0.0)


def test_aligned_depth_to_color_uses_color_intrinsics() -> None:
    builder = make_builder(aligned_depth_to_color=True, use_rgb=False, output_format="xyz")
    pc, meta = builder.from_live_frame({"depth": torch.ones((3, 3))})
    assert pc[0, 0].item() == pytest.approx(-1.0)
    assert meta["intrinsics"] == "color"
    assert meta["aligned_depth_to_color"] is True


def test_raw_depth_uses_depth_intrinsics() -> None:
    builder = make_builder(aligned_depth_to_color=False, use_rgb=True)
    rgb = torch.zeros((3, 3, 3), dtype=torch.uint8)
    pc, meta = builder.from_live_frame({"depth": torch.ones((3, 3)), "rgb": rgb})
    assert pc[0, 0].item() == pytest.approx(0.0)
    assert pc.shape == (9, 3)
    assert meta["intrinsics"] == "depth"
    assert meta["use_rgb"] is False


def test_use_rgb_true_and_aligned_outputs_xyzrgb() -> None:
    builder = make_builder(aligned_depth_to_color=True, use_rgb=True)
    rgb = torch.full((3, 3, 3), 255, dtype=torch.uint8)
    pc, meta = builder.from_recorded_frame(
        {
            "depth": torch.ones((3, 3)),
            "rgb": rgb,
            "timestamp": 12.5,
            "global_frame_index": 7,
        }
    )
    assert pc.shape == (9, 6)
    assert torch.allclose(pc[:, 3:6], torch.ones((9, 3)))
    assert meta["stage"] == "raw"
    assert meta["use_rgb"] is True
    assert meta["num_raw_points"] == 9
    assert meta["timestamp"] == 12.5
    assert meta["global_frame_index"] == 7


def test_use_rgb_false_outputs_xyz() -> None:
    builder = make_builder(aligned_depth_to_color=True, use_rgb=False, output_format="xyz")
    pc, meta = builder.from_live_frame(
        {
            "depth": torch.ones((3, 3)),
            "rgb": torch.ones((3, 3, 3), dtype=torch.uint8),
        }
    )
    assert pc.shape == (9, 3)
    assert meta["use_rgb"] is False


def test_invalid_depth_is_filtered() -> None:
    intrinsics = CameraIntrinsics(width=2, height=2, fx=1.0, fy=1.0, cx=0.0, cy=0.0)
    depth = torch.tensor([[0.0, 1.0], [-1.0, 2.0]], dtype=torch.float32)
    points, mask = deproject_depth(depth, intrinsics, depth_scale=1.0)
    assert points.shape == (2, 3)
    assert mask.tolist() == [False, True, False, True]


def test_cuda_request_falls_back_to_cpu_when_unavailable() -> None:
    builder = make_builder(aligned_depth_to_color=False, use_rgb=False, device="cuda")
    pc, meta = builder.from_live_frame({"depth": torch.ones((3, 3))})
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert builder.device.type == expected_device
    assert meta["device"].startswith(expected_device)
    assert pc.shape == (9, 3)


def test_yaml_new_schema_can_instantiate(tmp_path: Path) -> None:
    config_path = tmp_path / "camera.yaml"
    config_path.write_text(
        """
device: "cpu"
camera:
  name: "head"
  aligned_depth_to_color: true
  depth_scale: 0.001
  color_intrinsics:
    width: 3
    height: 3
    fx: 1.0
    fy: 1.0
    cx: 1.0
    cy: 1.0
  depth_intrinsics:
    width: 3
    height: 3
    fx: 1.0
    fy: 1.0
    cx: 0.0
    cy: 0.0
pointcloud:
  use_rgb: true
  output_format: "xyzrgb"
""",
        encoding="utf-8",
    )
    builder = PointCloudBuilder.from_yaml(config_path)
    assert builder.camera.name == "head"
    assert builder.camera.active_intrinsics.cx == 1.0
