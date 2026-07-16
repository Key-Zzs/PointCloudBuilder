from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from pointcloud_builder import PointCloudBuilder
from pointcloud_builder.camera_model import CameraExtrinsics, CameraIntrinsics
from pointcloud_builder.config import parse_config
from pointcloud_builder.ffs.geometry import disparity_to_depth
from pointcloud_builder.ffs.types import FFSDepthResult


H, W = 480, 640
INTRINSICS = CameraIntrinsics(width=W, height=H, fx=100.0, fy=100.0, cx=319.5, cy=239.5)
IDENTITY = CameraExtrinsics(
    rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    translation=(0.0, 0.0, 0.0),
)


@dataclass
class FakeEstimator:
    calls: int = 0

    def infer(self, frame: object) -> FFSDepthResult:
        self.calls += 1
        disparity = torch.full((H, W), 10.0, dtype=torch.float32)
        depth, valid, counts = disparity_to_depth(
            disparity,
            fx_px=INTRINSICS.fx,
            baseline_m=0.05,
            remove_invisible=True,
        )
        return FFSDepthResult(
            disparity_px=disparity,
            depth_m=depth,
            valid_mask=valid,
            intrinsics=INTRINSICS,
            depth_to_color_extrinsics=IDENTITY,
            metadata={
                "backend": "fake",
                "depth_source": "ffs_stereo",
                "valid_disparity_count": counts["valid"],
                "effective_depth_scale": 1.0,
            },
        )


def _config(*, use_rgb: bool = True):
    return parse_config(
        {
            "device": "cpu",
            "camera": {
                "name": "head",
                "depth_scale": 0.001,
                "aligned_depth_to_color": False,
                "color_intrinsics": {"width": W, "height": H, "fx": 100.0, "fy": 100.0, "cx": 319.5, "cy": 239.5},
                "depth_intrinsics": {"width": W, "height": H, "fx": 100.0, "fy": 100.0, "cx": 319.5, "cy": 239.5},
                "depth_to_color_extrinsics": {
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "translation": [0, 0, 0],
                },
            },
            "pointcloud": {
                "use_rgb": use_rgb,
                "output_format": "xyzrgb" if use_rgb else "xyz",
                "rgb_mapping": "project_depth_to_color",
            },
            "crop": {"enabled": True, "x": [-100, 100], "y": [-100, 100], "z": [0.4, 0.6]},
            "sampling": {"enabled": True, "mode": "stride", "num_points": 16, "stride": 1, "pad_mode": "repeat"},
            "depth_source": {
                "mode": "ffs_stereo",
                "ffs": {"backend": "pytorch", "baseline_m": 0.05, "max_disp": 192, "valid_iters": 8},
            },
        }
    )


def test_ffs_depth_uses_scale_one_and_reuses_builder_stages() -> None:
    fake = FakeEstimator()
    builder = PointCloudBuilder(_config(), depth_estimator=fake)
    frame = {
        "depth": torch.full((H, W), 1000, dtype=torch.float32),
        "left_ir": torch.zeros((H, W), dtype=torch.uint8),
        "right_ir": torch.zeros((H, W), dtype=torch.uint8),
        "rgb": torch.full((H, W, 3), 255, dtype=torch.uint8),
    }
    point_cloud, meta = builder.from_live_frame(frame)
    assert fake.calls == 1
    assert tuple(point_cloud.shape) == (16, 6)
    assert torch.allclose(point_cloud[:, 2], torch.full((16,), 0.5))
    assert torch.allclose(point_cloud[:, 3:], torch.ones((16, 3)))
    assert meta["depth_source"] == "ffs_stereo"
    assert meta["effective_depth_scale"] == 1.0
    assert meta["intrinsics"] == "ir1"
    stages, _ = builder.build_stages(frame)
    assert set(stages) == {"raw", "cropped", "sampled"}
    perception, _ = builder.build_perception_stages(frame)
    assert set(perception) == {"left_ir", "right_ir", "disparity", "depth", "valid_mask", "raw", "cropped", "sampled"}
    assert fake.calls == 3


def test_frame_mode_missing_depth_keeps_clear_error() -> None:
    config = parse_config(
        {
            "device": "cpu",
            "camera": {
                "name": "camera",
                "depth_scale": 1.0,
                "aligned_depth_to_color": False,
                "color_intrinsics": {"width": 2, "height": 2, "fx": 1, "fy": 1, "cx": 0, "cy": 0},
                "depth_intrinsics": {"width": 2, "height": 2, "fx": 1, "fy": 1, "cx": 0, "cy": 0},
            },
            "pointcloud": {"use_rgb": False, "output_format": "xyz"},
        }
    )
    with pytest.raises(KeyError, match="depth"):
        PointCloudBuilder(config).from_live_frame({"left_ir": torch.ones((2, 2)), "right_ir": torch.ones((2, 2))})
