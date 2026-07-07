"""High-level RGB-D to point-cloud builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from pointcloud_builder.camera_model import CameraModel
from pointcloud_builder.config import PointCloudBuilderConfig, load_config
from pointcloud_builder.deprojection import deproject_depth
from pointcloud_builder.types import Meta, RGBDFrame, Tensor
from pointcloud_builder.utils import (
    as_tensor,
    get_frame_value,
    get_optional_frame_value,
    normalize_color,
    pack_point_cloud,
    resolve_device,
)


class PointCloudBuilder:
    """Reusable RGB-D point-cloud builder for training and deployment."""

    def __init__(self, config: PointCloudBuilderConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.camera = CameraModel.from_config(config.camera)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PointCloudBuilder":
        """Instantiate a builder from a YAML configuration file."""

        return cls(load_config(path))

    def from_recorded_frame(self, frame: RGBDFrame | dict[str, Any]) -> tuple[Tensor, Meta]:
        """Build a raw camera-frame point cloud from an offline recorded frame."""

        return self._build_from_frame(frame, mode="recorded")

    def from_live_frame(self, frame: RGBDFrame | dict[str, Any]) -> tuple[Tensor, Meta]:
        """Build a raw camera-frame point cloud from a live inference frame."""

        return self._build_from_frame(frame, mode="live")

    def build_stages(self, frame: RGBDFrame | dict[str, Any]) -> tuple[dict[str, Tensor], Meta]:
        """Return raw stage tensors for offline inspection."""

        pc, meta = self._build_from_frame(frame, mode="staged")
        return {"raw": pc}, meta

    def _build_from_frame(self, frame: RGBDFrame | dict[str, Any], mode: str) -> tuple[Tensor, Meta]:
        depth = as_tensor(get_frame_value(frame, "depth"), self.device, torch.float32)
        intrinsics = self.camera.active_intrinsics
        points, valid_mask = deproject_depth(
            depth,
            intrinsics,
            self.camera.depth_scale,
            flatten=True,
        )
        colors = self._rgb_for_raw_points(frame, valid_mask)
        point_cloud = pack_point_cloud(points, colors)
        meta: Meta = {
            "stage": "raw",
            "mode": mode,
            "aligned_depth_to_color": self.camera.aligned_depth_to_color,
            "use_rgb": colors is not None,
            "num_raw_points": int(points.shape[0]),
            "device": str(self.device),
            "timestamp": get_optional_frame_value(frame, "timestamp"),
            "global_frame_index": get_optional_frame_value(frame, "global_frame_index"),
            "camera_name": self.camera.name,
            "intrinsics": "color" if self.camera.aligned_depth_to_color else "depth",
        }
        return point_cloud, meta

    def _rgb_for_raw_points(self, frame: RGBDFrame | dict[str, Any], valid_mask: Tensor) -> Tensor | None:
        if not self.camera.aligned_depth_to_color:
            return None
        if not self.config.pointcloud.use_rgb:
            return None
        if self.config.pointcloud.output_format != "xyzrgb":
            return None
        rgb_value = get_optional_frame_value(frame, "rgb")
        if rgb_value is None:
            rgb_value = get_optional_frame_value(frame, "color")
        if rgb_value is None:
            return None
        rgb = normalize_color(as_tensor(rgb_value, self.device, torch.float32))
        intrinsics = self.camera.color_intrinsics
        if int(rgb.shape[0]) != intrinsics.height or int(rgb.shape[1]) != intrinsics.width:
            raise ValueError(
                f"RGB shape {tuple(rgb.shape)} does not match "
                f"color height/width {(intrinsics.height, intrinsics.width)}"
            )
        return rgb.reshape(-1, 3)[valid_mask]
