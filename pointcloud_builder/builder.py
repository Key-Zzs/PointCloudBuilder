"""High-level RGB-D to point-cloud builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from pointcloud_builder.camera_model import CameraModel
from pointcloud_builder.config import PointCloudBuilderConfig, load_config
from pointcloud_builder.crop import crop_points
from pointcloud_builder.deprojection import deproject_depth
from pointcloud_builder.sampling import sample_points
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
        """Build a fixed-size point cloud from an offline recorded frame."""

        return self._from_frame(frame, source="recorded")

    def from_live_frame(self, frame: RGBDFrame | dict[str, Any]) -> tuple[Tensor, Meta]:
        """Build a fixed-size point cloud from a live inference frame."""

        return self._from_frame(frame, source="live")

    def build_stages(self, frame: RGBDFrame | dict[str, Any]) -> tuple[dict[str, Tensor], Meta]:
        """Return raw, cropped, and sampled stages for offline inspection."""

        pc, meta, stages = self._process_frame(frame, source="staged")
        stages["output"] = pc
        return stages, meta

    def _from_frame(self, frame: RGBDFrame | dict[str, Any], source: str) -> tuple[Tensor, Meta]:
        pc, meta, _ = self._process_frame(frame, source=source)
        return pc, meta

    def _process_frame(
        self,
        frame: RGBDFrame | dict[str, Any],
        source: str,
    ) -> tuple[Tensor, Meta, dict[str, Tensor]]:
        depth = as_tensor(get_frame_value(frame, "depth"), self.device, torch.float32)
        raw_points, valid_mask = deproject_depth(depth, self.camera)
        raw_colors = self._aligned_colors(frame, valid_mask)
        cropped_points, cropped_colors, _ = crop_points(raw_points, self.config.crop, raw_colors)
        sampled_points, sampled_colors, sampling_meta = sample_points(
            cropped_points,
            self.config.sampling,
            cropped_colors,
        )
        raw_pc = pack_point_cloud(raw_points, raw_colors)
        cropped_pc = pack_point_cloud(cropped_points, cropped_colors)
        sampled_pc = pack_point_cloud(sampled_points, sampled_colors)
        meta: Meta = {
            "source": source,
            "device": str(self.device),
            "aligned_depth_to_color": self.camera.aligned_depth_to_color,
            "has_rgb": sampled_colors is not None,
            "raw_count": int(raw_points.shape[0]),
            "cropped_count": int(cropped_points.shape[0]),
            "sampled_count": int(sampled_points.shape[0]),
            "sampling": sampling_meta,
        }
        return sampled_pc, meta, {
            "raw": raw_pc,
            "cropped": cropped_pc,
            "sampled": sampled_pc,
        }

    def _aligned_colors(self, frame: RGBDFrame | dict[str, Any], valid_mask: Tensor) -> Tensor | None:
        if not self.camera.aligned_depth_to_color:
            return None
        color_value = get_optional_frame_value(frame, "color")
        if color_value is None:
            return None
        color = normalize_color(as_tensor(color_value, self.device, torch.float32))
        if int(color.shape[0]) != self.camera.height or int(color.shape[1]) != self.camera.width:
            raise ValueError(
                f"Color shape {tuple(color.shape)} does not match "
                f"camera height/width {(self.camera.height, self.camera.width)}"
            )
        return color.reshape(-1, 3)[valid_mask]
