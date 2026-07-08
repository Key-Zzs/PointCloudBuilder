"""High-level RGB-D to point-cloud builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from pointcloud_builder.camera_model import CameraModel
from pointcloud_builder.config import PointCloudBuilderConfig, load_config
from pointcloud_builder.crop import crop_point_cloud
from pointcloud_builder.deprojection import deproject_depth
from pointcloud_builder.sampling import sample_point_cloud
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
        """Build a fixed-size camera-frame point cloud from an offline recorded frame."""

        pc, meta, _ = self._build_from_frame(frame, mode="recorded")
        return pc, meta

    def from_live_frame(self, frame: RGBDFrame | dict[str, Any]) -> tuple[Tensor, Meta]:
        """Build a fixed-size camera-frame point cloud from a live inference frame."""

        pc, meta, _ = self._build_from_frame(frame, mode="live")
        return pc, meta

    def build_stages(self, frame: RGBDFrame | dict[str, Any]) -> tuple[dict[str, Tensor], Meta]:
        """Return raw, cropped, and sampled stage tensors for offline inspection."""

        _, meta, stages = self._build_from_frame(frame, mode="staged")
        return stages, meta

    def _build_from_frame(
        self,
        frame: RGBDFrame | dict[str, Any],
        mode: str,
    ) -> tuple[Tensor, Meta, dict[str, Tensor]]:
        depth = as_tensor(get_frame_value(frame, "depth"), self.device, torch.float32)
        intrinsics = self.camera.active_intrinsics
        points, valid_mask = deproject_depth(
            depth,
            intrinsics,
            self.camera.depth_scale,
            flatten=True,
        )
        colors, rgb_meta = self._rgb_for_raw_points(frame, points, valid_mask)
        raw_point_cloud = pack_point_cloud(points, colors)
        cropped_point_cloud, _ = crop_point_cloud(raw_point_cloud, self.config.crop)
        sampled_point_cloud, sampling_meta = sample_point_cloud(cropped_point_cloud, self.config.sampling)
        output_stage = "sampled" if self.config.sampling.enabled else ("cropped" if self.config.crop.enabled else "raw")
        meta: Meta = {
            "stage": output_stage,
            "mode": mode,
            "aligned_depth_to_color": self.camera.aligned_depth_to_color,
            "use_rgb": colors is not None,
            "num_raw_points": int(points.shape[0]),
            "num_cropped_points": int(cropped_point_cloud.shape[0]),
            "num_sampled_points": int(sampled_point_cloud.shape[0]),
            "crop_enabled": self.config.crop.enabled,
            "crop_range": {
                "frame": self.config.crop.frame,
                "x": self.config.crop.x,
                "y": self.config.crop.y,
                "z": self.config.crop.z,
            },
            "crop_empty": self.config.crop.enabled and int(cropped_point_cloud.shape[0]) == 0,
            "sampling_enabled": self.config.sampling.enabled,
            "sampling_mode": self.config.sampling.mode,
            "target_num_points": self.config.sampling.num_points,
            "input_empty": bool(sampling_meta["input_empty"]),
            "padded": bool(sampling_meta["padded"]),
            "pad_mode": self.config.sampling.pad_mode,
            "voxel_size": self.config.sampling.voxel_size,
            "sampling": sampling_meta,
            "device": str(self.device),
            "timestamp": get_optional_frame_value(frame, "timestamp"),
            "global_frame_index": get_optional_frame_value(frame, "global_frame_index"),
            "camera_name": self.camera.name,
            "intrinsics": "color" if self.camera.aligned_depth_to_color else "depth",
            "rgb": rgb_meta,
        }
        return sampled_point_cloud, meta, {
            "raw": raw_point_cloud,
            "cropped": cropped_point_cloud,
            "sampled": sampled_point_cloud,
        }

    def _rgb_for_raw_points(
        self,
        frame: RGBDFrame | dict[str, Any],
        points: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor | None, Meta]:
        if not self._wants_rgb_output():
            return None, {
                "enabled": False,
                "mapping": self.config.pointcloud.rgb_mapping,
            }
        rgb_value = get_optional_frame_value(frame, "rgb")
        if rgb_value is None:
            rgb_value = get_optional_frame_value(frame, "color")
        if rgb_value is None:
            raise ValueError("RGB point cloud requested but frame has no 'rgb' or 'color' field")
        rgb = normalize_color(as_tensor(rgb_value, self.device, torch.float32))
        intrinsics = self.camera.color_intrinsics
        if int(rgb.shape[0]) != intrinsics.height or int(rgb.shape[1]) != intrinsics.width:
            raise ValueError(
                f"RGB shape {tuple(rgb.shape)} does not match "
                f"color height/width {(intrinsics.height, intrinsics.width)}"
            )
        if self.camera.aligned_depth_to_color:
            return rgb.reshape(-1, 3)[valid_mask], {
                "enabled": True,
                "mapping": "aligned",
                "sampling": "nearest",
                "invalid_projection_count": 0,
            }
        if self.config.pointcloud.rgb_mapping != "project_depth_to_color":
            return None, {
                "enabled": False,
                "mapping": self.config.pointcloud.rgb_mapping,
                "reason": "raw_depth_rgb_mapping_not_configured",
            }
        return self._project_depth_points_to_color(points, rgb)

    def _wants_rgb_output(self) -> bool:
        return self.config.pointcloud.use_rgb

    def _project_depth_points_to_color(self, points_depth: Tensor, rgb: Tensor) -> tuple[Tensor, Meta]:
        extrinsics = self.camera.depth_to_color_extrinsics
        if extrinsics is None:
            raise ValueError("camera.depth_to_color_extrinsics is required for RGB mapping")
        rotation = torch.tensor(extrinsics.rotation, dtype=points_depth.dtype, device=points_depth.device)
        translation = torch.tensor(extrinsics.translation, dtype=points_depth.dtype, device=points_depth.device)
        points_color = points_depth @ rotation.T + translation
        z = points_color[:, 2]
        finite_depth = torch.isfinite(points_color).all(dim=-1) & (z > 0.0)

        intrinsics = self.camera.color_intrinsics
        u = points_color[:, 0] * intrinsics.fx / z + intrinsics.cx
        v = points_color[:, 1] * intrinsics.fy / z + intrinsics.cy
        u_nearest = torch.round(u).to(dtype=torch.long)
        v_nearest = torch.round(v).to(dtype=torch.long)
        in_bounds = (
            finite_depth
            & (u_nearest >= 0)
            & (u_nearest < intrinsics.width)
            & (v_nearest >= 0)
            & (v_nearest < intrinsics.height)
        )

        colors = torch.zeros((points_depth.shape[0], 3), dtype=rgb.dtype, device=rgb.device)
        if bool(in_bounds.any()):
            colors[in_bounds] = rgb[v_nearest[in_bounds], u_nearest[in_bounds], :3]
        return colors, {
            "enabled": True,
            "mapping": "project_depth_to_color",
            "sampling": "nearest",
            "invalid_projection_count": int((~in_bounds).sum().item()),
            "valid_projection_count": int(in_bounds.sum().item()),
        }
