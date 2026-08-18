"""High-level RGB-D to point-cloud builder."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import torch

from pointcloud_builder.camera_model import CameraModel
from pointcloud_builder.config import PointCloudBuilderConfig, load_config
from pointcloud_builder.crop import crop_point_cloud
from pointcloud_builder.deprojection import deproject_depth
from pointcloud_builder.instance import InstanceFrameResult, build_instance_dense, build_instance_sparse
from pointcloud_builder.projection import ProjectionMap, project_points_to_color_image
from pointcloud_builder.sampling import sample_point_cloud
from pointcloud_builder.segmentation.types import InstanceMask
from pointcloud_builder.support_plane import SupportPlane, filter_support_plane, load_support_plane
from pointcloud_builder.ffs.types import ResolvedDepth
from pointcloud_builder.types import Meta, RGBDFrame, StereoIRFrame, Tensor
from pointcloud_builder.utils import (
    as_tensor,
    get_frame_value,
    get_optional_frame_value,
    normalize_color,
    pack_point_cloud,
    resolve_device,
)


class _StageTiming:
    """Low-overhead stage timing that uses one synchronized CUDA timeline."""

    def __init__(self, device: torch.device) -> None:
        self.cuda = device.type == "cuda"
        self._events: dict[str, torch.cuda.Event] = {}
        if self.cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._start_time = time.perf_counter()

    def mark(self, name: str) -> None:
        if self.cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._events[name] = event
        else:
            setattr(self, f"_{name}_time", time.perf_counter())

    def finish(self) -> dict[str, float]:
        if self.cuda:
            self._end_event.record()
            self._end_event.synchronize()
            ordered = ("deprojection", "rgb_mapping", "crop", "sampling")
            result: dict[str, float] = {}
            previous = self._start_event
            for name in ordered:
                event = self._events[name]
                result[name] = float(previous.elapsed_time(event))
                previous = event
            result["total_builder_pipeline"] = float(self._start_event.elapsed_time(self._end_event))
            return result
        end_time = time.perf_counter()
        result = {}
        previous_time = self._start_time
        for name in ("deprojection", "rgb_mapping", "crop", "sampling"):
            current = getattr(self, f"_{name}_time")
            result[name] = (current - previous_time) * 1000.0
            previous_time = current
        result["total_builder_pipeline"] = (end_time - self._start_time) * 1000.0
        return result


class PointCloudBuilder:
    """Reusable RGB-D point-cloud builder for training and deployment."""

    def __init__(self, config: PointCloudBuilderConfig, *, depth_estimator: Any | None = None) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.camera = CameraModel.from_config(config.camera)
        self.depth_estimator = depth_estimator
        self._support_plane: SupportPlane | None = None
        if config.support_plane.enabled and config.support_plane.cache_path:
            self._support_plane = load_support_plane(config.support_plane.cache_path)
        if config.support_plane.enabled and config.support_plane.model_source == "precomputed" and self._support_plane is None:
            raise ValueError("support_plane.model_source='precomputed' requires a readable cache_path")
        if config.depth_source.mode == "ffs_stereo":
            if config.depth_source.ffs is None:
                raise ValueError("depth_source.ffs is required for mode=ffs_stereo")
            if self.depth_estimator is None:
                from pointcloud_builder.ffs.estimator import FFSStereoDepthEstimator

                self.depth_estimator = FFSStereoDepthEstimator(
                    config.depth_source.ffs,
                    config.camera,
                    device=self.device,
                )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PointCloudBuilder":
        """Instantiate a builder from a YAML configuration file."""

        return cls(load_config(path))

    def from_recorded_frame(self, frame: RGBDFrame | StereoIRFrame | dict[str, Any]) -> tuple[Tensor, Meta]:
        """Build a fixed-size camera-frame point cloud from an offline recorded frame."""

        pc, meta, _ = self._build_from_frame(frame, mode="recorded")
        return pc, meta

    def from_live_frame(self, frame: RGBDFrame | StereoIRFrame | dict[str, Any]) -> tuple[Tensor, Meta]:
        """Build a fixed-size camera-frame point cloud from a live inference frame."""

        pc, meta, _ = self._build_from_frame(frame, mode="live")
        return pc, meta

    def from_live_frame_with_stages(
        self,
        frame: RGBDFrame | StereoIRFrame | dict[str, Any],
    ) -> tuple[Tensor, Meta, dict[str, Tensor]]:
        """Build one live frame and expose the same-pass intermediate tensors."""

        return self._build_from_frame(frame, mode="live")

    def build_stages(self, frame: RGBDFrame | StereoIRFrame | dict[str, Any]) -> tuple[dict[str, Tensor], Meta]:
        """Return raw, cropped, and sampled stage tensors for offline inspection."""

        _, meta, stages = self._build_from_frame(frame, mode="staged")
        return stages, meta

    def build_perception_stages(
        self,
        frame: RGBDFrame | StereoIRFrame | dict[str, Any],
        *,
        include_sampled: bool = True,
    ) -> tuple[dict[str, Tensor], Meta]:
        """Return FFS perception tensors plus the unchanged builder stages.

        ``build_stages`` deliberately remains limited to ``raw``, ``cropped``
        and ``sampled`` for compatibility. This separate entry point is only
        available for an FFS depth source.
        """

        resolved = self._resolve_depth(frame)
        if resolved.disparity_px is None or resolved.valid_mask is None:
            raise ValueError("build_perception_stages requires depth_source.mode=ffs_stereo")
        _, meta, stages = self._build_from_frame(
            frame,
            mode="staged",
            resolved=resolved,
            include_sampling=include_sampled,
        )
        perception = {
            "left_ir": as_tensor(get_frame_value(frame, self.config.depth_source.ffs.left_key), self.device, torch.float32),  # type: ignore[union-attr]
            "right_ir": as_tensor(get_frame_value(frame, self.config.depth_source.ffs.right_key), self.device, torch.float32),  # type: ignore[union-attr]
            "disparity": resolved.disparity_px,
            "depth": resolved.depth,
            "valid_mask": resolved.valid_mask,
            **stages,
        }, meta
        return perception

    def build_unfiltered_perception_stages(
        self,
        frame: RGBDFrame | StereoIRFrame | dict[str, Any],
    ) -> tuple[dict[str, Tensor], Meta]:
        """Return raw/workspace geometry for one episode-plane estimate.

        This is the sole escape hatch from an enabled support-plane profile and
        is deliberately explicit: it permits the offline episode estimator to
        see workspace geometry before a plane exists, but never performs a
        frame-by-frame plane fit or changes normal inference behavior.
        """

        resolved = self._resolve_depth(frame)
        if resolved.disparity_px is None or resolved.valid_mask is None:
            raise ValueError("build_unfiltered_perception_stages requires depth_source.mode=ffs_stereo")
        _, meta, stages = self._build_from_frame(frame, mode="stage2_plane_estimation", resolved=resolved, apply_support_plane=False)
        return {
            "left_ir": as_tensor(get_frame_value(frame, self.config.depth_source.ffs.left_key), self.device, torch.float32),  # type: ignore[union-attr]
            "right_ir": as_tensor(get_frame_value(frame, self.config.depth_source.ffs.right_key), self.device, torch.float32),  # type: ignore[union-attr]
            "disparity": resolved.disparity_px,
            "depth": resolved.depth,
            "valid_mask": resolved.valid_mask,
            **stages,
        }, meta

    @property
    def support_plane(self) -> SupportPlane | None:
        """Episode plane currently cached by the caller; never auto-refit per frame."""

        return self._support_plane

    def set_support_plane(self, plane: SupportPlane) -> None:
        """Install a precomputed/episode-consensus plane for subsequent frames."""

        self._support_plane = plane

    def project_points_to_color_image(self, points: Tensor, *, source_frame: str = "depth") -> ProjectionMap:
        """Expose 3D-to-RGB correspondence even for XYZ-only output profiles."""

        extrinsics = None if self.camera.aligned_depth_to_color else self.camera.depth_to_color_extrinsics
        if not self.camera.aligned_depth_to_color and extrinsics is None:
            raise ValueError("camera.depth_to_color_extrinsics is required for raw-depth RGB projection")
        return project_points_to_color_image(
            points,
            extrinsics=extrinsics,
            intrinsics=self.camera.color_intrinsics,
            source_frame=source_frame,
        )

    def build_instance_dense(
        self,
        frame: RGBDFrame | StereoIRFrame | dict[str, Any],
        masks: list[InstanceMask],
        *,
        expected_instances: dict[str, int] | None = None,
    ) -> tuple[InstanceFrameResult, Meta]:
        """Stage-2 primary: raw dense points -> RGB masks -> visible instances."""

        stages, meta = self.build_perception_stages(frame)
        raw = stages["raw"]
        return build_instance_dense(
            raw_dense_points=raw,
            projection=self.project_points_to_color_image(raw, source_frame="raw_dense"),
            masks=masks,
            sampling_config=self.config.instance_sampling.as_sampling_config(),
            support_plane=self._required_support_plane(),
            expected_instances=expected_instances,
            visibility_filter=self.config.visibility_filter,
        ), meta

    def build_instance_sparse(
        self,
        frame: RGBDFrame | StereoIRFrame | dict[str, Any],
        masks: list[InstanceMask],
        *,
        expected_instances: dict[str, int] | None = None,
    ) -> tuple[InstanceFrameResult, Meta]:
        """Diagnostic mode: workspace sampling occurs before mask selection."""

        stages, meta = self.build_perception_stages(frame)
        sampled = stages["sampled"]
        return build_instance_sparse(
            workspace_sampled_points=sampled,
            projection=self.project_points_to_color_image(sampled, source_frame="workspace_sampled"),
            masks=masks,
            sampling_config=self.config.instance_sampling.as_sampling_config(),
            expected_instances=expected_instances,
            visibility_filter=self.config.visibility_filter,
        ), meta

    def _required_support_plane(self) -> SupportPlane | None:
        if not self.config.support_plane.enabled:
            return None
        if self._support_plane is None:
            raise RuntimeError(
                "support-plane filtering is enabled but no episode/precomputed model is installed; "
                "estimate it outside the frame loop and call set_support_plane()"
            )
        return self._support_plane

    def _build_from_frame(
        self,
        frame: RGBDFrame | StereoIRFrame | dict[str, Any],
        mode: str,
        resolved: ResolvedDepth | None = None,
        apply_support_plane: bool = True,
        include_sampling: bool = True,
    ) -> tuple[Tensor, Meta, dict[str, Tensor]]:
        resolved = resolved or self._resolve_depth(frame)
        depth = resolved.depth
        intrinsics = resolved.intrinsics
        stage_timer = _StageTiming(self.device) if resolved.metadata is not None else None
        points, valid_mask = deproject_depth(
            depth,
            intrinsics,
            resolved.effective_depth_scale,
            flatten=True,
        )
        if stage_timer is not None:
            stage_timer.mark("deprojection")
        colors, rgb_meta = self._rgb_for_raw_points(
            frame,
            points,
            valid_mask,
            depth_to_color_extrinsics=resolved.depth_to_color_extrinsics,
        )
        if stage_timer is not None:
            stage_timer.mark("rgb_mapping")
        raw_point_cloud = pack_point_cloud(points, colors)
        cropped_point_cloud, _ = crop_point_cloud(raw_point_cloud, self.config.crop)
        if stage_timer is not None:
            stage_timer.mark("crop")
        support_filtered_point_cloud = cropped_point_cloud
        support_keep_mask: Tensor | None = None
        if self.config.support_plane.enabled and apply_support_plane:
            support_filtered_point_cloud, support_keep_mask = filter_support_plane(
                cropped_point_cloud,
                self._required_support_plane(),
                distance_threshold_m=self.config.support_plane.distance_threshold_m,
            )
        if include_sampling:
            sampled_point_cloud, sampling_meta = sample_point_cloud(support_filtered_point_cloud, self.config.sampling)
        else:
            # Mode1 consumes raw dense geometry. Avoid constructing the
            # policy-view global sample unless optional Mode2 requested it.
            sampled_point_cloud = support_filtered_point_cloud
            sampling_meta = {"input_empty": int(support_filtered_point_cloud.shape[0]) == 0, "padded": False}
        if stage_timer is not None:
            stage_timer.mark("sampling")
        stage_timing = stage_timer.finish() if stage_timer is not None else None
        output_stage = "sampled" if include_sampling and self.config.sampling.enabled else (
            "support_filtered" if self.config.support_plane.enabled and apply_support_plane else ("cropped" if self.config.crop.enabled else "raw")
        )
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
            "sampling_executed": include_sampling,
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
            "intrinsics": (
                "ir1"
                if resolved.frame_name == "ffs_stereo"
                else ("color" if self.camera.aligned_depth_to_color else "depth")
            ),
            "rgb": rgb_meta,
        }
        if resolved.metadata is not None:
            meta["depth_source"] = "ffs_stereo"
            meta["effective_depth_scale"] = resolved.effective_depth_scale
            meta["ffs"] = resolved.metadata
            meta["ffs"].setdefault("timing_ms", {}).update(
                stage_timing or {}
            )
        if self.config.support_plane.enabled and apply_support_plane:
            plane = self._required_support_plane()
            meta["support_plane"] = {
                "enabled": True,
                "model_source": self.config.support_plane.model_source,
                "normal": list(plane.normal),
                "offset": plane.offset,
                "distance_threshold_m": self.config.support_plane.distance_threshold_m,
                "source_frame_indices": list(plane.source_frame_indices),
                "input_count": int(cropped_point_cloud.shape[0]),
                "remaining_count": int(support_filtered_point_cloud.shape[0]),
                "removed_count": int((~support_keep_mask).sum().item()) if support_keep_mask is not None else 0,
            }
        stages = {
            "raw": raw_point_cloud,
            "cropped": cropped_point_cloud,
        }
        if include_sampling:
            stages["sampled"] = sampled_point_cloud
        # Preserve the legacy build_stages key set exactly unless the new
        # support-plane feature was explicitly enabled in the YAML.
        if self.config.support_plane.enabled and apply_support_plane:
            stages["support_filtered"] = support_filtered_point_cloud
        return sampled_point_cloud, meta, stages

    def _rgb_for_raw_points(
        self,
        frame: RGBDFrame | StereoIRFrame | dict[str, Any],
        points: Tensor,
        valid_mask: Tensor,
        *,
        depth_to_color_extrinsics: Any,
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
        return self._project_depth_points_to_color(points, rgb, depth_to_color_extrinsics)

    def _wants_rgb_output(self) -> bool:
        return self.config.pointcloud.use_rgb

    def _project_depth_points_to_color(
        self,
        points_depth: Tensor,
        rgb: Tensor,
        extrinsics: Any,
    ) -> tuple[Tensor, Meta]:
        if extrinsics is None:
            raise ValueError("camera.depth_to_color_extrinsics is required for RGB mapping")
        projection = project_points_to_color_image(
            points_depth,
            extrinsics=extrinsics,
            intrinsics=self.camera.color_intrinsics,
            source_frame="depth",
        )
        in_bounds = projection.valid
        uv = projection.uv

        colors = torch.zeros((points_depth.shape[0], 3), dtype=rgb.dtype, device=rgb.device)
        if bool(in_bounds.any()):
            colors[in_bounds] = rgb[uv[in_bounds, 1], uv[in_bounds, 0], :3]
        return colors, {
            "enabled": True,
            "mapping": "project_depth_to_color",
            "sampling": "nearest",
            "invalid_projection_count": int((~in_bounds).sum().item()),
            "valid_projection_count": int(in_bounds.sum().item()),
        }

    def _resolve_depth(
        self,
        frame: RGBDFrame | StereoIRFrame | dict[str, Any],
    ) -> ResolvedDepth:
        if self.config.depth_source.mode == "frame":
            return ResolvedDepth(
                depth=as_tensor(get_frame_value(frame, "depth"), self.device, torch.float32),
                effective_depth_scale=self.camera.depth_scale,
                intrinsics=self.camera.active_intrinsics,
                depth_to_color_extrinsics=self.camera.depth_to_color_extrinsics,
                frame_name="frame",
            )
        estimator = getattr(self, "depth_estimator", None)
        if estimator is None:
            raise RuntimeError("FFS depth estimator was not initialized")
        result = estimator.infer(frame)
        return ResolvedDepth(
            depth=result.depth_m,
            effective_depth_scale=1.0,
            intrinsics=result.intrinsics,
            depth_to_color_extrinsics=result.depth_to_color_extrinsics,
            frame_name="ffs_stereo",
            metadata=result.metadata,
            disparity_px=result.disparity_px,
            valid_mask=result.valid_mask,
        )
