"""Deterministic analytic plane-and-box rig scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from camera_rig.api import CameraBundle, CameraFrame, StreamFrame

from pointcloud_builder.config import SamplingConfig
from pointcloud_builder.integrations.camera_rig import create_native_builder
from pointcloud_builder.rig.config import RigConfig
from pointcloud_builder.rig.pipeline import OfflineRigPipeline, RigCameraRuntime
from pointcloud_builder.rig.sources import SyntheticCameraSource
from pointcloud_builder.workspace import SingleCameraWorkspacePipeline


@dataclass(frozen=True)
class SyntheticScene:
    bundles: dict[str, CameraBundle]
    frames: dict[str, tuple[CameraFrame, ...]]
    poses: dict[str, np.ndarray]
    plane_z_m: float = 0.0
    box_min: tuple[float, float, float] = (-0.18, -0.12, 0.0)
    box_max: tuple[float, float, float] = (0.18, 0.12, 0.25)


def create_synthetic_scene(
    camera_names: tuple[str, ...] = ("camera_a", "camera_b", "camera_c"),
    *,
    frame_count: int = 3,
    timestamp_offsets_ns: dict[str, int] | None = None,
    width: int = 96,
    height: int = 72,
) -> SyntheticScene:
    """Render each camera from its own optical pose; no depth image is copied."""

    if not 1 <= len(camera_names) <= 3:
        raise ValueError("synthetic scene supports one to three cameras")
    positions = (
        np.array((-0.62, -0.58, 0.88), dtype=np.float64),
        np.array((0.64, -0.52, 0.84), dtype=np.float64),
        np.array((0.02, 0.68, 0.96), dtype=np.float64),
    )
    target = np.array((0.0, 0.0, 0.08), dtype=np.float64)
    offsets = timestamp_offsets_ns or {}
    bundles: dict[str, CameraBundle] = {}
    frames: dict[str, tuple[CameraFrame, ...]] = {}
    poses: dict[str, np.ndarray] = {}
    for camera_index, name in enumerate(camera_names):
        pose = _look_at_optical(positions[camera_index], target)
        poses[name] = pose
        bundle = CameraBundle.from_dict(_bundle_dict(name, pose, width, height))
        bundles[name] = bundle
        depth = _render_depth(pose, width, height, fx=82.0, fy=82.0)
        camera_frames = []
        for frame_index in range(frame_count):
            timestamp = 1_000_000_000 + frame_index * 33_333_333 + int(offsets.get(name, 0))
            camera_frames.append(
                CameraFrame(
                    camera_name=name,
                    serial=f"SYNTHETIC-{name.upper()}",
                    streams={
                        "depth": StreamFrame(
                            stream_name="depth",
                            data=depth.copy(),
                            frame_number=frame_index,
                            sensor_timestamp_ns=timestamp,
                            timestamp_domain="synthetic-device-clock",
                        )
                    },
                    host_receive_timestamp_ns=timestamp,
                )
            )
        frames[name] = tuple(camera_frames)
    return SyntheticScene(bundles=bundles, frames=frames, poses=poses)


def build_synthetic_rig(config: RigConfig, scene: SyntheticScene) -> OfflineRigPipeline:
    no_sampling = SamplingConfig(mode="stride", num_points=1, enabled=False)
    runtimes: dict[str, RigCameraRuntime] = {}
    for camera in config.enabled_cameras:
        if camera.source.type != "synthetic":
            raise ValueError(f"camera {camera.name!r} is not configured as synthetic")
        try:
            bundle = scene.bundles[camera.name]
            frames = scene.frames[camera.name]
        except KeyError as error:
            raise ValueError(f"missing bundle for camera {camera.name!r}") from error
        context = create_native_builder(
            bundle,
            camera_name=camera.name,
            device="cpu",
            crop=camera.local_crop,
            sampling=no_sampling,
        )
        if context.workspace_frame != config.output_frame:
            raise ValueError(
                f"camera {camera.name!r} output frame differs from bundle parent frame"
            )
        runtimes[camera.name] = RigCameraRuntime(
            source=SyntheticCameraSource(camera.name, frames),
            pipeline=SingleCameraWorkspacePipeline(context, workspace_crop=config.workspace_crop),
            provenance={
                "source_type": "synthetic",
                "scene": "analytic_plane_box_v1",
                "depth_mode": "native",
                "bundle_id": bundle.bundle_id,
            },
        )
    return OfflineRigPipeline(config, runtimes)


def _look_at_optical(origin: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - origin
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack((right, down, forward))
    matrix[:3, 3] = origin
    return matrix


def _render_depth(
    T_workspace_from_camera: np.ndarray,
    width: int,
    height: int,
    *,
    fx: float,
    fy: float,
) -> np.ndarray:
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    rays_camera = np.stack(((u - cx) / fx, (v - cy) / fy, np.ones_like(u)), axis=-1)
    directions = rays_camera @ T_workspace_from_camera[:3, :3].T
    origin = T_workspace_from_camera[:3, 3]
    plane_t = np.full((height, width), np.inf, dtype=np.float64)
    downward = directions[..., 2] < -1e-12
    plane_t[downward] = -origin[2] / directions[..., 2][downward]
    box_t = _ray_box_entry(
        origin,
        directions,
        np.array((-0.18, -0.12, 0.0)),
        np.array((0.18, 0.12, 0.25)),
    )
    depth_m = np.minimum(plane_t, box_t)
    depth_m[~np.isfinite(depth_m) | (depth_m <= 0)] = 0.0
    return np.rint(depth_m / 0.001).clip(0, np.iinfo(np.uint16).max).astype(np.uint16)


def _ray_box_entry(
    origin: np.ndarray,
    directions: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    safe = np.where(np.abs(directions) < 1e-12, np.nan, directions)
    first = (lower - origin) / safe
    second = (upper - origin) / safe
    t_near = np.nanmax(np.minimum(first, second), axis=-1)
    t_far = np.nanmin(np.maximum(first, second), axis=-1)
    hit = (t_far >= np.maximum(t_near, 0.0)) & (t_near > 0.0)
    return np.where(hit, t_near, np.inf)


def _bundle_dict(name: str, pose: np.ndarray, width: int, height: int) -> dict[str, Any]:
    frames = {stream: f"{name}/{stream}_optical" for stream in ("color", "depth", "ir_left", "ir_right")}
    profiles = {}
    intrinsics = {}
    formats = {"color": "rgb8", "depth": "z16", "ir_left": "y8", "ir_right": "y8"}
    for index, stream in enumerate(("color", "depth", "ir_left", "ir_right")):
        profiles[stream] = {
            "stream_name": stream,
            "width": width,
            "height": height,
            "fps": 30,
            "format": formats[stream],
            "index": 0 if stream not in {"ir_left", "ir_right"} else index - 1,
            "sensor_identifier": f"synthetic-{stream}",
        }
        intrinsics[stream] = {
            "width": width,
            "height": height,
            "fx": 82.0,
            "fy": 82.0,
            "cx": (width - 1) / 2.0,
            "cy": (height - 1) / 2.0,
            "distortion_model": "none",
            "distortion_coeffs": [],
            "frame": frames[stream],
        }
    identity_edges = []
    for stream in ("color", "ir_left", "ir_right"):
        identity_edges.append(
            {
                "source_frame": frames["depth"],
                "target_frame": frames[stream],
                "matrix": np.eye(4).tolist(),
            }
        )
    quality = {
        "passed": True,
        "metrics": {"analytic_render": True},
        "thresholds": {},
        "failure_reasons": [],
        "warnings": [],
    }
    return {
        "schema_version": "camera-rig.bundle.v1",
        "status": "passed",
        "bundle_id": f"synthetic-rig-{name}",
        "created_at": "2026-08-26T00:00:00Z",
        "coordinate_convention": {
            "vector": "column",
            "handedness": "right-handed",
            "length_unit": "meter",
            "time_unit": "nanosecond",
            "transform_naming": "T_target_from_source",
        },
        "device": {
            "camera_name": name,
            "serial": f"SYNTHETIC-{name.upper()}",
            "expected_model": "Synthetic Pinhole v1",
            "reported_model": "Synthetic Pinhole v1",
            "canonical_model": "Synthetic Pinhole v1",
            "product_id": "SYNTHETIC",
            "product_line": "synthetic",
            "firmware_version": "synthetic",
            "physical_port": f"synthetic-{name}",
            "usb_type": "synthetic",
            "sdk_version": "synthetic",
            "driver": "synthetic",
            "metadata": {"real_hardware": False},
        },
        "stream_profiles": profiles,
        "intrinsics": intrinsics,
        "internal_transforms": identity_edges,
        "depth_scale_m_per_unit": 0.001,
        "fixed_mount_calibration": {
            "mount_type": "fixed",
            "parent_frame": "workspace",
            "camera_reference_frame": frames["depth"],
            "T_parent_from_camera_reference": {
                "source_frame": frames["depth"],
                "target_frame": "workspace",
                "matrix": pose.tolist(),
            },
            "quality": quality,
            "provenance": {"real_hardware": False, "source": "analytic_plane_box_v1"},
        },
        "quality": quality,
        "provenance": {"real_hardware": False, "source": "analytic_plane_box_v1"},
    }
