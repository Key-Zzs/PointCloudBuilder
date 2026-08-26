from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from camera_rig.api import CameraFrame, RigidTransform, StreamFrame, load_camera_bundle
from pointcloud_builder.config import CropConfig, SamplingConfig
from pointcloud_builder.integrations.camera_rig import create_native_builder
from pointcloud_builder.workspace import (
    ExpectedPlaneRegion,
    FramedPointCloud,
    SingleCameraWorkspacePipeline,
    evaluate_expected_plane,
    transform_point_cloud,
)
from pointcloud_builder.integrations.camera_rig.types import FrameExplicitTransform

ROOT = Path(__file__).parents[1]
BUNDLE_FIXTURE = ROOT / "third_party/CameraRig/tests/fixtures/consumer/fixed_camera_bundle_v1.json"


def _plane_bundle():
    bundle = load_camera_bundle(BUNDLE_FIXTURE)
    fixed = bundle.fixed_mount_calibration
    assert fixed is not None
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag([1.0, -1.0, -1.0])
    matrix[2, 3] = 1.0
    transform = RigidTransform(fixed.camera_reference_frame, fixed.parent_frame, matrix)
    return replace(bundle, fixed_mount_calibration=replace(fixed, T_parent_from_camera_reference=transform))


def _depth_frame() -> CameraFrame:
    depth = np.full((3, 4), 1000, dtype=np.uint16)
    color = np.zeros((3, 4, 3), dtype=np.uint8)
    streams = {
        "depth": StreamFrame("depth", depth, 1, 1_000_000_000, "synthetic"),
        "color": StreamFrame("color", color, 1, 1_000_000_000, "synthetic"),
    }
    return CameraFrame(
        camera_name="synthetic_camera",
        serial="SYNTHETIC-CONSUMER-0001",
        streams=streams,
        host_receive_timestamp_ns=1_100_000_000,
    )


def test_torch_transform_preserves_device_dtype_and_xyzrgb_features() -> None:
    points = torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]], dtype=torch.float32)
    matrix = np.eye(4)
    matrix[:3, 3] = [0.5, -1.0, 2.0]
    cloud = FramedPointCloud(points, "depth")
    transformed = transform_point_cloud(cloud, FrameExplicitTransform("depth", "workspace", matrix))
    assert transformed.points.device == points.device
    assert transformed.points.dtype == points.dtype
    torch.testing.assert_close(transformed.points[:, :3], torch.tensor([[1.5, 1.0, 5.0]]))
    torch.testing.assert_close(transformed.points[:, 3:], points[:, 3:])
    with pytest.raises(ValueError, match="does not match transform source"):
        transform_point_cloud(cloud, FrameExplicitTransform("color", "workspace", matrix))


def test_framed_cloud_rejects_bad_shape_empty_frame_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="Nx3 or Nx6"):
        FramedPointCloud(torch.zeros((3, 4)), "depth")
    with pytest.raises(ValueError, match="non-empty"):
        FramedPointCloud(torch.zeros((3, 3)), "")
    points = torch.zeros((3, 3))
    points[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        FramedPointCloud(points, "depth")


def test_synthetic_native_depth_plane_deprojection_transform_and_metric_scale() -> None:
    context = create_native_builder(
        _plane_bundle(),
        device="cpu",
        sampling=SamplingConfig(mode="stride", num_points=12, deterministic=True, seed=7),
    )
    pipeline = SingleCameraWorkspacePipeline(
        context,
        workspace_crop=CropConfig(
            enabled=False,
            x=(-10.0, 10.0),
            y=(-10.0, 10.0),
            z=(-10.0, 10.0),
            frame="workspace",
        ),
    )
    result = pipeline.process(_depth_frame())
    assert result.camera_raw.frame == "synthetic_camera/depth_optical"
    assert result.workspace_raw.frame == "workspace"
    assert result.camera_raw.points.shape == (12, 3)
    torch.testing.assert_close(result.camera_raw.points[:, 2], torch.ones(12))
    torch.testing.assert_close(result.workspace_raw.points[:, 2], torch.zeros(12), atol=1e-6, rtol=0)
    metrics = evaluate_expected_plane(
        result.workspace_raw,
        ExpectedPlaneRegion(
            frame="workspace",
            x=(-1.0, 1.0),
            y=(-1.0, 1.0),
            expected_z_m=0.0,
            z_search_range_m=(-0.01, 0.01),
        ),
    )
    assert metrics.point_count == 12
    assert metrics.median_abs_z_m <= 1e-6
    assert metrics.p95_abs_z_m <= 1e-6
    assert metrics.normal_angle_to_expected_deg <= 1e-3
    assert result.workspace_sampled.points.shape == (12, 3)


def test_workspace_crop_happens_after_transform() -> None:
    context = create_native_builder(
        _plane_bundle(),
        device="cpu",
        sampling=SamplingConfig(mode="stride", num_points=4, enabled=False),
    )
    pipeline = SingleCameraWorkspacePipeline(
        context,
        workspace_crop=CropConfig(
            enabled=True,
            x=(-0.2, 0.2),
            y=(-0.2, 0.2),
            z=(-0.01, 0.01),
            frame="workspace",
        ),
    )
    result = pipeline.process(_depth_frame())
    assert 0 < result.workspace_cropped.points.shape[0] < result.workspace_raw.points.shape[0]
    assert result.workspace_cropped.metadata["workspace_crop"]["input_count"] == 12
