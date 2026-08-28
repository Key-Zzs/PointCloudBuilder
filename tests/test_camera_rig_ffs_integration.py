from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import torch
from camera_rig.api import (
    CameraFrame,
    CameraIntrinsics,
    StreamFrame,
    load_camera_bundle,
)

from pointcloud_builder.config import CropConfig, FFSConfig, SamplingConfig
from pointcloud_builder.integrations.camera_rig import (
    create_ffs_builder,
    ffs_calibration_from_camera_bundle,
)
from pointcloud_builder.projection_parity import _audit_ffs_contract
from pointcloud_builder.workspace import SingleCameraWorkspacePipeline

ROOT = Path(__file__).parents[1]
BUNDLE_FIXTURE = (
    ROOT / "third_party/CameraRig/tests/fixtures/consumer/fixed_camera_bundle_v1.json"
)


class ConstantDisparityBackend:
    name = "synthetic"
    provenance: ClassVar[dict[str, bool]] = {"synthetic": True}
    last_timing_ms: ClassVar[dict[str, float]] = {}

    def __init__(self, disparity: float) -> None:
        self.disparity = disparity

    def infer_disparity(
        self, left_ir: torch.Tensor, right_ir: torch.Tensor
    ) -> torch.Tensor:
        assert left_ir.shape == right_ir.shape == (1, 3, 480, 640)
        return torch.full((1, 480, 640), self.disparity, device=left_ir.device)


def _full_size_bundle():
    bundle = load_camera_bundle(BUNDLE_FIXTURE)
    intrinsics = {
        name: replace(
            value, width=640, height=480, fx=400.0, fy=400.0, cx=319.5, cy=239.5
        )
        for name, value in bundle.intrinsics.items()
    }
    profiles = {
        name: replace(value, width=640, height=480)
        for name, value in bundle.stream_profiles.items()
    }
    return replace(bundle, intrinsics=intrinsics, stream_profiles=profiles)


def _ffs_config() -> FFSConfig:
    return FFSConfig(
        backend="pytorch",
        width=640,
        height=480,
        baseline_m=999.0,
        rectification_mode="opencv",
        max_disp=192,
        valid_iters=8,
    )


def _stereo_frame(*, with_color: bool = False) -> CameraFrame:
    left = np.full((480, 640), 127, dtype=np.uint8)
    right = np.full((480, 640), 125, dtype=np.uint8)
    streams = {
        "ir_left": StreamFrame("ir_left", left, 10, 1_000_000_000, "synthetic"),
        "ir_right": StreamFrame("ir_right", right, 10, 1_000_000_000, "synthetic"),
    }
    if with_color:
        color = np.empty((480, 640, 3), dtype=np.uint8)
        color[..., 0] = 32
        color[..., 1] = 128
        color[..., 2] = 255
        streams["color"] = StreamFrame("color", color, 10, 1_000_000_000, "synthetic")
    return CameraFrame(
        camera_name="synthetic_camera",
        serial="SYNTHETIC-CONSUMER-0001",
        streams=streams,
        host_receive_timestamp_ns=1_100_000_000,
    )


def test_bundle_ffs_calibration_uses_frames_intrinsics_distortion_and_norm_baseline() -> (
    None
):
    calibration = ffs_calibration_from_camera_bundle(load_camera_bundle(BUNDLE_FIXTURE))
    assert calibration.baseline_m == pytest.approx(0.05)
    assert calibration.left_intrinsics.fx == pytest.approx(3.0)
    assert calibration.right_intrinsics.fx == pytest.approx(3.0)
    assert calibration.left_to_right.translation == pytest.approx((-0.05, 0.0, 0.0))
    assert calibration.rectification_identity is True
    assert calibration.left_intrinsics.pixel_geometry == "rectified"
    assert calibration.left_intrinsics.distortion_model == "none"
    assert calibration.left_intrinsics.distortion_coeffs == ()


def test_bundle_ffs_calibration_fails_strict_gate_without_fabrication() -> None:
    bundle = load_camera_bundle(BUNDLE_FIXTURE)
    intrinsics = dict(bundle.intrinsics)
    right = intrinsics["ir_right"]
    intrinsics["ir_right"] = CameraIntrinsics(
        frame=right.frame,
        width=right.width,
        height=right.height,
        fx=right.fx + 0.25,
        fy=right.fy,
        cx=right.cx,
        cy=right.cy,
        distortion_model=right.distortion_model,
        distortion_coeffs=right.distortion_coeffs,
    )
    with pytest.raises(ValueError, match="requires rectified IR input"):
        ffs_calibration_from_camera_bundle(replace(bundle, intrinsics=intrinsics))


def test_ffs_parity_gate_binds_derived_model_to_camera_rig_source() -> None:
    bundle = load_camera_bundle(BUNDLE_FIXTURE)
    calibration = ffs_calibration_from_camera_bundle(bundle)
    mutated = replace(
        calibration,
        left_intrinsics=replace(
            calibration.left_intrinsics,
            fx=calibration.left_intrinsics.fx + 50.0,
        ),
    )
    report = _audit_ffs_contract(mutated, bundle)
    assert report["status"] == "FAIL"
    assert report["source_to_rectified_left_match"] is False


def test_ffs_factory_injects_bundle_calibration_and_emits_ir_left_workspace_cloud() -> (
    None
):
    bundle = _full_size_bundle()
    calibration = ffs_calibration_from_camera_bundle(bundle)
    disparity_for_one_meter = calibration.left_intrinsics.fx * calibration.baseline_m
    context = create_ffs_builder(
        bundle,
        ffs_config=_ffs_config(),
        device="cpu",
        sampling=SamplingConfig(mode="stride", num_points=64),
        backend=ConstantDisparityBackend(disparity_for_one_meter),
    )
    assert context.builder.depth_estimator.calibration is not None
    assert context.builder.depth_estimator.calibration.baseline_m == pytest.approx(0.05)
    assert context.builder.depth_estimator.calibration.baseline_m != 999.0
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
    result = pipeline.process(_stereo_frame())
    assert result.camera_raw.frame == "synthetic_camera/ir_left_optical"
    assert result.workspace_raw.frame == "workspace"
    assert result.metadata["depth_mode"] == "ffs_stereo"
    assert result.camera_raw.points.shape[1] == 3
    assert torch.median(result.camera_raw.points[:, 2]).item() == pytest.approx(
        1.0, abs=1e-6
    )


def test_ffs_factory_projects_color_and_preserves_xyzrgb_to_workspace() -> None:
    bundle = _full_size_bundle()
    calibration = ffs_calibration_from_camera_bundle(bundle)
    disparity_for_one_meter = calibration.left_intrinsics.fx * calibration.baseline_m
    context = create_ffs_builder(
        bundle,
        ffs_config=_ffs_config(),
        device="cpu",
        sampling=SamplingConfig(mode="stride", num_points=64),
        backend=ConstantDisparityBackend(disparity_for_one_meter),
        use_rgb=True,
    )
    assert context.builder.config.pointcloud.output_format == "xyzrgb"
    assert context.builder.config.pointcloud.rgb_mapping == "project_depth_to_color"
    assert context.frame_adapter.required_streams == ("color", "ir_left", "ir_right")
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
    result = pipeline.process(_stereo_frame(with_color=True))
    assert result.camera_raw.points.shape[1] == 6
    assert result.workspace_raw.points.shape[1] == 6
    expected_rgb = torch.tensor([32.0, 128.0, 255.0]) / 255.0
    colored = result.camera_raw.points[:, 3:].sum(dim=1) > 0
    assert bool(colored.any())
    torch.testing.assert_close(
        result.camera_raw.points[colored][0, 3:], expected_rgb, atol=1e-6, rtol=0
    )

    with pytest.raises(ValueError, match="missing required streams"):
        pipeline.process(_stereo_frame())
