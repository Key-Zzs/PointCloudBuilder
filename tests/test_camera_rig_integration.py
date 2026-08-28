from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from camera_rig.api import (
    CameraFrame,
    CameraIntrinsics,
    RigidTransform,
    StreamFrame,
    load_camera_bundle,
)

from pointcloud_builder.integrations.camera_rig import (
    CameraRigFrameAdapter,
    TransformResolutionError,
    calibration_from_camera_bundle,
    create_native_builder,
    resolve_transform,
)
from pointcloud_builder.projection_parity import _audit_propagation

ROOT = Path(__file__).parents[1]
BUNDLE_FIXTURE = (
    ROOT / "third_party/CameraRig/tests/fixtures/consumer/fixed_camera_bundle_v1.json"
)


def _bundle():
    return load_camera_bundle(BUNDLE_FIXTURE)


def _transform(source: str, target: str, translation: tuple[float, float, float]):
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    return RigidTransform(source, target, matrix)


def _camera_frame(*, omit: str | None = None) -> CameraFrame:
    arrays = {
        "color": np.asarray([[[255, 1, 2], [3, 4, 5]]], dtype=np.uint8),
        "depth": np.asarray([[1000, 2000]], dtype=np.uint16),
        "ir_left": np.asarray([[7, 8]], dtype=np.uint8),
        "ir_right": np.asarray([[9, 10]], dtype=np.uint8),
    }
    streams = {
        name: StreamFrame(
            stream_name=name,
            data=data,
            frame_number=index + 10,
            sensor_timestamp_ns=1_000_000_000 + index,
            timestamp_domain="hardware_clock",
        )
        for index, (name, data) in enumerate(arrays.items())
        if name != omit
    }
    return CameraFrame(
        camera_name="synthetic_camera",
        serial="SYNTHETIC-CONSUMER-0001",
        streams=streams,
        host_receive_timestamp_ns=2_000_000_000,
    )


def test_frame_adapter_preserves_rgb_depth_ir_and_timing_without_copy() -> None:
    frame = _camera_frame()
    adapted = CameraRigFrameAdapter(
        _bundle(),
        required_streams=("color", "depth", "ir_left", "ir_right"),
        timestamp_stream="depth",
    ).adapt(frame)
    assert adapted["rgb"] is frame.color.data
    assert adapted["rgb"][0, 0].tolist() == [255, 1, 2]
    assert adapted["depth"] is frame.depth.data
    assert adapted["depth"].dtype == np.uint16
    assert adapted["left_ir"].dtype == adapted["right_ir"].dtype == np.uint8
    assert adapted["timestamp_ns"] == frame.depth.sensor_timestamp_ns
    assert adapted["camera_name"] == "synthetic_camera"
    assert adapted["frame_numbers"]["ir_left"] == 12
    assert adapted["stream_timestamp_domains"]["depth"] == "hardware_clock"


def test_frame_adapter_fails_closed_for_pipeline_required_stream() -> None:
    with pytest.raises(ValueError, match="missing required streams"):
        CameraRigFrameAdapter(
            _bundle(), required_streams=("ir_left", "ir_right")
        ).adapt(_camera_frame(omit="ir_right"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("camera_name", "another_camera", "camera_name"),
        ("serial", "ANOTHER-SYNTHETIC-SERIAL", "serial"),
    ],
)
def test_frame_adapter_binds_frame_identity_to_bundle(
    field: str, value: str, message: str
) -> None:
    frame = replace(_camera_frame(), **{field: value})
    with pytest.raises(ValueError, match=message):
        CameraRigFrameAdapter(_bundle(), required_streams=("depth",)).adapt(frame)


def test_transform_resolver_inverse_multihop_direction_and_determinism() -> None:
    transforms = [
        _transform("source", "b", (20.0, 0.0, 0.0)),
        _transform("b", "workspace", (-10.0, 0.0, 0.0)),
        _transform("source", "a", (10.0, 0.0, 0.0)),
        _transform("a", "workspace", (0.0, 0.0, 0.0)),
    ]
    forward = resolve_transform(transforms, "source", "workspace")
    reverse = resolve_transform(transforms, "workspace", "source")
    assert (forward.source_frame, forward.target_frame) == ("source", "workspace")
    np.testing.assert_allclose(forward.matrix[:3, 3], [10.0, 0.0, 0.0])
    np.testing.assert_allclose(reverse.matrix, forward.inverse().matrix)
    permuted = resolve_transform(list(reversed(transforms)), "source", "workspace")
    np.testing.assert_allclose(permuted.matrix, forward.matrix)


def test_transform_resolver_rejects_duplicate_conflict_and_missing_path() -> None:
    edge = _transform("a", "b", (1.0, 0.0, 0.0))
    with pytest.raises(TransformResolutionError, match="duplicate"):
        resolve_transform([edge, edge], "a", "b")
    with pytest.raises(TransformResolutionError, match="conflicting"):
        resolve_transform([edge, _transform("a", "b", (2.0, 0.0, 0.0))], "a", "b")
    with pytest.raises(TransformResolutionError, match="conflicting transform paths"):
        resolve_transform(
            [
                _transform("source", "a", (1.0, 0.0, 0.0)),
                _transform("a", "workspace", (0.0, 0.0, 0.0)),
                _transform("source", "b", (2.0, 0.0, 0.0)),
                _transform("b", "workspace", (0.0, 0.0, 0.0)),
            ],
            "source",
            "workspace",
        )
    with pytest.raises(TransformResolutionError, match="conflicting transform"):
        resolve_transform(
            [
                _transform("source", "a", (1.0, 0.0, 0.0)),
                _transform("a", "b", (1.0, 0.0, 0.0)),
                _transform("b", "source", (-1.5, 0.0, 0.0)),
            ],
            "source",
            "b",
        )
    with pytest.raises(TransformResolutionError, match="no transform path"):
        resolve_transform([edge], "a", "workspace")


def test_calibration_adapter_preserves_nonidentity_workspace_directions() -> None:
    calibration = calibration_from_camera_bundle(_bundle())
    source_intrinsics = _bundle().intrinsics["color"]
    converted_intrinsics = calibration.intrinsics["color"]
    assert converted_intrinsics.frame == source_intrinsics.frame
    assert converted_intrinsics.distortion_model == source_intrinsics.distortion_model
    assert converted_intrinsics.distortion_coeffs == source_intrinsics.distortion_coeffs
    assert converted_intrinsics.pixel_geometry == "raw"
    reference = calibration.intrinsic_frames["ir_left"]
    depth = calibration.intrinsic_frames["depth"]
    workspace_from_reference = calibration.transform(reference, "workspace")
    workspace_from_depth = calibration.transform(depth, "workspace")
    assert workspace_from_reference.source_frame == reference
    assert workspace_from_reference.target_frame == "workspace"
    assert not np.allclose(workspace_from_reference.matrix, np.eye(4))
    point_depth = np.asarray([0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(
        workspace_from_depth.matrix @ point_depth,
        [0.5, -0.26, 2.0, 1.0],
    )
    assert calibration.transform("workspace", depth).target_frame == depth


def test_projection_propagation_gate_detects_adapter_mutation() -> None:
    bundle = _bundle()
    calibration = calibration_from_camera_bundle(bundle)
    intrinsics = dict(calibration.intrinsics)
    intrinsics["color"] = replace(
        intrinsics["color"], fx=intrinsics["color"].fx + 1.0
    )
    mutated = replace(calibration, intrinsics=intrinsics)
    report = _audit_propagation(bundle, mutated)
    assert report["status"] == "FAIL"
    assert report["per_stream"]["color"]["checks"]["fx"] is False


def test_native_factory_uses_bundle_scale_intrinsics_and_frames() -> None:
    context = create_native_builder(_bundle(), device="cpu")
    assert context.depth_mode == "native"
    assert context.source_frame == "synthetic_camera/depth_optical"
    assert context.workspace_frame == "workspace"
    assert context.builder.config.camera.depth_scale == pytest.approx(0.001)
    assert context.builder.config.camera.depth_intrinsics.fx == pytest.approx(3.0)
    assert context.builder.config.camera.aligned_depth_to_color is False
    assert context.builder.config.pointcloud.use_rgb is False
    assert context.T_workspace_from_source.source_frame == context.source_frame
    assert context.T_workspace_from_source.target_frame == "workspace"

    rgb_context = create_native_builder(_bundle(), device="cpu", use_rgb=True)
    assert rgb_context.builder.config.pointcloud.use_rgb is True
    assert rgb_context.builder.config.pointcloud.output_format == "xyzrgb"
    assert rgb_context.builder.config.pointcloud.rgb_mapping == "project_depth_to_color"
    assert rgb_context.frame_adapter.required_streams == ("color", "depth")


def test_bundle_validation_failed_missing_mount_name_and_frame_mismatch() -> None:
    bundle = _bundle()
    with pytest.raises(ValueError, match="status"):
        calibration_from_camera_bundle(replace(bundle, status="failed"))
    with pytest.raises(ValueError, match="fixed_mount"):
        calibration_from_camera_bundle(replace(bundle, fixed_mount_calibration=None))
    with pytest.raises(ValueError, match="camera_name mismatch"):
        calibration_from_camera_bundle(bundle, camera_name="another_camera")
    disconnected = CameraIntrinsics(
        frame="synthetic_camera/disconnected_depth",
        width=4,
        height=3,
        fx=3.0,
        fy=3.0,
        cx=1.5,
        cy=1.0,
        distortion_model="none",
    )
    intrinsics = dict(bundle.intrinsics)
    intrinsics["depth"] = disconnected
    with pytest.raises(ValueError, match="no transform"):
        calibration_from_camera_bundle(replace(bundle, intrinsics=intrinsics))


def test_integration_imports_only_camera_rig_stable_api() -> None:
    integration = ROOT / "pointcloud_builder/integrations/camera_rig"
    violations: list[str] = []
    for path in sorted(integration.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("camera_rig")
                and node.module != "camera_rig.api"
            ):
                violations.append(f"{path.name}:{node.lineno}:{node.module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if (
                        alias.name.startswith("camera_rig")
                        and alias.name != "camera_rig.api"
                    ):
                        violations.append(f"{path.name}:{node.lineno}:{alias.name}")
    assert violations == []


def test_core_import_isolated_from_missing_camera_rig_dependency() -> None:
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith('camera_rig'):
        raise ImportError('simulated missing CameraRig')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import pointcloud_builder
print('PASS')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "PASS"


def test_integration_missing_dependency_error_is_actionable() -> None:
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith('camera_rig'):
        raise ImportError('simulated missing CameraRig')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
try:
    import pointcloud_builder.integrations.camera_rig
except ImportError as error:
    assert error.__class__.__name__ == 'CameraRigDependencyError'
    assert 'third_party/CameraRig' in str(error)
    print('PASS')
else:
    raise AssertionError('integration import unexpectedly succeeded')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "PASS"
