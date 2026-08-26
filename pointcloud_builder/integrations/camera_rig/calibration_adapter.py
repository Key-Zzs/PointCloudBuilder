"""Convert passed CameraRig bundle calibration without losing frame semantics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pointcloud_builder.camera_model import CameraIntrinsics as PCBIntrinsics
from pointcloud_builder.integrations.camera_rig.dependencies import (
    CameraBundle,
    CameraIntrinsics,
    RigidTransform,
)
from pointcloud_builder.integrations.camera_rig.transform_resolver import resolve_transform
from pointcloud_builder.integrations.camera_rig.types import (
    CameraRigCalibration,
    FrameExplicitTransform,
)
from pointcloud_builder.integrations.camera_rig.validation import validate_passed_fixed_bundle


def camera_intrinsics_to_pcb(intrinsics: CameraIntrinsics) -> PCBIntrinsics:
    """Copy pinhole values while frame metadata remains in CameraRigCalibration."""

    return PCBIntrinsics(
        width=int(intrinsics.width),
        height=int(intrinsics.height),
        fx=float(intrinsics.fx),
        fy=float(intrinsics.fy),
        cx=float(intrinsics.cx),
        cy=float(intrinsics.cy),
    )


def rigid_transform_to_frame_explicit(transform: RigidTransform) -> FrameExplicitTransform:
    return FrameExplicitTransform(
        source_frame=transform.source_frame,
        target_frame=transform.target_frame,
        matrix=transform.matrix,
    )


def calibration_from_camera_bundle(
    bundle: CameraBundle,
    *,
    camera_name: str | None = None,
    required_streams: Iterable[str] = ("color", "depth", "ir_left", "ir_right"),
) -> CameraRigCalibration:
    """Validate one fixed bundle and resolve all required stream/workspace transforms."""

    validate_passed_fixed_bundle(bundle, camera_name)
    fixed = bundle.fixed_mount_calibration
    assert fixed is not None
    required = tuple(required_streams)
    missing = sorted(set(required) - set(bundle.intrinsics))
    if missing:
        raise ValueError(f"CameraBundle is missing required intrinsics: {missing}")
    transforms = tuple(bundle.internal_transforms) + (fixed.T_parent_from_camera_reference,)
    resolved: dict[str, FrameExplicitTransform] = {}
    frames = {name: value.frame for name, value in bundle.intrinsics.items()}

    relevant_frames = sorted(set(frames.values()) | {fixed.parent_frame})
    for source in relevant_frames:
        for target in relevant_frames:
            if source == target:
                continue
            try:
                transform = resolve_transform(transforms, source, target)
            except ValueError:
                continue
            resolved[f"{target}<-{source}"] = rigid_transform_to_frame_explicit(transform)

    for stream_name in required:
        source_frame = frames[stream_name]
        key = f"{fixed.parent_frame}<-{source_frame}"
        if key not in resolved:
            raise ValueError(
                f"CameraBundle has no transform from {stream_name!r} frame "
                f"{source_frame!r} to workspace frame {fixed.parent_frame!r}"
            )
    actual_name = str(getattr(bundle.device, "camera_name"))
    return CameraRigCalibration(
        camera_name=actual_name,
        workspace_frame=fixed.parent_frame,
        camera_reference_frame=fixed.camera_reference_frame,
        depth_scale_m_per_unit=float(bundle.depth_scale_m_per_unit),
        intrinsics={name: camera_intrinsics_to_pcb(value) for name, value in bundle.intrinsics.items()},
        intrinsic_frames=frames,
        transforms=resolved,
        bundle=bundle,
    )


def resolve_bundle_transform(
    bundle: CameraBundle,
    source_frame: str,
    target_frame: str,
) -> RigidTransform:
    """Resolve a public CameraRig transform directly from one passed fixed bundle."""

    validate_passed_fixed_bundle(bundle)
    fixed = bundle.fixed_mount_calibration
    assert fixed is not None
    transforms = tuple(bundle.internal_transforms) + (fixed.T_parent_from_camera_reference,)
    return resolve_transform(transforms, source_frame, target_frame)
