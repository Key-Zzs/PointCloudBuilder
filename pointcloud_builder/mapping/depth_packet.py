"""Build strict per-camera depth observations from the same deprojection pass."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from pointcloud_builder.mapping.types import RigDepthObservation


def provision_identity_sha256(path: str | Path) -> str:
    """Hash the validated provision manifest, or a legacy single-file bundle."""

    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "manifest.json"
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bundle_sha256(bundle: Any) -> str:
    """Return a privacy-safe content identity when no artifact path is available."""

    payload = json.dumps(
        bundle.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def observation_from_same_pass(
    *,
    mapping: dict[str, Any],
    resolved_depth: Any,
    context: Any,
    provision_sha256: str,
) -> RigDepthObservation:
    """Copy only the depth geometry consumed by TSDF; never derive it from clouds."""

    if context.depth_mode == "native":
        depth = mapping.get("depth")
        if not isinstance(depth, np.ndarray) or depth.dtype != np.uint16:
            raise TypeError("native TSDF observation requires original uint16 depth")
        depth = depth
        depth_unit = "raw_units"
        scale = float(context.calibration.depth_scale_m_per_unit)
        stream_name = "depth"
        rectified = False
    elif context.depth_mode == "ffs_stereo":
        tensor = resolved_depth.depth.detach()
        depth = np.ascontiguousarray(tensor.to(device="cpu").numpy(), dtype=np.float32)
        depth_unit = "meters"
        scale = 1.0
        stream_name = "ir_left"
        rectified = True
    else:
        raise ValueError(f"unsupported rig depth mode: {context.depth_mode!r}")
    valid = np.isfinite(depth) & (depth > 0)
    bundle_intrinsics = context.calibration.bundle.intrinsics[stream_name]
    return RigDepthObservation(
        camera_name=str(mapping["camera_name"]),
        depth=depth,
        depth_unit=depth_unit,
        depth_scale_m_per_unit=scale,
        valid_mask=valid,
        intrinsics=resolved_depth.intrinsics,
        T_workspace_from_camera=context.T_workspace_from_source.matrix,
        timestamp_ns=int(mapping["timestamp_ns"]),
        depth_source=context.depth_mode,
        source_frame=context.source_frame,
        workspace_frame=context.workspace_frame,
        bundle_identity=str(context.calibration.bundle.bundle_id),
        provision_sha256=provision_sha256,
        distortion_model=str(bundle_intrinsics.distortion_model),
        distortion_coeffs=tuple(float(x) for x in bundle_intrinsics.distortion_coeffs),
        rectified=rectified,
    )
