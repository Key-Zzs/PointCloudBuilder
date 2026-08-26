"""Frame-explicit types owned by the CameraRig consumer integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pointcloud_builder.camera_model import CameraExtrinsics, CameraIntrinsics


@dataclass(frozen=True)
class FrameExplicitTransform:
    """A validated ``T_target_from_source`` without discarding frame names."""

    source_frame: str
    target_frame: str
    matrix: np.ndarray

    def __post_init__(self) -> None:
        if not self.source_frame.strip() or not self.target_frame.strip():
            raise ValueError("source_frame and target_frame must be non-empty")
        matrix = np.asarray(self.matrix, dtype=np.float64).copy()
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("frame-explicit transform must be a finite 4x4 matrix")
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9, rtol=0.0):
            raise ValueError("frame-explicit transform must have a homogeneous last row")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)

    @property
    def extrinsics(self) -> CameraExtrinsics:
        """Return the numeric PCB extrinsics while this wrapper retains frames."""

        rotation = tuple(tuple(float(value) for value in row) for row in self.matrix[:3, :3])
        translation = tuple(float(value) for value in self.matrix[:3, 3])
        return CameraExtrinsics(rotation=rotation, translation=translation)  # type: ignore[arg-type]


@dataclass(frozen=True)
class CameraRigCalibration:
    """PCB-ready calibration with the original frame-explicit CameraRig geometry."""

    camera_name: str
    workspace_frame: str
    camera_reference_frame: str
    depth_scale_m_per_unit: float
    intrinsics: dict[str, CameraIntrinsics]
    intrinsic_frames: dict[str, str]
    transforms: dict[str, FrameExplicitTransform]
    bundle: Any

    def transform(self, source_frame: str, target_frame: str) -> FrameExplicitTransform:
        key = f"{target_frame}<-{source_frame}"
        try:
            return self.transforms[key]
        except KeyError as error:
            raise KeyError(
                f"CameraRig calibration has no resolved transform {target_frame}<-{source_frame}"
            ) from error


@dataclass(frozen=True)
class CameraRigBuilderContext:
    """A builder paired with its authoritative source/workspace frame contract."""

    builder: Any
    calibration: CameraRigCalibration
    source_frame: str
    workspace_frame: str
    T_workspace_from_source: FrameExplicitTransform
    depth_mode: str
    frame_adapter: Any
