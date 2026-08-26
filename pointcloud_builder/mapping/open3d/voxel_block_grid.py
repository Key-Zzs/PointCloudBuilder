"""Lifecycle-managed geometry-only Open3D VoxelBlockGrid backend."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
from typing import Any

import numpy as np

from pointcloud_builder.mapping.config import TsdfMapConfig
from pointcloud_builder.mapping.open3d.camera_model import (
    T_camera_from_workspace,
    intrinsic_matrix,
    open3d_depth_scale,
)
from pointcloud_builder.mapping.open3d.dependencies import require_open3d
from pointcloud_builder.mapping.open3d.extraction import extract_geometry
from pointcloud_builder.mapping.open3d.serialization import load_volume, save_volume
from pointcloud_builder.mapping.types import (
    MapExtraction,
    RigDepthFrameSet,
    RigDepthObservation,
    TsdfIntegrationResult,
    TsdfMapState,
)


class FeatureNotSupportedError(NotImplementedError):
    """Raised for operations the backend cannot implement safely."""


class Open3dTsdfMap:
    """Fixed-workspace TSDF map with explicit create/freeze/reset lifecycle."""

    def __init__(self, config: TsdfMapConfig, *, workspace_frame: str) -> None:
        if not workspace_frame.strip():
            raise ValueError("workspace_frame must be non-empty")
        self.config = config
        self.workspace_frame = workspace_frame
        self.o3d = require_open3d()
        self._volume = self._create_volume()
        self._state = TsdfMapState(
            lifecycle="created",
            workspace_frame=workspace_frame,
            integrated_frame_sets=0,
            integrated_observations=0,
            active_block_count=0,
            last_matched_set_index=None,
            map_revision=0,
        )

    def _create_volume(self) -> Any:
        o3d = self.o3d
        return o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3d.core.float32, o3d.core.uint16, o3d.core.uint16),
            attr_channels=((1,), (1,), (3,)),
            voxel_size=self.config.volume.voxel_size_m,
            block_resolution=self.config.volume.block_resolution,
            block_count=self.config.volume.block_count,
            device=o3d.core.Device(self.config.backend.device),
        )

    @property
    def state(self) -> TsdfMapState:
        return self._state

    @property
    def volume(self) -> Any:
        self._require_open()
        return self._volume

    def integrate(self, frame_set: RigDepthFrameSet) -> TsdfIntegrationResult:
        self._require_open()
        if self._state.lifecycle == "frozen":
            return TsdfIntegrationResult(
                matched_set_index=frame_set.matched_set_index,
                integrated_cameras=(),
                active_block_count=self._state.active_block_count,
                integration_ms=0.0,
                skipped=True,
                reason="map_is_frozen",
            )
        if frame_set.matched_set_index % self.config.integration.frame_stride:
            return TsdfIntegrationResult(
                matched_set_index=frame_set.matched_set_index,
                integrated_cameras=(),
                active_block_count=self._state.active_block_count,
                integration_ms=0.0,
                skipped=True,
                reason="frame_stride",
            )
        self._validate_frame_set(frame_set)
        started = time.perf_counter()
        integrated_cameras = []
        for observation in frame_set.observations:
            if self._integrate_observation(observation):
                integrated_cameras.append(observation.camera_name)
        if not integrated_cameras:
            return TsdfIntegrationResult(
                matched_set_index=frame_set.matched_set_index,
                integrated_cameras=(),
                active_block_count=self._state.active_block_count,
                integration_ms=(time.perf_counter() - started) * 1000.0,
                skipped=True,
                reason="no_valid_depth",
            )
        active = int(self._volume.hashmap().size())
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._state = replace(
            self._state,
            lifecycle="integrating",
            integrated_frame_sets=self._state.integrated_frame_sets + 1,
            integrated_observations=(
                self._state.integrated_observations + len(integrated_cameras)
            ),
            active_block_count=active,
            last_matched_set_index=frame_set.matched_set_index,
            map_revision=self._state.map_revision + 1,
        )
        return TsdfIntegrationResult(
            matched_set_index=frame_set.matched_set_index,
            integrated_cameras=tuple(integrated_cameras),
            active_block_count=active,
            integration_ms=elapsed_ms,
        )

    def _validate_frame_set(self, frame_set: RigDepthFrameSet) -> None:
        if any(
            x.workspace_frame != self.workspace_frame for x in frame_set.observations
        ):
            raise ValueError("depth observation workspace differs from TSDF map frame")
        if any(
            x.depth_source != self.config.integration.source
            for x in frame_set.observations
        ):
            raise ValueError("depth observation source differs from TSDF config")
        for observation in frame_set.observations:
            distorted = any(
                abs(value) > 1e-12 for value in observation.distortion_coeffs
            )
            if distorted and not observation.rectified:
                raise ValueError(
                    "non-identity camera distortion requires explicitly rectified depth"
                )

    def _integrate_observation(self, observation: RigDepthObservation) -> bool:
        o3d = self.o3d
        device = o3d.core.Device(self.config.backend.device)
        depth = observation.depth.copy()
        metric = depth.astype(np.float32) * observation.depth_scale_m_per_unit
        valid = (
            observation.valid_mask
            & (metric >= self.config.depth.minimum_m)
            & (metric <= self.config.depth.maximum_m)
        )
        depth[~valid] = 0
        if not valid.any():
            return False
        depth_image = o3d.t.geometry.Image(o3d.core.Tensor(depth, device=device))
        intrinsic = o3d.core.Tensor(
            intrinsic_matrix(observation),
            dtype=o3d.core.Dtype.Float64,
            device=device,
        )
        extrinsic = o3d.core.Tensor(
            T_camera_from_workspace(observation),
            dtype=o3d.core.Dtype.Float64,
            device=device,
        )
        scale = open3d_depth_scale(observation)
        coords = self._volume.compute_unique_block_coordinates(
            depth_image,
            intrinsic,
            extrinsic,
            depth_scale=scale,
            depth_max=self.config.depth.maximum_m,
            trunc_voxel_multiplier=self.config.volume.trunc_voxel_multiplier,
        )
        self._volume.integrate(
            coords,
            depth_image,
            intrinsic,
            extrinsic,
            depth_scale=scale,
            depth_max=self.config.depth.maximum_m,
            trunc_voxel_multiplier=self.config.volume.trunc_voxel_multiplier,
        )
        return True

    def extract(self) -> MapExtraction:
        self._require_open()
        return extract_geometry(
            self._volume,
            weight_threshold=self.config.extraction.weight_threshold,
        )

    def volume_statistics(self) -> dict[str, Any]:
        self._require_open()
        hashmap = self._volume.hashmap()
        active = hashmap.active_buf_indices()
        result: dict[str, Any] = {
            "active_block_count": int(hashmap.size()),
            "attributes": {},
        }
        for name in ("tsdf", "weight", "color"):
            tensor = self._volume.attribute(name)
            values = tensor[active].cpu().numpy()
            entry: dict[str, Any] = {
                "shape": list(values.shape),
                "dtype": str(tensor.dtype),
            }
            if name in {"tsdf", "weight"} and values.size:
                entry.update(
                    {
                        "minimum": float(values.min()),
                        "maximum": float(values.max()),
                        "mean": float(values.mean()),
                        "nonzero_count": int(np.count_nonzero(values)),
                    }
                )
            result["attributes"][name] = entry
        return result

    def raycast_depth(self, observation: RigDepthObservation) -> np.ndarray:
        self._require_open()
        self._validate_frame_set(
            RigDepthFrameSet(
                matched_set_index=0,
                host_timestamp_ns=observation.timestamp_ns,
                maximum_skew_ms=0.0,
                observations=(observation,),
            )
        )
        hashmap = self._volume.hashmap()
        if int(hashmap.size()) == 0:
            return np.zeros(
                (observation.intrinsics.height, observation.intrinsics.width),
                dtype=np.float32,
            )
        active = hashmap.active_buf_indices()
        block_coords = hashmap.key_tensor()[active]
        result = self._volume.ray_cast(
            block_coords,
            self.o3d.core.Tensor(
                intrinsic_matrix(observation),
                dtype=self.o3d.core.float64,
                device=self.o3d.core.Device(self.config.backend.device),
            ),
            self.o3d.core.Tensor(
                T_camera_from_workspace(observation),
                dtype=self.o3d.core.float64,
                device=self.o3d.core.Device(self.config.backend.device),
            ),
            observation.intrinsics.width,
            observation.intrinsics.height,
            render_attributes=("depth",),
            depth_scale=1.0,
            depth_min=self.config.depth.minimum_m,
            depth_max=self.config.depth.maximum_m,
            weight_threshold=self.config.extraction.weight_threshold,
            trunc_voxel_multiplier=self.config.volume.trunc_voxel_multiplier,
        )
        return np.asarray(result["depth"].cpu().numpy(), dtype=np.float32).reshape(
            observation.intrinsics.height, observation.intrinsics.width
        )

    def freeze(self) -> TsdfMapState:
        self._require_open()
        self._state = replace(self._state, lifecycle="frozen")
        return self._state

    def unfreeze(self) -> TsdfMapState:
        self._require_open()
        if self._state.lifecycle != "frozen":
            raise RuntimeError("only a frozen TSDF map can be explicitly unfrozen")
        self._state = replace(self._state, lifecycle="integrating")
        return self._state

    def save(self, path: str | Path) -> None:
        self._require_open()
        if int(self._volume.hashmap().size()) == 0:
            # Open3D 0.19 cannot load a native VoxelBlockGrid saved with zero
            # keys. One zero-weight allocation preserves an empty logical map
            # while keeping the required native save/load route recoverable.
            device = self.o3d.core.Device(self.config.backend.device)
            key = self.o3d.core.Tensor(
                [[0, 0, 0]], dtype=self.o3d.core.int32, device=device
            )
            self._volume.hashmap().activate(key)
            self._state = replace(self._state, active_block_count=1)
        save_volume(self._volume, path)

    def load(self, path: str | Path) -> TsdfMapState:
        self._require_open()
        self._volume = load_volume(self.o3d, path)
        statistics = self.volume_statistics()
        expected_tail = [
            self.config.volume.block_resolution,
            self.config.volume.block_resolution,
            self.config.volume.block_resolution,
        ]
        for name, channels, dtype in (
            ("tsdf", 1, "Float32"),
            ("weight", 1, "UInt16"),
            ("color", 3, "UInt16"),
        ):
            entry = statistics["attributes"][name]
            if (
                entry["shape"][1:] != [*expected_tail, channels]
                or entry["dtype"] != dtype
            ):
                raise ValueError(
                    f"loaded TSDF volume attribute contract mismatch: {name}"
                )
        active = int(self._volume.hashmap().size())
        self._state = replace(
            self._state,
            lifecycle="frozen",
            active_block_count=active,
            map_revision=self._state.map_revision + 1,
        )
        return self._state

    def reset(self) -> TsdfMapState:
        self._require_open()
        revision = self._state.map_revision + 1
        self._volume = self._create_volume()
        self._state = TsdfMapState(
            lifecycle="created",
            workspace_frame=self.workspace_frame,
            integrated_frame_sets=0,
            integrated_observations=0,
            active_block_count=0,
            last_matched_set_index=None,
            map_revision=revision,
        )
        return self._state

    def invalidate_aabb(self, minimum: np.ndarray, maximum: np.ndarray) -> None:
        del minimum, maximum
        raise FeatureNotSupportedError(
            "Open3D 0.19 VoxelBlockGrid has no validated atomic AABB weight reset"
        )

    def close(self) -> TsdfMapState:
        if self._state.lifecycle != "closed":
            self._volume = None
            self._state = replace(self._state, lifecycle="closed")
        return self._state

    def _require_open(self) -> None:
        if self._state.lifecycle == "closed":
            raise RuntimeError("TSDF map is closed")

    def __enter__(self) -> "Open3dTsdfMap":
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
