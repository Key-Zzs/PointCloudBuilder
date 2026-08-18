"""Shared FFS perception layer used by every backend."""

from __future__ import annotations

import time
import hashlib
import json
from typing import Any

import torch

from pointcloud_builder.ffs.calibration import calibration_from_builder_config
from pointcloud_builder.ffs.factory import create_backend
from pointcloud_builder.ffs.geometry import disparity_to_depth
from pointcloud_builder.ffs.preprocessing import normalize_disparity_output, prepare_ir_batch
from pointcloud_builder.ffs.types import FFSDepthResult, FFSDisparityBackend, frame_field, optional_frame_field
from pointcloud_builder.ffs.exact_cache import ExactDisparityCache
from pointcloud_builder.ffs.manifest import sha256_file


class FFSStereoDepthEstimator:
    """Normalize IR, run one selected backend, and produce metric depth."""

    def __init__(
        self,
        config: Any,
        camera_config: Any,
        *,
        device: torch.device,
        backend: FFSDisparityBackend | None = None,
    ) -> None:
        if (int(config.height), int(config.width)) != (480, 640):
            raise ValueError("Current FFS estimator accepts only height=480,width=640")
        self.config = config
        self.device = device
        self.calibration = calibration_from_builder_config(camera_config, config)
        self.backend = backend or create_backend(config, device=device)
        self._exact_cache: ExactDisparityCache | None = None
        if self.calibration.left_intrinsics.height != int(config.height) or self.calibration.left_intrinsics.width != int(config.width):
            raise ValueError(
                "FFS calibration shape does not match the fixed model shape: "
                f"calibration={(self.calibration.left_intrinsics.height, self.calibration.left_intrinsics.width)}, "
                f"model={(config.height, config.width)}"
            )

    def infer(self, frame: Any) -> FFSDepthResult:
        frame_index = optional_frame_field(frame, "global_frame_index")
        disparity = (
            self._exact_cache.load(int(frame_index), device=self.device)
            if self._exact_cache is not None and frame_index is not None
            else None
        )
        cache_hit = disparity is not None
        if disparity is None:
            disparity, inference_ms = self._infer_normalized_disparity(frame)
            if self._exact_cache is not None and frame_index is not None:
                self._exact_cache.store(int(frame_index), disparity)
        else:
            inference_ms = 0.0
        conversion_start = time.perf_counter()
        depth_m, valid_mask, counts = disparity_to_depth(
            disparity,
            fx_px=self.calibration.left_intrinsics.fx,
            baseline_m=self.calibration.baseline_m,
            epsilon=float(self.config.min_disparity_px),
            remove_invisible=bool(self.config.remove_invisible),
            min_depth_m=float(self.config.min_depth_m),
            max_depth_m=self.config.max_depth_m,
        )
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        conversion_ms = (time.perf_counter() - conversion_start) * 1000.0
        backend_timing = dict(getattr(self.backend, "last_timing_ms", {}))
        backend_timing.setdefault("inference_ms", inference_ms)
        metadata: dict[str, Any] = {
            "depth_source": "ffs_stereo",
            "backend": self.backend.name,
            "input_shape": [int(self.config.height), int(self.config.width)],
            "output_shape": [int(self.config.height), int(self.config.width)],
            "unit": "meter",
            "fx_px": self.calibration.left_intrinsics.fx,
            "baseline_m": self.calibration.baseline_m,
            "rectification": self.calibration.metadata,
            "max_disp": int(self.config.max_disp),
            "valid_iters": int(self.config.valid_iters),
            "precision": str(self.config.precision),
            "min_disparity_px": float(self.config.min_disparity_px),
            "min_depth_m": float(self.config.min_depth_m),
            "max_depth_m": self.config.max_depth_m,
            "valid_disparity_count": counts["valid"],
            "invalid_disparity_count": counts["invalid"],
            "invalid_disparity_reasons": {
                key: value for key, value in counts.items() if key not in {"total", "valid", "invalid"}
            },
            "effective_depth_scale": 1.0,
            "remove_invisible": bool(self.config.remove_invisible),
            "model_provenance": dict(self.backend.provenance),
            "runtime": _runtime_metadata(),
            "timing_ms": {
                "inference": inference_ms,
                "disparity_to_depth": conversion_ms,
                **{key: value for key, value in backend_timing.items() if key != "inference_ms"},
            },
            "exact_cache": {"enabled": False, "hit": False} if self._exact_cache is None else {**self._exact_cache.metadata(), "hit": cache_hit},
            "device": str(self.device),
        }
        return FFSDepthResult(
            disparity_px=disparity,
            depth_m=depth_m,
            valid_mask=valid_mask,
            intrinsics=self.calibration.left_intrinsics,
            depth_to_color_extrinsics=self.calibration.left_to_color,
            metadata=metadata,
        )

    def _infer_normalized_disparity(self, frame: Any) -> tuple[torch.Tensor, float]:
        left = prepare_ir_batch(
            frame_field(frame, self.config.left_key),
            name=self.config.left_key,
            height=int(self.config.height),
            width=int(self.config.width),
            device=self.device,
        )
        right = prepare_ir_batch(
            frame_field(frame, self.config.right_key),
            name=self.config.right_key,
            height=int(self.config.height),
            width=int(self.config.width),
            device=self.device,
        )
        inference_start = time.perf_counter()
        if self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            disparity = self.backend.infer_disparity(left, right)
            end_event.record()
            end_event.synchronize()
            inference_ms = float(start_event.elapsed_time(end_event))
        else:
            disparity = self.backend.infer_disparity(left, right)
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
        disparity = normalize_disparity_output(
            disparity,
            height=int(self.config.height),
            width=int(self.config.width),
            device=self.device,
        )
        return disparity, inference_ms

    def configure_exact_cache(
        self,
        *,
        source_dataset_hash: str,
        frame_join_hash: str,
        cache_root: str | None = None,
    ) -> None:
        """Bind lossless cache reuse to the source and runtime provenance."""

        root = cache_root or getattr(self.config, "exact_cache_root", None)
        if not root:
            self._exact_cache = None
            return
        config_value = {
            key: getattr(self.config, key)
            for key in vars(self.config)
            if key != "exact_cache_root"
        }
        config_hash = hashlib.sha256(json.dumps(config_value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
        backend = dict(self.backend.provenance)
        manifest = backend.get("manifest", {})
        self._exact_cache = ExactDisparityCache(
            root,
            provenance={
                "source_dataset_hash": source_dataset_hash,
                "frame_join_hash": frame_join_hash,
                "ffs_engine_sha": backend.get("engine_sha256", backend.get("feature_engine_sha256")),
                "plugin_sha": backend.get("plugin_library_sha256", manifest.get("plugin_library_sha256") if isinstance(manifest, dict) else None),
                "calibration_hash": sha256_file(self.config.calibration_path) if self.config.calibration_path else None,
                "ffs_config_hash": config_hash,
            },
        )


def _runtime_metadata() -> dict[str, Any]:
    """Record the runtime binding without importing optional packages in old mode."""

    value: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": None,
        "compute_capability": None,
        "tensorrt_version": None,
    }
    if torch.cuda.is_available():
        value["gpu_name"] = torch.cuda.get_device_name(0)
        value["compute_capability"] = list(torch.cuda.get_device_capability(0))
    try:
        import tensorrt as trt
    except ImportError:
        pass
    else:
        value["tensorrt_version"] = getattr(trt, "__version__", "unknown")
    return value
