"""Shared FFS perception layer used by every backend."""

from __future__ import annotations

import time
from typing import Any

import torch

from pointcloud_builder.ffs.calibration import calibration_from_builder_config
from pointcloud_builder.ffs.factory import create_backend
from pointcloud_builder.ffs.geometry import disparity_to_depth
from pointcloud_builder.ffs.preprocessing import normalize_disparity_output, prepare_ir_batch
from pointcloud_builder.ffs.types import FFSDepthResult, FFSDisparityBackend, frame_field


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
        if self.calibration.left_intrinsics.height != int(config.height) or self.calibration.left_intrinsics.width != int(config.width):
            raise ValueError(
                "FFS calibration shape does not match the fixed model shape: "
                f"calibration={(self.calibration.left_intrinsics.height, self.calibration.left_intrinsics.width)}, "
                f"model={(config.height, config.width)}"
            )

    def infer(self, frame: Any) -> FFSDepthResult:
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
