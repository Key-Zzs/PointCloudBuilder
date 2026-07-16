"""Strict backend factory. There is intentionally no fallback route."""

from __future__ import annotations

from typing import Any

import torch

from pointcloud_builder.ffs.types import FFSDisparityBackend


BACKEND_NAMES = ("pytorch", "tensorrt_single", "tensorrt_two_stage", "tensorrt_plugin")


def create_backend(config: Any, *, device: torch.device) -> FFSDisparityBackend:
    name = str(config.backend).lower()
    if name == "pytorch":
        from pointcloud_builder.ffs.pytorch_backend import PyTorchFFSBackend

        return PyTorchFFSBackend(config, device=device)
    if name == "tensorrt_single":
        from pointcloud_builder.ffs.tensorrt_single_backend import TensorRTSingleBackend

        return TensorRTSingleBackend(config, device=device)
    if name == "tensorrt_two_stage":
        from pointcloud_builder.ffs.tensorrt_two_stage_backend import TensorRTTwoStageBackend

        return TensorRTTwoStageBackend(config, device=device)
    if name == "tensorrt_plugin":
        from pointcloud_builder.ffs.tensorrt_plugin_backend import TensorRTPluginBackend

        return TensorRTPluginBackend(config, device=device)
    raise ValueError(f"Unsupported FFS backend {config.backend!r}; expected one of {BACKEND_NAMES}")
