"""Ordinary single-engine TensorRT backend."""

from __future__ import annotations

from typing import Any

import torch

from pointcloud_builder.ffs.manifest import load_manifest
from pointcloud_builder.ffs.preprocessing import imagenet_normalize_0_255, normalize_disparity_output
from pointcloud_builder.ffs.tensorrt_common import TensorRTEngine


class TensorRTSingleBackend:
    name = "tensorrt_single"
    normalization_contract = "external_imagenet_0_255"

    def __init__(self, config: Any, *, device: torch.device) -> None:
        if device.type != "cuda":
            raise RuntimeError("TensorRT requires CUDA")
        if not config.engine_path:
            raise ValueError("tensorrt_single backend requires engine_path")
        if not config.manifest_path:
            raise ValueError("tensorrt_single backend requires manifest_path")
        self.height = int(config.height)
        self.width = int(config.width)
        manifest = load_manifest(
            config.manifest_path,
            backend=self.name,
            height=self.height,
            width=self.width,
            max_disp=int(config.max_disp),
            valid_iters=int(config.valid_iters),
            precision=str(config.precision),
            normalization_contract=self.normalization_contract,
            artifact_paths=(config.engine_path,),
            input_names=("left_image", "right_image"),
            output_names=("disparity",),
            config_path=getattr(config, "config_path", None),
            builder_optimization_level=getattr(config, "builder_optimization_level", None),
            workspace_gib=getattr(config, "workspace_gib", None),
        )
        self.engine = TensorRTEngine(
            config.engine_path,
            input_shapes={"left_image": (1, 3, self.height, self.width), "right_image": (1, 3, self.height, self.width)},
            expected_inputs=("left_image", "right_image"),
            expected_outputs=("disparity",),
        )
        self.provenance = {"engine_path": str(self.engine.path), "engine_sha256": self.engine.sha256, "manifest": manifest}
        self.last_timing_ms: dict[str, float] = {}

    @torch.inference_mode()
    def infer_disparity(self, left_ir: torch.Tensor, right_ir: torch.Tensor) -> torch.Tensor:
        left = imagenet_normalize_0_255(left_ir)
        right = imagenet_normalize_0_255(right_ir)
        outputs = self.engine.infer({"left_image": left, "right_image": right})
        return normalize_disparity_output(outputs["disparity"], height=self.height, width=self.width, device=left_ir.device)
