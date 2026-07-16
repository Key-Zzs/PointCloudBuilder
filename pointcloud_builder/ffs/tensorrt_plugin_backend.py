"""TensorRT plugin backend with explicit library registration before deserialize."""

from __future__ import annotations

from typing import Any

import torch

from pointcloud_builder.ffs.manifest import load_manifest
from pointcloud_builder.ffs.preprocessing import normalize_disparity_output
from pointcloud_builder.ffs.tensorrt_common import TensorRTEngine, load_plugin_library


class TensorRTPluginBackend:
    name = "tensorrt_plugin"
    normalization_contract = "internal_imagenet_0_255"

    def __init__(self, config: Any, *, device: torch.device) -> None:
        if device.type != "cuda":
            raise RuntimeError("TensorRT plugin backend requires CUDA")
        missing = [key for key in ("engine_path", "plugin_library_path", "manifest_path") if not getattr(config, key, None)]
        if missing:
            raise ValueError(f"tensorrt_plugin backend requires {', '.join(missing)}")
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
            artifact_paths=(config.engine_path, config.plugin_library_path),
            input_names=("left", "right"),
            output_names=("disp",),
            config_path=getattr(config, "config_path", None),
            builder_optimization_level=getattr(config, "builder_optimization_level", None),
            workspace_gib=getattr(config, "workspace_gib", None),
        )
        self.plugin_library = load_plugin_library(config.plugin_library_path)
        self.engine = TensorRTEngine(
            config.engine_path,
            input_shapes={"left": (1, 3, self.height, self.width), "right": (1, 3, self.height, self.width)},
            expected_inputs=("left", "right"),
            expected_outputs=("disp",),
        )
        self.provenance = {
            "engine_path": str(self.engine.path),
            "engine_sha256": self.engine.sha256,
            "plugin_library_path": str(config.plugin_library_path),
            "plugin_library_sha256": manifest.get("plugin_library_sha256"),
            "manifest": manifest,
        }
        self.last_timing_ms: dict[str, float] = {}

    @torch.inference_mode()
    def infer_disparity(self, left_ir: torch.Tensor, right_ir: torch.Tensor) -> torch.Tensor:
        outputs = self.engine.infer({"left": left_ir, "right": right_ir})
        return normalize_disparity_output(outputs["disp"], height=self.height, width=self.width, device=left_ir.device)
