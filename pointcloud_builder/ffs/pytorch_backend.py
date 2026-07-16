"""PyTorch FFS reference backend using only the copied vendor tree."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from pointcloud_builder.ffs.preprocessing import normalize_disparity_output
from pointcloud_builder.ffs.types import Tensor
from pointcloud_builder.ffs.vendor_loader import scoped_vendor_imports, vendor_root
from pointcloud_builder.ffs.manifest import sha256_file


class PyTorchFFSBackend:
    name = "pytorch"
    normalization_contract = "internal_imagenet_0_255"

    def __init__(self, config: Any, *, device: torch.device) -> None:
        if device.type != "cuda":
            raise RuntimeError("The real FFS PyTorch backend requires CUDA; use a fake backend for CPU unit tests")
        checkpoint = getattr(config, "checkpoint_path", None)
        if not checkpoint:
            raise ValueError("pytorch backend requires depth_source.ffs.checkpoint_path")
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"FFS checkpoint does not exist: {checkpoint_path}")
        self.device = device
        self.height = int(config.height)
        self.width = int(config.width)
        self.max_disp = int(config.max_disp)
        self.valid_iters = int(config.valid_iters)
        self.precision = str(config.precision)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = sha256_file(checkpoint_path)
        model_config_path = getattr(config, "model_config_path", None)
        self.model_config_sha256 = None
        if model_config_path:
            model_config_file = Path(model_config_path).expanduser().resolve()
            if not model_config_file.is_file():
                raise FileNotFoundError(f"FFS model config does not exist: {model_config_file}")
            self.model_config_sha256 = sha256_file(model_config_file)
        with scoped_vendor_imports(vendor_root()):
            # This is deliberately explicit. The official checkpoint is a
            # trusted local pickle whose module names are provided by the
            # scoped copied vendor tree; arbitrary untrusted pickle files must
            # not be loaded with weights_only=False.
            self.model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.args.max_disp = self.max_disp
        self.model.args.valid_iters = self.valid_iters
        self.model.args.mixed_precision = self.precision == "fp16"
        self.model = self.model.to(device=device).eval()
        self.provenance = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_config_path": str(Path(model_config_path).expanduser().resolve()) if model_config_path else None,
            "model_config_sha256": self.model_config_sha256,
            "vendor_root": str(vendor_root()),
            "normalization_contract": self.normalization_contract,
            "max_disp": self.max_disp,
            "valid_iters": self.valid_iters,
            "precision": self.precision,
            "artifact_id": getattr(config, "artifact_id", None),
            "cv_group": int(getattr(self.model, "cv_group", 8)),
            "normalize": bool(getattr(self.model.args, "normalize", False)),
        }
        self.last_timing_ms: dict[str, float] = {}

    @torch.inference_mode()
    def infer_disparity(self, left_ir: Tensor, right_ir: Tensor) -> Tensor:
        import time

        start = time.perf_counter()
        output = self.model(
            left_ir,
            right_ir,
            iters=self.valid_iters,
            test_mode=True,
            optimize_build_volume="pytorch1",
        )
        if left_ir.is_cuda:
            torch.cuda.current_stream(left_ir.device).synchronize()
        self.last_timing_ms = {"inference_ms": (time.perf_counter() - start) * 1000.0}
        return normalize_disparity_output(output, height=self.height, width=self.width, device=self.device)
