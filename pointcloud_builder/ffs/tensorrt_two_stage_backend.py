"""TensorRT feature engine + Triton GWC + TensorRT post engine backend."""

from __future__ import annotations

from typing import Any

import torch

from pointcloud_builder.ffs.manifest import load_manifest
from pointcloud_builder.ffs.preprocessing import normalize_disparity_output
from pointcloud_builder.ffs.tensorrt_common import TensorRTEngine
from pointcloud_builder.ffs.vendor_loader import scoped_vendor_imports, vendor_root


def _load_triton_gwc():
    with scoped_vendor_imports(vendor_root()):
        from core.submodule import build_gwc_volume_triton

        return build_gwc_volume_triton


class TensorRTTwoStageBackend:
    name = "tensorrt_two_stage"
    normalization_contract = "internal_imagenet_0_255"

    def __init__(self, config: Any, *, device: torch.device) -> None:
        if device.type != "cuda":
            raise RuntimeError("TensorRT two-stage backend requires CUDA")
        missing = [key for key in ("feature_engine_path", "post_engine_path", "manifest_path") if not getattr(config, key, None)]
        if missing:
            raise ValueError(f"tensorrt_two_stage backend requires {', '.join(missing)}")
        self.height = int(config.height)
        self.width = int(config.width)
        self.max_disp = int(config.max_disp)
        self.gwc_builder = _load_triton_gwc()
        manifest = load_manifest(
            config.manifest_path,
            backend=self.name,
            height=self.height,
            width=self.width,
            max_disp=int(config.max_disp),
            valid_iters=int(config.valid_iters),
            precision=str(config.precision),
            normalization_contract=self.normalization_contract,
            artifact_paths=(config.feature_engine_path, config.post_engine_path),
            input_names=("left", "right"),
            output_names=("disp",),
            config_path=getattr(config, "config_path", None),
            builder_optimization_level=getattr(config, "builder_optimization_level", None),
            workspace_gib=getattr(config, "workspace_gib", None),
        )
        self.cv_group = int(getattr(config, "cv_group", manifest.get("cv_group", 8)))
        if manifest.get("cv_group") is not None and int(manifest["cv_group"]) != self.cv_group:
            raise ValueError(
                f"FFS manifest mismatch for cv_group: manifest={manifest['cv_group']!r}, configured={self.cv_group!r}"
            )
        self.feature_engine = TensorRTEngine(
            config.feature_engine_path,
            input_shapes={"left": (1, 3, self.height, self.width), "right": (1, 3, self.height, self.width)},
            expected_inputs=("left", "right"),
            expected_outputs=("features_left_04", "features_left_08", "features_left_16", "features_left_32", "features_right_04", "stem_2x"),
        )
        feature_shapes = {name: tuple(int(x) for x in self.feature_engine._outputs[name].shape) for name in self.feature_engine.output_names}
        feature_shapes["gwc_volume"] = (1, self.cv_group, self.max_disp // 4, self.height // 4, self.width // 4)
        self.post_engine = TensorRTEngine(
            config.post_engine_path,
            input_shapes=feature_shapes,
            expected_inputs=("features_left_04", "features_left_08", "features_left_32", "features_right_04", "stem_2x", "gwc_volume"),
            expected_outputs=("disp",),
        )
        expected_post_inputs = [
            "features_left_04",
            "features_left_08",
            "features_left_32",
            "features_right_04",
            "stem_2x",
            "gwc_volume",
        ]
        if manifest.get("post_input_names") and list(manifest["post_input_names"]) != expected_post_inputs:
            raise ValueError("FFS two-stage manifest post_input_names do not match the runtime contract")
        self.provenance = {
            "feature_engine_path": str(self.feature_engine.path),
            "feature_engine_sha256": self.feature_engine.sha256,
            "post_engine_path": str(self.post_engine.path),
            "post_engine_sha256": self.post_engine.sha256,
            "manifest": manifest,
            "normalization": self.normalization_contract,
            "cv_group": self.cv_group,
            "gwc_normalize": bool(manifest.get("gwc_normalize", False)),
        }
        self.gwc_normalize = bool(manifest.get("gwc_normalize", False))
        self.last_timing_ms: dict[str, float] = {}

    @torch.inference_mode()
    def infer_disparity(self, left_ir: torch.Tensor, right_ir: torch.Tensor) -> torch.Tensor:
        start = torch.cuda.Event(enable_timing=True)
        feature_end = torch.cuda.Event(enable_timing=True)
        gwc_end = torch.cuda.Event(enable_timing=True)
        post_end = torch.cuda.Event(enable_timing=True)
        start.record()
        feature = self.feature_engine.infer({"left": left_ir, "right": right_ir})
        feature_end.record()
        gwc = self.gwc_builder(
            feature["features_left_04"].half(),
            feature["features_right_04"].half(),
            self.max_disp // 4,
            self.cv_group,
            normalize=self.gwc_normalize,
        )
        gwc_end.record()
        post_inputs = {name: feature[name] for name in self.post_engine.input_names if name != "gwc_volume"}
        post_inputs["gwc_volume"] = gwc
        output = self.post_engine.infer(post_inputs)
        post_end.record()
        post_end.synchronize()
        self.last_timing_ms = {
            "feature_engine_ms": float(start.elapsed_time(feature_end)),
            "gwc_triton_ms": float(feature_end.elapsed_time(gwc_end)),
            "post_engine_ms": float(gwc_end.elapsed_time(post_end)),
            "inference_ms": float(start.elapsed_time(post_end)),
        }
        return normalize_disparity_output(output["disp"], height=self.height, width=self.width, device=left_ir.device)
