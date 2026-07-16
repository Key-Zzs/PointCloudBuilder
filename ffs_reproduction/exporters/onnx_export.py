"""ONNX exporters for the three TensorRT routes.

All functions run in the caller's dp3 interpreter and import only the copied
FFS vendor tree.  They never invoke ``trtexec``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcloud_builder.ffs.vendor_loader import scoped_vendor_imports, vendor_root


ONNX_EXPORT_CONTRACT = {
    "opset_version": 17,
    "dynamo": False,
    "do_constant_folding": True,
}


def validate_onnx_contract(
    path: str | Path,
    *,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    height: int = 480,
    width: int = 640,
    custom_plugin: bool = False,
) -> dict[str, Any]:
    """Run the exporter-side checker and reject dynamic shape/name drift."""

    import onnx

    model = onnx.load(str(Path(path).expanduser().resolve()))
    checker_status = "passed"
    if custom_plugin:
        # The standard checker intentionally rejects the private
        # FFSGWCVolume op because it is not registered in ai.onnx. TensorRT
        # parser/plugin registration is the authoritative check for this
        # graph; still validate all ordinary graph metadata here.
        checker_status = "skipped_custom_op"
    else:
        onnx.checker.check_model(model)
    opsets = {item.domain or "ai.onnx": int(item.version) for item in model.opset_import}
    if opsets.get("ai.onnx") != int(ONNX_EXPORT_CONTRACT["opset_version"]):
        raise ValueError(f"ONNX opset must be 17, got {opsets}")
    graph_inputs = tuple(value.name for value in model.graph.input)
    graph_outputs = tuple(value.name for value in model.graph.output)
    if graph_inputs != input_names or graph_outputs != output_names:
        raise ValueError(
            f"ONNX IO names mismatch: inputs={graph_inputs}, outputs={graph_outputs}; "
            f"expected inputs={input_names}, outputs={output_names}"
        )
    input_shapes: list[list[int]] = []
    for value in model.graph.input:
        dims = value.type.tensor_type.shape.dim
        shape = [int(dim.dim_value) for dim in dims]
        if any(int(dim.dim_value) <= 0 or dim.dim_param for dim in dims):
            raise ValueError(f"ONNX input {value.name} must have a static positive shape")
        if shape != [1, 3, int(height), int(width)]:
            raise ValueError(f"ONNX input {value.name} shape {shape} is not [1,3,{height},{width}]")
        input_shapes.append(shape)
    if custom_plugin:
        if not any(node.op_type == "FFSGWCVolume" for node in model.graph.node):
            raise ValueError(f"Plugin ONNX has no FFSGWCVolume node: {path}")
    return {
        "opset": 17,
        "input_names": list(graph_inputs),
        "output_names": list(graph_outputs),
        "input_shapes": input_shapes,
        "node_count": len(model.graph.node),
        "custom_plugin": bool(custom_plugin),
        "checker_status": checker_status,
    }


def _disable_compile() -> None:
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def load_model(checkpoint_path: str | Path, *, max_disp: int, valid_iters: int, precision: str, device: torch.device):
    """Load the trusted official pickle through the copied module aliases."""

    _disable_compile()
    with scoped_vendor_imports(vendor_root()):
        model = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location="cpu", weights_only=False)
    model.args.max_disp = int(max_disp)
    model.args.valid_iters = int(valid_iters)
    model.args.mixed_precision = precision == "fp16"
    return model.to(device=device).eval()


def _build_gwc_volume_onnx(refimg_fea: torch.Tensor, targetimg_fea: torch.Tensor, maxdisp: int, num_groups: int, normalize: bool = True):
    dtype = refimg_fea.dtype
    batch, channels, height, width = refimg_fea.shape
    channels_per_group = channels // num_groups
    ref_volume = refimg_fea.unsqueeze(2).expand(batch, channels, maxdisp, height, width)
    shifted = [F.pad(targetimg_fea, (d, 0, 0, 0), "constant", 0.0)[:, :, :, :width] for d in range(maxdisp)]
    target_volume = torch.stack(shifted, dim=2)
    ref_volume = ref_volume.view(batch, num_groups, channels_per_group, maxdisp, height, width)
    target_volume = target_volume.view(batch, num_groups, channels_per_group, maxdisp, height, width)
    if normalize:
        ref_volume = F.normalize(ref_volume.float(), dim=2).to(dtype)
        target_volume = F.normalize(target_volume.float(), dim=2).to(dtype)
    return (ref_volume * target_volume).sum(dim=2).contiguous()


def _build_concat_volume_onnx(refimg_fea: torch.Tensor, targetimg_fea: torch.Tensor, maxdisp: int):
    batch, channels, height, width = refimg_fea.shape
    ref_volume = refimg_fea.unsqueeze(2).expand(batch, channels, maxdisp, height, width)
    shifted = [F.pad(targetimg_fea, (d, 0, 0, 0), "constant", 0.0)[:, :, :, :width] for d in range(maxdisp)]
    target_volume = torch.stack(shifted, dim=2)
    return torch.cat((ref_volume, target_volume), dim=1).contiguous()


class _SingleWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, left_image: torch.Tensor, right_image: torch.Tensor):
        return self.model.forward(
            left_image,
            right_image,
            iters=self.model.args.valid_iters,
            test_mode=True,
            optimize_build_volume="pytorch1",
        )


class _PluginOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left: torch.Tensor, right: torch.Tensor, max_disp: int, cv_group: int, normalize: int):
        return left.new_zeros((left.shape[0], int(cv_group), int(max_disp), left.shape[2], left.shape[3]))

    @staticmethod
    def symbolic(g, left, right, max_disp, cv_group, normalize):
        from torch.onnx import symbolic_helper

        def as_int(value):
            if isinstance(value, int):
                return value
            return symbolic_helper._parse_arg(value, "i")

        max_disp = as_int(max_disp)
        cv_group = as_int(cv_group)
        normalize = as_int(normalize)
        out = g.op(
            "FFSGWCVolume",
            left,
            right,
            max_disp_i=int(max_disp),
            cv_group_i=int(cv_group),
            normalize_i=int(normalize),
        )
        sizes = left.type().sizes()
        if sizes is not None and len(sizes) == 4:
            out.setType(left.type().with_sizes([sizes[0], int(cv_group), int(max_disp), sizes[2], sizes[3]]))
        return out


class _PluginWrapper(nn.Module):
    def __init__(self, model: nn.Module, *, max_disp_levels: int, cv_group: int, normalize: bool):
        super().__init__()
        from core.foundation_stereo import TrtFeatureRunner, TrtPostRunner

        self.feature_runner = TrtFeatureRunner(model)
        self.post_runner = TrtPostRunner(model)
        self.max_disp_levels = int(max_disp_levels)
        self.cv_group = int(cv_group)
        self.normalize = int(bool(normalize))

    @torch.no_grad()
    def forward(self, left: torch.Tensor, right: torch.Tensor):
        feature = self.feature_runner(left, right)
        gwc = _PluginOp.apply(feature[0], feature[4], self.max_disp_levels, self.cv_group, self.normalize)
        return self.post_runner(feature[0], feature[1], feature[2], feature[3], feature[4], feature[5], gwc.float())


def export_single(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    height: int,
    width: int,
    max_disp: int,
    valid_iters: int,
    precision: str,
    device: torch.device,
) -> dict[str, Any]:
    """Export the ordinary single ONNX with external ImageNet normalization."""

    _disable_compile()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with scoped_vendor_imports(vendor_root()):
        import core.foundation_stereo as fs_module

        model = load_model(checkpoint_path, max_disp=max_disp, valid_iters=valid_iters, precision="fp32", device=device)
        fs_module.normalize_image = lambda image: image
        fs_module.build_gwc_volume_optimized_pytorch1 = _build_gwc_volume_onnx
        fs_module.build_concat_volume_optimized_pytorch1 = _build_concat_volume_onnx
        wrapper = _SingleWrapper(model).to(device=device).eval()
        left = torch.randn((1, 3, height, width), device=device, dtype=torch.float32)
        right = torch.randn((1, 3, height, width), device=device, dtype=torch.float32)
        torch.onnx.export(
            wrapper,
            (left, right),
            str(output_path),
            opset_version=17,
            input_names=["left_image", "right_image"],
            output_names=["disparity"],
            do_constant_folding=True,
            dynamo=False,
        )
    return {"input_names": ["left_image", "right_image"], "output_names": ["disparity"], "normalization_contract": "external_imagenet_0_255"}


def export_two_stage(
    checkpoint_path: str | Path,
    feature_path: str | Path,
    post_path: str | Path,
    *,
    height: int,
    width: int,
    max_disp: int,
    valid_iters: int,
    precision: str,
    device: torch.device,
) -> dict[str, Any]:
    """Export feature and post runners; Triton GWC stays between them."""

    _disable_compile()
    feature_path, post_path = Path(feature_path), Path(post_path)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    with scoped_vendor_imports(vendor_root()):
        from core.foundation_stereo import TrtFeatureRunner, TrtPostRunner

        model = load_model(checkpoint_path, max_disp=max_disp, valid_iters=valid_iters, precision="fp32", device=device)
        feature_runner = TrtFeatureRunner(model).to(device=device).eval()
        post_runner = TrtPostRunner(model).to(device=device).eval()
        left = torch.randn((1, 3, height, width), device=device, dtype=torch.float32) * 255.0
        right = torch.randn((1, 3, height, width), device=device, dtype=torch.float32) * 255.0
        feature = feature_runner(left, right)
        cv_group = int(getattr(model, "cv_group", 8))
        gwc = torch.zeros((1, cv_group, max_disp // 4, height // 4, width // 4), device=device, dtype=torch.float32)
        torch.onnx.export(
            feature_runner,
            (left, right),
            str(feature_path),
            opset_version=17,
            input_names=["left", "right"],
            output_names=["features_left_04", "features_left_08", "features_left_16", "features_left_32", "features_right_04", "stem_2x"],
            do_constant_folding=True,
            dynamo=False,
        )
        torch.onnx.export(
            post_runner,
            (*feature, gwc),
            str(post_path),
            opset_version=17,
            input_names=["features_left_04", "features_left_08", "features_left_16", "features_left_32", "features_right_04", "stem_2x", "gwc_volume"],
            output_names=["disp"],
            do_constant_folding=True,
            dynamo=False,
        )
    return {
        "input_names": ["left", "right"],
        "output_names": ["disp"],
        "feature_output_names": ["features_left_04", "features_left_08", "features_left_16", "features_left_32", "features_right_04", "stem_2x"],
        # The current post runner does not consume features_left_16; Torch ONNX
        # correctly prunes that unused input from the graph.
        "post_input_names": ["features_left_04", "features_left_08", "features_left_32", "features_right_04", "stem_2x", "gwc_volume"],
        "post_output_names": ["disp"],
        "normalization_contract": "internal_imagenet_0_255",
        "cv_group": cv_group,
        "gwc_normalize": bool(getattr(model.args, "normalize", False)),
    }


def export_plugin(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    height: int,
    width: int,
    max_disp: int,
    valid_iters: int,
    precision: str,
    device: torch.device,
) -> dict[str, Any]:
    """Export one raw-input ONNX graph containing FFSGWCVolume."""

    _disable_compile()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with scoped_vendor_imports(vendor_root()):
        model = load_model(checkpoint_path, max_disp=max_disp, valid_iters=valid_iters, precision="fp32", device=device)
        cv_group = int(getattr(model, "cv_group", 8))
        normalize = bool(getattr(model.args, "normalize", False))
        wrapper = _PluginWrapper(model, max_disp_levels=max_disp // 4, cv_group=cv_group, normalize=normalize).to(device=device).eval()
        left = torch.randn((1, 3, height, width), device=device, dtype=torch.float32) * 255.0
        right = torch.randn((1, 3, height, width), device=device, dtype=torch.float32) * 255.0
        torch.onnx.export(
            wrapper,
            (left, right),
            str(output_path),
            opset_version=17,
            input_names=["left", "right"],
            output_names=["disp"],
            do_constant_folding=True,
            dynamo=False,
        )
    return {
        "input_names": ["left", "right"],
        "output_names": ["disp"],
        "normalization_contract": "internal_imagenet_0_255",
        "cv_group": cv_group,
        "gwc_normalize": normalize,
    }
