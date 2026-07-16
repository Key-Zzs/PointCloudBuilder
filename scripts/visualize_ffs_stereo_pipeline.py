#!/usr/bin/env python3
"""Offline one-pass visualization for the FFS perception and builder stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pointcloud_builder import PointCloudBuilder
from pointcloud_builder.frame_io import load_frame
from pointcloud_builder.visualization import save_ascii_ply, show_open3d


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().to(device="cpu").numpy()
    return np.asarray(value)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _save_png(path: Path, value: Any, *, mask: bool = False) -> None:
    from PIL import Image

    image = _array(value)
    if image.ndim == 3 and image.shape[-1] >= 3:
        data = image[..., :3]
        if np.nanmax(data) <= 1.0:
            data = data * 255.0
        Image.fromarray(np.clip(data, 0, 255).astype(np.uint8), mode="RGB").save(path)
        return
    image = image.astype(np.float32)
    if mask:
        data = np.where(image > 0, 255, 0).astype(np.uint8)
    else:
        finite = np.isfinite(image)
        if not finite.any():
            data = np.zeros(image.shape, dtype=np.uint8)
        else:
            lo, hi = np.percentile(image[finite], [1, 99])
            if hi <= lo:
                hi = lo + 1.0
            data = np.clip((image - lo) / (hi - lo), 0, 1)
            data[~finite] = 0
            data = (data * 255.0).astype(np.uint8)
    Image.fromarray(data, mode="L").save(path)


def _stage_count_breakdown(perception: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """Separate model invalidity, geometric filtering, crop, and sampling."""

    disparity = _array(perception["disparity"]).astype(np.float32)
    depth = _array(perception["depth"]).astype(np.float32)
    valid_mask = _array(perception["valid_mask"]).astype(bool)
    ffs_meta = meta.get("ffs", {}) if isinstance(meta.get("ffs", {}), dict) else {}
    epsilon = float(ffs_meta.get("min_disparity_px", 0.001))
    finite = np.isfinite(disparity)
    positive = finite & (disparity > epsilon)
    invalid_disparity = ~positive
    if disparity.ndim == 2:
        u = np.arange(disparity.shape[1], dtype=np.float32)[None, :]
        remove_invisible = positive & ((u - disparity) < 0.0)
    else:
        remove_invisible = np.zeros_like(disparity, dtype=bool)
    min_depth = float(ffs_meta.get("min_depth_m", 0.0))
    max_depth_value = ffs_meta.get("max_depth_m")
    z_range_invalid = np.isfinite(depth) & ((depth < min_depth) | ((depth > float(max_depth_value)) if max_depth_value is not None else False))
    total_pixels = int(disparity.size)
    raw_count = int(_array(perception["raw"]).shape[0])
    cropped_count = int(_array(perception["cropped"]).shape[0])
    sampled_count = int(_array(perception["sampled"]).shape[0])
    sampling_meta = meta.get("sampling", {}) if isinstance(meta.get("sampling", {}), dict) else {}
    breakdown = {
        "visualization_contract": {"denoise_cloud": False, "zfar_m": 100.0, "zfar_applied": False},
        "ffs": {
            "invalid_disparity_count": int(invalid_disparity.sum()),
            "invalid_disparity_definition": f"non-finite or disparity <= {epsilon}",
            "remove_invisible_count": int(remove_invisible.sum()),
            "remove_invisible_definition": "left_pixel_x - disparity < 0",
            "z_range_invalid_count": int(z_range_invalid.sum()),
            "z_range_definition": {"min_depth_m": min_depth, "max_depth_m": max_depth_value},
            "final_valid_disparity_count": int(valid_mask.sum()),
            "final_invalid_disparity_count": int((~valid_mask).sum()),
            "image_pixel_count": total_pixels,
        },
        "builder": {
            "deprojection_input_points": raw_count,
            "deprojection_removed_points": total_pixels - raw_count,
            "crop_input_points": raw_count,
            "crop_output_points": cropped_count,
            "crop_removed_points": raw_count - cropped_count,
            "sampling_input_points": cropped_count,
            "sampling_output_points": sampled_count,
            "sampling_removed_or_padded_points": cropped_count - sampled_count,
            "sampling_padded": bool(sampling_meta.get("padded", False)),
        },
    }
    return breakdown


def save_pipeline_artifacts(perception: dict[str, Any], meta: dict[str, Any], output_dir: str | Path, *, no_show: bool = True) -> dict[str, Any]:
    """Write the required image/PLY/numeric artifacts for one inference."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _save_png(output / "left_ir.png", perception["left_ir"])
    _save_png(output / "right_ir.png", perception["right_ir"])
    np.save(output / "disparity.npy", _array(perception["disparity"]).astype(np.float32))
    np.save(output / "depth_m.npy", _array(perception["depth"]).astype(np.float32))
    _save_png(output / "disparity.png", perception["disparity"])
    _save_png(output / "depth.png", perception["depth"])
    _save_png(output / "valid_mask.png", perception["valid_mask"], mask=True)
    breakdown = _stage_count_breakdown(perception, meta)
    _save_png(output / "invalid_disparity_mask.png", _invalid_disparity_mask(_array(perception["disparity"]), float(meta.get("ffs", {}).get("min_disparity_px", 0.001))), mask=True)
    _save_png(output / "remove_invisible_mask.png", _remove_invisible_mask(_array(perception["disparity"]), float(meta.get("ffs", {}).get("min_disparity_px", 0.001))), mask=True)
    _save_png(output / "z_range_invalid_mask.png", _z_range_invalid_mask(_array(perception["depth"]), meta), mask=True)
    save_ascii_ply(perception["raw"], output / "raw.ply")
    save_ascii_ply(perception["cropped"], output / "cropped.ply")
    save_ascii_ply(perception["sampled"], output / "sampled.ply")
    meta.setdefault("visualization", {}).update(breakdown)
    (output / "metadata.json").write_text(json.dumps(_jsonable(meta), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "stage_counts.json").write_text(json.dumps(_jsonable(breakdown), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    timing = meta.get("ffs", {}).get("timing_ms", {})
    (output / "timing.json").write_text(json.dumps(_jsonable(timing), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "output_dir": str(output),
        "files": sorted(path.name for path in output.iterdir() if path.is_file()),
        "point_counts": {name: int(_array(perception[name]).shape[0]) for name in ("raw", "cropped", "sampled")},
        "stage_counts": breakdown,
    }
    if not no_show:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 4, figsize=(16, 8))
        panels = (("left_ir", "left IR"), ("right_ir", "right IR"), ("disparity", "disparity"), ("depth", "metric depth"), ("valid_mask", "valid mask"))
        for axis, (name, title) in zip(axes.flat, panels):
            axis.imshow(_array(perception[name]), cmap="gray")
            axis.set_title(title)
            axis.axis("off")
        axes.flat[5].axis("off")
        axes.flat[6].axis("off")
        axes.flat[7].axis("off")
        plt.tight_layout()
        plt.show()
        for name in ("raw", "cropped", "sampled"):
            show_open3d(perception[name])
    return result


def _remove_invisible_mask(disparity: np.ndarray, epsilon: float) -> np.ndarray:
    if disparity.ndim != 2:
        return np.zeros_like(disparity, dtype=bool)
    finite_positive = np.isfinite(disparity) & (disparity > epsilon)
    u = np.arange(disparity.shape[1], dtype=np.float32)[None, :]
    return finite_positive & ((u - disparity) < 0.0)


def _invalid_disparity_mask(disparity: np.ndarray, epsilon: float) -> np.ndarray:
    return ~(np.isfinite(disparity) & (disparity > epsilon))


def _z_range_invalid_mask(depth: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    ffs_meta = meta.get("ffs", {}) if isinstance(meta.get("ffs", {}), dict) else {}
    min_depth = float(ffs_meta.get("min_depth_m", 0.0))
    max_depth = ffs_meta.get("max_depth_m")
    return np.isfinite(depth) & ((depth < min_depth) | ((depth > float(max_depth)) if max_depth is not None else False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="NPZ/NPY containing left_ir, right_ir, and optional rgb")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    builder = PointCloudBuilder.from_yaml(args.config)
    perception, meta = builder.build_perception_stages(load_frame(args.input))
    print(json.dumps(_jsonable(save_pipeline_artifacts(perception, meta, args.output_dir, no_show=args.no_show)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
