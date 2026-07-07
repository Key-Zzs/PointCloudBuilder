# PointCloudBuilder

PointCloudBuilder is a reusable RGB-D to camera-frame point-cloud module for
robot learning pipelines. Training data conversion and real-time deployment must
share the same `PointCloudBuilder` implementation and YAML schema.

This repository currently implements stage 1: raw RGB-D deprojection.

## Stage 1 Scope

- Load camera intrinsics from YAML.
- Use PyTorch tensors for depth deprojection.
- Prefer CUDA when requested and available; fall back to CPU when CUDA is not
  available.
- Select color intrinsics when `camera.aligned_depth_to_color: true`.
- Select depth intrinsics when `camera.aligned_depth_to_color: false`.
- Attach RGB only when depth is aligned to color, `pointcloud.use_rgb: true`,
  `pointcloud.output_format: "xyzrgb"`, and the input frame contains `rgb`.
- Filter invalid `depth <= 0` points.
- Keep Open3D and GUI visualization out of the real-time builder path.

Crop and sampling modules remain in the package as extension points for later
stages, but the stage 1 builder output is the raw point cloud.

## Install

```bash
conda create -n pointcloud-builder python=3.10 -y
conda run -n pointcloud-builder python -m pip install -e ".[dev]"
```

Optional offline visualization dependency:

```bash
conda run -n pointcloud-builder python -m pip install -e ".[viz]"
```

## Core API

```python
from pointcloud_builder import PointCloudBuilder

builder = PointCloudBuilder.from_yaml("configs/example_head_aligned.yaml")

pc, meta = builder.from_recorded_frame(frame)
pc, meta = builder.from_live_frame(frame)
```

`frame` is a mapping with required `depth` and optional `rgb`:

```python
frame = {
    "depth": depth_image,  # H x W numpy array or torch tensor
    "rgb": rgb_image,      # H x W x 3 optional numpy array or torch tensor
    "timestamp": 1.23,
    "global_frame_index": 42,
}
```

`pc` is a `torch.Tensor` shaped `N x 3` for XYZ or `N x 6` for XYZRGB. `meta`
contains `stage`, `aligned_depth_to_color`, `use_rgb`, `num_raw_points`,
`device`, `timestamp`, and `global_frame_index`.

## YAML

```yaml
device: "cuda"

camera:
  name: "head"
  aligned_depth_to_color: true
  depth_scale: 0.001

  color_intrinsics:
    width: 640
    height: 480
    fx: 600.0
    fy: 600.0
    cx: 320.0
    cy: 240.0

  depth_intrinsics:
    width: 640
    height: 480
    fx: 600.0
    fy: 600.0
    cx: 320.0
    cy: 240.0

pointcloud:
  use_rgb: true
  output_format: "xyzrgb"
```

## Offline Visualization

Visualization is intentionally separate from the real-time builder:

```bash
python scripts/visualize_raw_pointcloud.py \
  --config configs/example_head_aligned.yaml \
  --input examples/sample_rgbd.npz
```

## Benchmark

```bash
python scripts/benchmark_deprojection.py \
  --config configs/example_head_aligned.yaml \
  --iters 1000 \
  --warmup 100
```

The benchmark prints p50, p95, and mean latency in milliseconds, plus point
count, device, and image resolution.

## Tests

```bash
pytest -q
python scripts/benchmark_deprojection.py --config configs/example_head_aligned.yaml --iters 100 --warmup 10
```
