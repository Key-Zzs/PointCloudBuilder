# PointCloudBuilder

PointCloudBuilder is a lightweight, reusable RGB-D to point-cloud module for
robot learning pipelines. It is designed to be shared by training conversion
jobs and real-time deployment code, so the same camera intrinsics, crop ranges,
sampling policy, and fixed output size are used in both paths.

The project is intentionally small at this stage. It provides an extensible
package skeleton with a minimal working PyTorch tensor pipeline:

- Load YAML camera, crop, sampling, and alignment configuration.
- Back-project depth images into camera-frame XYZ point clouds.
- Optionally attach RGB when aligned depth-to-color mode is enabled.
- Crop by configured 3D bounds.
- Return a fixed number of points with stride, random, FPS, voxel, voxel-random,
  or voxel-FPS sampling modes.
- Fall back to CPU when CUDA is unavailable while keeping the public API stable.

Offline visualization and benchmark scripts are kept outside the real-time path.
The `PointCloudBuilder` runtime package does not depend on Open3D.

## Install

```bash
conda create -n pointcloud-builder python=3.10 -y
conda run -n pointcloud-builder python -m pip install -e ".[dev]"
```

Optional offline visualization dependencies are separate:

```bash
conda run -n pointcloud-builder python -m pip install -e ".[viz]"
```

## Core API

```python
from pointcloud_builder import PointCloudBuilder

builder = PointCloudBuilder.from_yaml("configs/example_head_aligned.yaml")

# Offline zarr or recorded-frame conversion.
pc, meta = builder.from_recorded_frame(frame)

# Real-time inference.
pc, meta = builder.from_live_frame(frame)
```

`frame` may be a mapping with at least a `depth` entry and optionally a `color`
entry:

```python
frame = {
    "depth": depth_image,  # H x W, uint16 millimeters or float depth units
    "color": color_image,  # H x W x 3, optional and used only when aligned
}
```

The returned `pc` is a fixed-size `torch.Tensor` with shape `(num_points, 3)` for
XYZ-only output or `(num_points, 6)` when RGB is enabled. `meta` contains counts
for raw, cropped, and sampled stages plus the selected device and sampling mode.

## Configuration

YAML files define:

- camera intrinsics: `fx`, `fy`, `cx`, `cy`, width, height, and depth scale;
- aligned depth-to-color behavior;
- crop bounds in camera coordinates;
- sampling mode and fixed number of output points;
- device policy, where `auto` means CUDA when available and CPU otherwise.

Example configs live in `configs/`:

- `example_head_aligned.yaml`
- `example_head_depth_raw.yaml`

Training defaults should prefer `voxel_random` or `fps`.
Deployment defaults should prefer `voxel_random` or `voxel_fps`.

## Shared Training and Deployment Constraint

Training and deployment must use the same `PointCloudBuilder` package and the
same YAML schema. Do not duplicate RGB-D conversion, crop, or sampling logic in
training-only zarr conversion scripts or robot-control code. Deployment code may
call `from_live_frame`, while offline conversion may call `from_recorded_frame`,
but both routes resolve to the same builder implementation.

## Tests

```bash
conda run -n pointcloud-builder python -m pytest
```
