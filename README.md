# PointCloudBuilder

PointCloudBuilder is a reusable RGB-D to camera-frame point-cloud module for
robot learning pipelines. Training data conversion and real-time deployment must
share the same `PointCloudBuilder` implementation and YAML schema.

This repository currently implements stage 2: raw RGB-D deprojection plus
workspace crop, plus
offline tools for capturing one aligned D435i RGB-D frame, visualizing the
resulting point cloud, and benchmarking the deprojection/crop/sampling building
blocks.

## Stage 2 Scope

- Load camera intrinsics from YAML.
- Use PyTorch tensors for depth deprojection.
- Prefer CUDA when requested and available; fall back to CPU when CUDA is not
  available.
- Select color intrinsics when `camera.aligned_depth_to_color: true`.
- Select depth intrinsics when `camera.aligned_depth_to_color: false`.
- Attach RGB only when depth is aligned to color, `pointcloud.use_rgb: true`,
  `pointcloud.output_format: "xyzrgb"`, and the input frame contains `rgb`.
- Filter invalid `depth <= 0` points.
- Load workspace crop bounds from YAML.
- Crop `N x 3` XYZ and `N x 6` XYZRGB point clouds by XYZ while preserving RGB
  columns.
- Return an empty `0 x C` tensor without crashing when the crop contains no
  points.
- Expose raw and cropped stages through `build_stages()` for offline debugging.
- Keep Open3D and GUI visualization out of the real-time builder path.
- Capture a single RealSense D435i aligned RGB-D frame as a local `.npz` debug
  artifact and write a matching YAML config from the camera intrinsics.
- Provide fixed-size sampling utilities (`stride`, `random`, `voxel`, `fps`,
  `voxel_random`, `voxel_fps`) for offline testing and later pipeline stages.

The high-level `PointCloudBuilder` output is the cropped point cloud when
`crop.enabled: true`; otherwise it is the raw point cloud.

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

`pc` is a `torch.Tensor` shaped `N x 3` for XYZ or `N x 6` for XYZRGB. With
`crop.enabled: true`, `N` is the cropped point count. `meta` contains `stage`,
`aligned_depth_to_color`, `use_rgb`, `num_raw_points`, `num_cropped_points`,
`crop_enabled`, `crop_range`, `crop_empty`, `device`, `timestamp`, and
`global_frame_index`.

Raw `N` starts as the number of valid depth pixels after filtering `depth <= 0`.
XYZ values are in meters after applying `camera.depth_scale`. RGB values are
normalized to `[0, 1]` when `pointcloud.output_format: "xyzrgb"` is enabled.

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

crop:
  enabled: true
  frame: "camera"
  x: [-0.5, 0.5]
  y: [-0.5, 0.5]
  z: [0.05, 1.5]

sampling:
  mode: voxel_random
  num_points: 1024
  stride: 2
  voxel_size: 0.01
```

## Real D435i One-Frame Capture

Use this path to validate that the local RealSense RGB-D frame matches the data
shape that will later be stored by LeRobot plus RGB-D sidecar data.

`pyrealsense2` is intentionally not a package dependency. Run the camera tools in
an environment that already has the RealSense Python wrapper, for example the
`dual_arm_teleop` environment on the Flexiv workstation:

```bash
cd /home/deepcybo/workspace/3D-Diffusion-Policy/PointCloudBuilder
/home/deepcybo/miniconda3/envs/dual_arm_teleop/bin/python -m pip install -e ".[viz]"
```

Quick camera sanity check:

```bash
/home/deepcybo/miniconda3/envs/dual_arm_teleop/bin/python \
  tools/camera/detect_realsense.py
```

Find the camera serial with `rs-enumerate-devices`, then capture one aligned
RGB-D frame:

```bash
/home/deepcybo/miniconda3/envs/dual_arm_teleop/bin/python \
  tools/camera/capture_d435i_aligned_rgbd.py \
  --serial 344522070241 \
  --width 424 \
  --height 240 \
  --fps 30 \
  --out captures/head_frame_000000.npz \
  --config-out configs/captures/head_aligned.yaml
```

The `.npz` contains:

```text
rgb: uint8 [H, W, 3], depth-to-color aligned color frame
depth: uint16 [H, W], depth aligned to the color pixel grid
rgb_timestamp, depth_timestamp
depth_scale
width, height, fx, fy, cx, cy
```

The generated YAML uses the color intrinsics because
`camera.aligned_depth_to_color: true`. Heavy capture artifacts under `captures/`
are ignored by `.gitignore`; generated YAML files under `configs/captures/` can
be kept for reproducible local tests.

## Offline Visualization

Visualization is intentionally separate from the real-time builder:

```bash
python scripts/visualize_raw_pointcloud.py \
  --config configs/captures/head_aligned.yaml \
  --input captures/head_frame_000000.npz \
  --output captures/head_raw.ply
```

Disable the Open3D window when running headless:

```bash
python scripts/visualize_raw_pointcloud.py \
  --config configs/captures/head_aligned.yaml \
  --input captures/head_frame_000000.npz \
  --output captures/head_raw.ply \
  --no-show
```

Visualize raw and cropped stages:

```bash
python scripts/visualize_cropped_pointcloud.py \
  --config configs/captures/head_aligned.yaml \
  --input captures/head_frame_000000.npz \
  --raw-output captures/head_raw.ply \
  --output captures/head_cropped.ply
```

## Benchmark

Benchmark raw deprojection with the captured camera resolution and intrinsics:

```bash
python scripts/benchmark_deprojection.py \
  --config configs/captures/head_aligned.yaml \
  --iters 1000 \
  --warmup 100
```

The benchmark prints p50, p95, and mean latency in milliseconds, plus point
count, device, and image resolution.

Crop and sampling utilities can be benchmarked independently:

```bash
python scripts/benchmark_crop.py \
  --config configs/example_head_aligned.yaml \
  --num-points 307200 \
  --iters 1000 \
  --warmup 100

python scripts/benchmark_sampling.py \
  --config configs/example_train_voxel_random.yaml \
  --points 20000 \
  --iterations 20
```

## Data Boundary

The `.npz` capture format is for one-frame debugging and visualization only. It
is not the planned LeRobot dataset format. For later integration, RGB should
remain in LeRobot video fields, while depth/IR should be stored in a sidecar
array store such as zarr and joined by `episode_index`, `frame_index`, and
`camera_name`. PointCloudBuilder should be reused by both offline conversion and
real-time deployment so training and inference share the same deprojection
configuration.

## Tests

```bash
pytest -q
python scripts/benchmark_deprojection.py --config configs/example_head_aligned.yaml --iters 100 --warmup 10
python scripts/benchmark_crop.py --config configs/example_head_aligned.yaml --num-points 307200 --iters 100 --warmup 10
```
