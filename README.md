# PointCloudBuilder

PointCloudBuilder is a plug-and-play real-time RGB-D point cloud construction module for robot learning pipelines. It uses PyTorch CUDA tensors to deproject RGB-D frames into camera-frame point clouds from YAML-configured camera intrinsics, optionally produces RGB point clouds when aligned_depth_to_color is enabled, applies YAML-configured workspace cropping, and outputs fixed-size point clouds using FPS, stride, random, voxel, voxel_random, or voxel_fps sampling. Offline visualization and benchmark tools are provided for raw, cropped, and sampled point clouds, while realtime control remains decoupled from visualization.

Training and deployment must share the same PointCloudBuilder.

## Repository Description

This repository provides one reusable pipeline for offline training data conversion and realtime robot inference:

1. deproject RGB-D to raw camera-frame point cloud;
2. crop point cloud by YAML workspace bounds;
3. sample point cloud to a fixed number of points;
4. return a `torch.Tensor` point cloud and metadata.

The realtime builder path does not import Open3D, matplotlib, or GUI code. Visualization lives in offline scripts and `pointcloud_builder.visualization`.

## Installation

```bash
python -m pip install -e ".[dev]"
```

Optional offline visualization dependency:

```bash
python -m pip install -e ".[viz]"
```

## Quick Start

```python
import torch

from pointcloud_builder import PointCloudBuilder

builder = PointCloudBuilder.from_yaml("configs/example_head_aligned.yaml")

depth = torch.ones((builder.camera.height, builder.camera.width), dtype=torch.float32)
rgb = torch.ones((builder.camera.height, builder.camera.width, 3), dtype=torch.float32)
frame = {"depth": depth, "rgb": rgb, "timestamp": 1.23, "global_frame_index": 42}

pc, meta = builder.from_live_frame(frame)
print(pc.shape)
print(meta["sampling_mode"], meta["num_sampled_points"])
```

Stable public API:

```python
from pointcloud_builder import PointCloudBuilder

builder = PointCloudBuilder.from_yaml(config_path)

pc, meta = builder.from_recorded_frame(frame)
pc, meta = builder.from_live_frame(frame)
```

`from_recorded_frame` and `from_live_frame` share the same internal pipeline. All point cloud outputs are `torch.Tensor` objects.

## YAML Config Example

The repository includes:

- `configs/example_head_aligned.yaml`
- `configs/example_head_depth_raw.yaml`
- `configs/example_train_voxel_random.yaml`
- `configs/example_deploy_voxel_fps.yaml`

Example:

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
  enabled: true
  mode: "voxel_random"
  num_points: 1024
  stride: 2
  voxel_size: 0.005
  seed: 42
  deterministic: false
  pad_mode: "repeat"
```

`device: "cuda"` gracefully falls back to CPU when CUDA is unavailable.

## Offline Zarr Conversion Example

Use the recorded-frame API inside the dataset conversion loop. The builder call is independent of the storage backend:

```python
from pointcloud_builder import PointCloudBuilder

builder = PointCloudBuilder.from_yaml("configs/example_train_voxel_random.yaml")

def convert_recorded_frame(frame: dict[str, object]) -> tuple[object, dict[str, object]]:
    pc, meta = builder.from_recorded_frame(frame)
    return pc, meta
```

`examples/export_zarr_example.py` contains the same minimal conversion helper.

## Realtime Inference Example

```python
import torch

from pointcloud_builder import PointCloudBuilder

builder = PointCloudBuilder.from_yaml("configs/example_deploy_voxel_fps.yaml")

frame = {
    "depth": torch.ones((builder.camera.height, builder.camera.width), dtype=torch.float32),
    "rgb": torch.ones((builder.camera.height, builder.camera.width, 3), dtype=torch.float32),
}
pc, meta = builder.from_live_frame(frame)
```

Realtime control code should only depend on `pointcloud_builder.PointCloudBuilder`, not visualization scripts.

## Sampling Modes Explanation

- `fps`: farthest point sampling over XYZ.
- `stride`: select points at a fixed interval, then pad or trim.
- `random`: random sample without replacement when enough points exist.
- `voxel`: voxel downsample by XYZ, keep one representative per voxel, then pad or trim.
- `voxel_random`: voxel downsample first, then random sample to fixed size.
- `voxel_fps`: voxel downsample first, then FPS to fixed size.

Training default: `voxel_random` or `fps`.

Deployment default: `voxel_random` or `voxel_fps`.

## Aligned Depth To Color Explanation

When `camera.aligned_depth_to_color: true`, depth is interpreted on the color pixel grid and deprojected with `color_intrinsics`. RGB columns are attached only when all of these are true:

- `camera.aligned_depth_to_color: true`
- `pointcloud.use_rgb: true`
- the frame contains `rgb` or `color`

When `camera.aligned_depth_to_color: false`, depth is deprojected with `depth_intrinsics` and the output remains XYZ even if the frame also contains RGB.

## Fixed-Size Output Explanation

The public builder output is the sampled point cloud. With the provided configs, output shape is always:

- `sampling.num_points x 3` for XYZ;
- `sampling.num_points x 6` for XYZRGB.

If the sampler receives fewer than `sampling.num_points`, it pads with repeated points or zeros according to `sampling.pad_mode`.

## Empty Crop No-Crash Behavior Explanation

If cropping removes every point, the crop stage returns an empty `0 x C` tensor and sampling returns a fixed-size zero tensor. The builder does not crash; metadata marks `crop_empty`, `input_empty`, and `padded`.

## Offline Visualization Commands

Visualization is offline only.

```bash
python scripts/visualize_raw_pointcloud.py \
  --config configs/example_head_aligned.yaml \
  --input captures/head_frame_000000.npz \
  --output captures/head_raw.ply \
  --no-show

python scripts/visualize_cropped_pointcloud.py \
  --config configs/example_head_aligned.yaml \
  --input captures/head_frame_000000.npz \
  --raw-output captures/head_raw.ply \
  --output captures/head_cropped.ply \
  --no-show

python scripts/visualize_sampled_pointcloud.py \
  --config configs/example_train_voxel_random.yaml \
  --input captures/head_frame_000000.npz \
  --raw-output captures/head_raw.ply \
  --cropped-output captures/head_cropped.ply \
  --output captures/head_sampled.ply \
  --no-show
```

Required offline script names:

- `visualize_raw_pointcloud.py`
- `visualize_cropped_pointcloud.py`
- `visualize_sampled_pointcloud.py`

## Benchmark Commands

CUDA is used when available. CPU fallback is allowed and must not crash.

```bash
python scripts/benchmark_deprojection.py --config configs/example_head_aligned.yaml --iters 20 --warmup 5
python scripts/benchmark_crop.py --config configs/example_head_aligned.yaml --num-points 307200 --iters 20 --warmup 5
python scripts/benchmark_sampling.py --num-points 50000 --target-num-points 1024 --iters 20 --warmup 5
python scripts/benchmark_full_pipeline.py --config configs/example_train_voxel_random.yaml --iters 20 --warmup 5
```

Required benchmark script names:

- `benchmark_deprojection.py`
- `benchmark_crop.py`
- `benchmark_sampling.py`
- `benchmark_full_pipeline.py`

## Real D435i One-Frame Capture

`pyrealsense2` is intentionally not a package dependency. Run the camera tools in an environment that already has the RealSense Python wrapper:

```bash
python tools/camera/detect_realsense.py

python tools/camera/capture_d435i_aligned_rgbd.py \
  --serial 344522070241 \
  --width 424 \
  --height 240 \
  --fps 30 \
  --out captures/head_frame_000000.npz \
  --config-out configs/captures/head_aligned.yaml
```

The `.npz` contains `rgb`, `depth`, timestamps, depth scale, and camera intrinsics. The generated YAML uses color intrinsics because `camera.aligned_depth_to_color: true`.

## Tests

```bash
pip install -e .
pytest -q
```

## Fast-FoundationStereo depth source

FFS is optional. The default `depth_source.mode=frame` keeps the existing
native RGB-D path and public Builder API unchanged. `mode=ffs_stereo` consumes
a rectified `480x640` IR1/IR2 pair, estimates metric depth, and then reuses the
same deprojection, crop, and sampling implementation.

Available routes are `pytorch`, `tensorrt_single`, `tensorrt_two_stage`, and
`tensorrt_plugin`. There is no silent backend or precision fallback. The
copied FFS code remains under NVIDIA's non-commercial research license.

The verified optional environment is the existing `dp3` environment with
Python 3.10, PyTorch 2.11/CUDA 13, and TensorRT 10.16.1.11:

```bash
cd ~/workspace/3D-Diffusion-Policy/PointCloudBuilder
export PY=~/miniconda3/envs/dp3/bin/python

PYTHONNOUSERSITE=1 "$PY" -m pip install -e '.[dev,viz]'
PYTHONNOUSERSITE=1 "$PY" -m pip install \
  timm==1.0.28 onnx==1.18.0 onnxscript==0.5.6 \
  imageio opencv-python-headless pyarrow av
```

The checkpoint, ONNX, Engines, plugin library, and build outputs are
gitignored. The official checkpoint can be downloaded again; ONNX, manifests,
Engines, and the plugin can be regenerated in `dp3`. TensorRT Engines must be
rebuilt on the target TensorRT/GPU stack.

After restoring the checkpoint, a PyTorch smoke requires no TensorRT build:

```bash
PYTHONNOUSERSITE=1 "$PY" scripts/run_v05_ffs_frame.py \
  --dataset-root ~/.cache/huggingface/lerobot/flexiv_dual_arm_3d/pick_place_20260713_v05 \
  --camera head --global-frame-index 0 --backend pytorch \
  --builder-config ffs_reproduction/configs/v05_ffs.yaml \
  --artifact-id fp16_o3 --precision fp16 \
  --builder-optimization-level 3 --workspace-gib 8 \
  --output-dir ffs_reproduction/outputs/v05_verify --no-show
```

Download instructions, fresh-clone recovery, all TensorRT build commands,
four-route smoke/parity checks, and simultaneous raw/cropped/sampled Open3D
visualization are documented in the dedicated guides:

- [English FFS reproduction and deployment guide](ffs_reproduction/README.md)
- [中文 FFS 复现、构建与可视化指南](ffs_reproduction/README_zh-CN.md)
