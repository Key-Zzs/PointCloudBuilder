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
python -m pip install -e .
python -m pip install pytest
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

`depth_source.mode` defaults to `frame`, so existing RGB-D configurations and
all public builder methods keep their original behavior. Set it to
`ffs_stereo` only for raw 480x640 IR1/IR2 input. The FFS layer converts the
same left/right pair through one of four explicit backends:

| backend | route | input normalization |
| --- | --- | --- |
| `pytorch` | copied PyTorch reference model | internal ImageNet |
| `tensorrt_single` | one ordinary ONNX/TRT engine | builder applies ImageNet |
| `tensorrt_two_stage` | TRT feature + Triton GWC + TRT post | internal ImageNet |
| `tensorrt_plugin` | TRT feature/post graph with `FFSGWCVolume` CUDA plugin | internal ImageNet |

There is no silent fallback between routes. Missing or mismatched checkpoints,
engines, plugins, manifests, shapes, I/O names, precision, or artifact hashes
fail at construction time. The estimator returns full-resolution disparity,
metric depth in meters, a validity mask, calibration/provenance, and timing;
the existing deprojection, RGB mapping, crop, and fixed-size sampling stages
are then reused unchanged. FFS point clouds use IR1 intrinsics and the
IR1-to-color transform when RGB projection is explicitly enabled.

TensorRT precision is an explicit artifact property. The builder also records
`builder_optimization_level: 0..5` and `workspace_gib`. A failed FP16 build is
recorded with its complete traceback and is never relabeled as FP32; an
explicit diagnostic FP32 build uses a separate artifact id such as `fp32_o0`.
The runtime checks the Engine/config contract (shape, max disparity, iterations,
normalization, precision, resource settings, and hashes) and fails on missing
or ambiguous config candidates.

The reproduction package vendors the exact FFS `core/` modules and records
file hashes in `ffs_reproduction/UPSTREAM_SOURCE.json`. Those copied files
remain under the complete NVIDIA license in `ffs_reproduction/LICENSE.txt`;
the FFS route is for non-commercial research use only.

### FFS environment and artifacts

Use the requested `dp3` interpreter for every build or run:

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python -m pip install -e .
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python -m pip install \
  timm==1.0.28 onnx==1.18.0 onnxscript==0.5.6
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python -m pip install \
  --extra-index-url https://pypi.nvidia.com/ --no-deps \
  tensorrt_cu13_bindings==10.16.1.11 tensorrt_cu13_libs==10.16.1.11
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python -m pip install \
  --extra-index-url https://pypi.nvidia.com/ --no-deps \
  tensorrt-cu13==10.16.1.11
```

Prepare the trusted checkpoint/config, fixed-shape ONNX graphs, and manifests:

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python \
  scripts/prepare_ffs_artifacts.py --skip-tensorrt
```

By default this reads the repository-local
`ffs_reproduction/artifacts/model_best_bp2_serialize.pth` and `cfg.yaml`; it
does not access a Fast-FoundationStereo checkout. `--source-root` is only an
optional one-time import path for an empty artifact directory.
The artifact and build directories are intentionally gitignored, so preserve
or separately restore the checkpoint, ONNX/Engine files, plugin library, and
manifests when moving to a fresh clone.

Existing derived ONNX/engine/manifest files are never silently overwritten;
use `--force` only for an explicit rebuild.

Build the local CUDA plugin for the RTX 5080's SM120 target after the
TensorRT C++ headers are available:

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python \
  scripts/build_ffs_plugin.py \
  --tensorrt-root ffs_reproduction/tensorrt_sdk
```

Then build all TensorRT engines through the TensorRT Python API (the helper
does not invoke `trtexec`):

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python \
  scripts/prepare_ffs_artifacts.py
```

The ONNX exporter is deliberately pinned to the legacy PyTorch exporter with
`opset_version=17`, static `480x640` shapes, and `dynamo=False`. Every ordinary
single export is checked with `onnx.checker.check_model`, its I/O contract is
validated, and the resulting graph is parsed by the target TensorRT Python API
before the engine is accepted. A future Dynamo attempt must pass the same full
ONNX/TRT/parity checks before this pin is changed.

The default variant is named `fp16_o3`. To make an explicit FP32/o0
diagnostic variant after an FP16 failure, run a separate command; it does not
overwrite the FP16 artifacts:

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python \
  scripts/prepare_ffs_artifacts.py --precision fp32 \
  --builder-optimization-level 0 --artifact-suffix fp32_o0 --force
```

### FFS YAML contract

```yaml
device: cuda
camera:
  name: head
  aligned_depth_to_color: false
  color_intrinsics: {width: 640, height: 480, fx: 606.17749, fy: 606.63562, cx: 320.82126, cy: 256.06561}
  depth_intrinsics: {width: 640, height: 480, fx: 392.50119, fy: 392.50119, cx: 316.79514, cy: 235.79944}
pointcloud: {use_rgb: false, output_format: xyz}
depth_source:
  mode: ffs_stereo
  ffs:
    backend: pytorch
    checkpoint_path: ffs_reproduction/artifacts/model_best_bp2_serialize.pth
    model_config_path: ffs_reproduction/artifacts/cfg.yaml
    calibration_path: ~/.cache/huggingface/lerobot/<dataset>/meta/realsense_calibration.json
    calibration_camera: head
    width: 640
    height: 480
    max_disp: 192
    valid_iters: 8
    precision: fp16
    builder_optimization_level: 3
    workspace_gib: 8.0
    artifact_id: fp16_o3
```

The live/offline frame must contain `left_ir` and `right_ir` as raw uint8
`480x640` arrays (or grayscale tensors in the inclusive 0..255 range). The
current calibration gate accepts only the v05 identity/no-op rectified
contract: equal IR intrinsics, zero distortion, identity rotation, and
`IR1 -> IR2 = (-baseline, 0, 0)`. Non-identity calibration is rejected rather
than silently rectified. Native depth is used only for the diagnostic parity
report, never as the FFS builder input.

The online call is the same API as the old RGB-D path:

```python
from pointcloud_builder import PointCloudBuilder, StereoIRFrame

builder = PointCloudBuilder.from_yaml("ffs_reproduction/configs/v05_ffs.yaml")
frame = StereoIRFrame(left_ir=left_ir, right_ir=right_ir, timestamp=timestamp)
point_cloud, metadata = builder.from_live_frame(frame)
```

### v05 frame, visualization, parity, and benchmark

The v05 helper reads the authoritative raw RGB-D sidecar through the existing
parent reader; it does not modify the parent repository:

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python \
  scripts/run_v05_ffs_frame.py \
  --dataset-root ~/.cache/huggingface/lerobot/flexiv_dual_arm_3d/pick_place_20260713_v05 \
  --camera head --global-frame-index 0 --backend pytorch \
  --builder-config ffs_reproduction/configs/v05_ffs.yaml \
  --output-dir ffs_reproduction/outputs/v05 --no-show
```

`scripts/visualize_ffs_stereo_pipeline.py` writes IR PNGs, disparity/depth
arrays and PNGs, validity masks for invalid disparity/remove-invisible/z-range,
raw/cropped/sampled PLYs, metadata, timing, and `stage_counts.json`. The
offline point-cloud contract is `denoise_cloud=false`, `zfar=100`, with no
zfar filtering applied. The report separates FFS invalid disparity,
`remove_invisible`, z-range, Builder crop, and sampling counts.
Use `scripts/compare_ffs_backends.py` for PyTorch-referenced parity and
`scripts/benchmark_ffs_backends.py` for the required 20-warmup/100-run CUDA
benchmark:

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python \
  scripts/compare_ffs_backends.py \
  --dataset-root ~/.cache/huggingface/lerobot/flexiv_dual_arm_3d/pick_place_20260713_v05 \
  --camera head --global-frame-index 0 \
  --builder-config ffs_reproduction/configs/v05_ffs.yaml \
  --artifact-id fp16_o3 \
  --output ffs_reproduction/outputs/v05/parity_all.json

PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python \
  scripts/benchmark_ffs_backends.py \
  --config ffs_reproduction/configs/v05_ffs.yaml \
  --input ~/data/stereo_frame.npz \
  --output ffs_reproduction/outputs/v05/benchmark.json
```

Run `pytest -q` for CPU/fake-backend contract tests and GPU-specific smoke,
parity, and engine checks only after the corresponding TensorRT artifacts are
available. Build products and runtime outputs are intentionally ignored by
Git.
