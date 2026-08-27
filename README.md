# PointCloudBuilder

PointCloudBuilder is a deployment-oriented RGB-D geometry pipeline for robot learning.
It supports single-camera point-cloud construction, CameraRig fixed-camera integration,
native depth and FFS stereo, real concurrent fixed multi-camera capture, host-time frame
matching, workspace transforms, deterministic current-snapshot fusion, independent
Rerun visualization, and optional persistent TSDF mapping. Realtime policy input stays
decoupled from visualization and map maintenance.

## CameraRig, workspace, and rig fusion

CameraRig is pinned to the reviewed `develop` deployment-readiness commit under
`third_party/CameraRig` and owns one camera's
frame, calibration, and fixed-mount bundle. PointCloudBuilder consumes only the stable
`camera_rig.api`, defines native depth XYZ in the depth optical frame and FFS XYZ in
the left-IR optical frame, then owns workspace transformation, the versioned
multi-camera list, host-timestamp matching, deterministic snapshot voxel fusion, and
one global post-fusion sampling step.

The rig pipeline exposes per-camera camera/workspace clouds, concatenated and cropped
workspace clouds, fused voxel centroids, the final sampled tensor, and a provenance
sidecar. The policy-facing path remains current-snapshot only. A separate optional
Open3D process consumes same-pass per-camera depth rays to maintain a persistent map;
the 4096-point policy tensor is never used as TSDF input. See `docs/camera-rig-integration.md`,
`docs/offline-rig-orchestration.md`, `docs/live-rig-acquisition.md`, and
`docs/workspace-fusion.md`. Real dual-camera snapshot acceptance is documented in
`docs/real-multicamera-fusion.md`.

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

Optional independent Rerun and persistent TSDF dependencies are pinned to the
versions validated on Python 3.10:

```bash
python -m pip install -e ".[rerun]"  # rerun-sdk==0.36.3
python -m pip install -e ".[tsdf]"   # open3d==0.19.0
```

## Deployment-ready multi-camera mapping

Initialize the reviewed CameraRig pin before installation:

```bash
git submodule update --init --recursive
python -m pip install -e third_party/CameraRig
```

Keep runtime YAML, provision bundles, and reports private. Before a dual-camera run,
inspect USB link/profile state, perform the pose-free target preflight, provision each
fixed camera against the same resolved target, and validate each provision:

```bash
camera-rig device inspect --config .local/camera_a/runtime.yaml --show-profiles
camera-rig target preflight --camera-config .local/camera_a/runtime.yaml \
  --target .local/target/target_spec.json --frames 60 \
  --policy pose_validated --report .local/reports/camera_a_preflight.json \
  --overlays .local/overlays/camera_a
camera-rig provision fixed ...
camera-rig provision validate ...
```

Repeat the preflight/provision pair for camera B, then run bounded concurrent capture
and host-time matching before enabling fusion:

```bash
python tools/mapping/run_live_rig.py \
  --rig-config .local/configs/live_rig.yaml \
  --mapping-config .local/configs/mapping.yaml \
  --frames 1000 --reopen-frames 60 \
  --output .local/evidence/live_rig \
  --report .local/reports/live_rig.json
```

The runtime has two deliberately separate outputs:

```text
matched cameras -> per-camera cloud -> snapshot voxel fusion -> global sampling
matched cameras -> per-camera depth + K + T_workspace_from_camera -> async TSDF map
```

The production persistent-map route is `ffs_stereo` with the TensorRT plugin backend.
It fails closed if its engine/plugin/manifest is unavailable and never falls back to
native depth. Native TSDF is an optional diagnostic baseline; degraded native geometry
does not fail FFS production acceptance. The currently validated legacy ChArUco target
remains the deployment target for camera A/B. The 500 x 700 mm board is a future preset
with status `DEFERRED`; it requires neither real-camera evidence nor reprovisioning for
the current M9 closure.

The first path remains the low-latency policy/dynamic observation. The second is
static workspace history. A frozen TSDF plus the current fused cloud is the default
production composition; `guarded_continuous` is opt-in and masks transient pixels
until a fixed surface persists for the configured frame count.
After a frozen map is loaded and extracted, no live depth packets are sent to it;
only the separate current-snapshot overlay continues to change.

Record native or FFS depth without re-running inference, then reconstruct offline:

```bash
python tools/mapping/record_live_rig_depth.py \
  --config .local/configs/live_rig.yaml \
  --matched-sets 300 --depth-source ffs_stereo \
  --output .local/recordings/rig_depth

python tools/mapping/build_tsdf_offline.py \
  --recording .local/recordings/rig_depth \
  --config configs/mapping/tsdf_example.yaml \
  --output .local/maps/static_tsdf
```

FFS recording manifests checksummed per-camera backend, precision, artifact ID, and
pipeline-config SHA. Production acceptance verifies this lineage through the map's
source-recording receipt and requires `tensorrt_plugin`; native map input is optional.

The persistent map exports the full mesh plus raw, workspace-cropped, and sampled
point clouds. `configs/mapping/tsdf_example.yaml` shows the optional `postprocess`
contract. Omitting it disables both point-cloud stages for backward compatibility;
mesh geometry is never presented as cropped.

Benchmark the same frozen replay frames with reconstruction only, crop, and
crop-plus-sampling scenarios:

```bash
python tools/mapping/benchmark_world_reconstruction.py \
  --rig-config .local/configs/replay_rig.yaml \
  --start-frame 0 --frames 100 --warmup 10 --device cuda \
  --report .local/reports/world_reconstruction_benchmark.json
```

The stable timing schema separates processing-only and capture-inclusive values. Key
fields are `raw_to_world_fused_ms`, `workspace_crop_ms`, `global_sampling_ms`,
`combined_per_camera_sequential_ms`,
`raw_to_world_sampled_ms`, `raw_to_tsdf_update_ms`, `extract_point_cloud_ms`,
`extract_mesh_ms`, `post_crop_ms`, and `post_sampling_ms`; reports aggregate p50, p95,
mean, and max. TSDF update and map extraction remain distinct rates and metrics.
Cold live-map acceptance may use `--build-warmup-sets` to prebuild the map before
opening the fixed performance/RSS window; warmup and measured sets are both recorded.

Run the independent Rerun viewer on the existing snapshot acceptance command:

```bash
python tools/mapping/run_live_rig_fusion.py \
  --rig-config .local/configs/live_rig.yaml \
  --mapping-config .local/configs/mapping_acceptance.yaml \
  --matched-sets 300 --output .local/evidence/fusion \
  --report .local/reports/fusion.json \
  --viewer rerun --rerun-spawn \
  --rerun-record .local/rerun/fusion.rrd \
  --viewer-point-budget 30000
```

For persistent live mapping use `tools/mapping/run_live_tsdf_mapping.py`. Rerun and
TSDF each run in their own spawned process with a size-1/2 latest-only queue; viewer
or mapper lag drops packets instead of building backlog. All real configs, captures,
depth recordings, maps, screenshots, reports, and `.rrd` files belong under ignored
`.local/`. Never commit serial numbers, real extrinsics, images, depth, maps, or RRDs.

Live map publication additionally requires a passing, same-rig/same-count
snapshot-only report via `--snapshot-baseline-report`. The CLI compares processed FPS
and end-to-end p95, blocks publication above 10% FPS loss or 5 ms p95 increase, and
records accepted/rejected submissions plus mapper queue/RSS telemetry. Publication
also requires at least 32 child-RSS samples and, after a 20% warmup, no more than
256 MiB quartile-median growth or 5 MiB per 100 frames fitted growth.
Reports also split end-to-end latency into frame-match wait and snapshot processing
diagnostics without changing the gate.

Coordinate transforms always mean `T_target_from_source`, with column vectors. PCB
stores `T_workspace_from_camera`; Open3D receives its tested inverse
`T_camera_from_workspace`. Native TSDF uses raw `uint16` plus the device scale; FFS
uses rectified metric float depth with invalid pixels equal to zero and scale 1.

Deployment and architecture details:

- [Architecture](docs/architecture.md)
- [Calibration deployment](docs/deployment-calibration.md)
- [Rerun visualization](docs/rerun-visualization.md)
- [Snapshot versus persistent map](docs/current-snapshot-vs-persistent-map.md)
- [TSDF mapping](docs/tsdf-mapping.md)
- [TSDF dynamic handling](docs/tsdf-dynamic-handling.md)

Known environment debt: `sapien 2.2.1` declares `opencv-python`, while this environment
uses the headless OpenCV distribution. This pre-existing `pip check` warning is not
resolved by adding Rerun or TSDF, and no Torch/CUDA/TensorRT/OpenCV upgrade is required.

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
  --serial YOUR_DEVICE_SERIAL \
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
