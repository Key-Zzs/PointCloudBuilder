# PointCloudBuilder

PointCloudBuilder is a fixed multi-camera RGB-D reconstruction system supporting
CameraRig calibration, FFS stereo depth, metric XYZ/XYZRGB reconstruction, workspace
fusion, optional crop/sampling, Rerun visualization, and persistent TSDF mapping.

[中文手册](README_zh-CN.md)

## 1. Overview

The production route is two fixed Intel RealSense D435i cameras, a shared calibrated
workspace, Fast-FoundationStereo (FFS) TensorRT-plugin FP16 depth, dense XYZRGB voxel
fusion, and an independent Open3D TSDF mapper. Reconstruction tensors, visualization,
and persistent maps remain separate outputs.

## 2. Architecture

```text
CameraRig frames -> FFS depth -> camera-frame XYZRGB -> local crop
-> T_workspace_from_camera -> workspace crop -> canonical concatenate
-> voxel centroid fusion -> optional global FPS -> current snapshot

same-pass per-camera depth + K + T_workspace_from_camera
-> independent TSDF process -> extract/crop/sample/mesh -> persistent map
```

TSDF never consumes the fused or sampled point cloud. Rerun uses a bounded latest-only
process queue and cannot change reconstruction tensors.

## 3. Supported hardware

- Two fixed RealSense D435i cameras on USB 3 links.
- NVIDIA GPU compatible with the selected PyTorch, CUDA and TensorRT packages.
- One known ChArUco target shared by both fixed-camera provisions.
- Linux is the validated deployment platform.

## 4. Coordinate conventions

Transforms are named `T_target_from_source` and act on column vectors. Native depth
XYZ starts in `<camera>/depth_optical`; FFS XYZ starts in `<camera>/ir_left_optical`.
PCB stores `T_workspace_from_camera`; Open3D receives the tested inverse
`T_camera_from_workspace`. XYZ is `float32` metres. RGB is float RGB in `[0,1]`.

## 5. Clone

```bash
git clone --branch develop/mapping --recurse-submodules \
  https://github.com/Key-Zzs/PointCloudBuilder.git PointCloudBuilder
cd PointCloudBuilder
git submodule update --init --recursive
```

## 6. Clean environment setup

The standard environment is `pcb-reconstruction`; `dp3` is not part of the public
setup contract. Its declarative specification is `environment.reconstruction.yml`.

```bash
./scripts/bootstrap_reconstruction_env.sh
conda activate pcb-reconstruction
```

Set `PCB_ENV_NAME=my-env` to choose another isolated name. The environment pins the
critical Python/PyTorch/CUDA/TensorRT/OpenCV ABI and installs exactly one `cv2`
provider: `opencv-contrib-python-headless==4.14.0.94`. It also pins OmegaConf,
which is required to deserialize the official FFS checkpoint metadata.

## 7. Doctor

```bash
python scripts/doctor_reconstruction_env.py --no-hardware
python scripts/doctor_reconstruction_env.py --asset-root .local/ffs
```

The full doctor checks Python, Conda, Torch/CUDA/GPU, TensorRT, OpenCV/ArUco,
pyrealsense2, Open3D, Rerun, CameraRig, two D435i devices, USB 3 descriptors, and the
private FFS bundle. It never prints camera serial numbers.

## 8. Camera discovery

```bash
camera-rig device list
camera-rig device inspect --config .local/camera_a/runtime.yaml --show-profiles
python tools/mapping/check_usb_topology.py \
  --identity-map .local/camera_rig/camera_identity_map.json \
  --report .local/reports/usb-topology.json
```

Discover identities interactively, then write serials only to ignored `.local/` YAML.

## 9. Calibration target

Use one target specification for both cameras. The validated standard target is
`charuco_a4_v1`: `DICT_5X5_100`, 7x5 squares, 30 mm squares, 22 mm markers, and
`legacy_pattern=false`. Generate/inspect it with CameraRig, print at 100% scale, and
preserve `target_spec.json` beside the physical target. Its frame has +X right, +Y up,
and +Z out of the board.

```bash
mkdir -p .local/target
camera-rig target generate \
  --config third_party/CameraRig/configs/targets/charuco_a4_v1.yaml \
  --output .local/target/charuco_a4_v1
camera-rig target inspect \
  --target .local/target/charuco_a4_v1/target_spec.json
```

Regenerating this artifact is sufficient when the already printed board is unchanged;
compare the resolved specification and print-scale rulers before preflight.

## 10. Fixed camera calibration

For each camera, run pose-only preflight before provisioning; use the same workspace
and target for camera A and B.

```bash
camera-rig target preflight --camera-config .local/camera_a/runtime.yaml \
  --target .local/target/charuco_a4_v1/target_spec.json \
  --frames 60 --policy pose_validated \
  --report .local/reports/camera_a-preflight.json \
  --overlays .local/overlays/camera_a
camera-rig provision fixed \
  --config .local/camera_a/fixed_provision.yaml \
  --output .local/camera_a/provision
camera-rig provision validate --artifact .local/camera_a/provision
```

Repeat those commands for camera B. Start private runtime YAML from
`third_party/CameraRig/configs/examples/single_camera_contract.yaml` and provision YAML
from `third_party/CameraRig/configs/examples/fixed_provision_contract.yaml`; insert only
the discovered serial under `.local/` and point both provisions at the same target.

Never reuse extrinsics after a failed physical coverage gate.

## 11. FFS setup

FFS assets are external and private. Place the official `20-30-48` checkpoint and
`cfg.yaml` under `.local/ffs/artifacts/`. Expected SHA-256 values are:

```text
model_best_bp2_serialize.pth  98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692
cfg.yaml                       d45afe99b176454d5aff416edf16c8da6a99579f8f374b927f37907442a7d6bc
```

The source is the [official NVlabs repository](https://github.com/NVlabs/Fast-FoundationStereo)
and its [official weights folder](https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap?usp=drive_link).
Download `20-30-48` in a browser, then manually copy the two named files into the
destination above. TensorRT C++ headers can be obtained without an old checkout:

```bash
git clone --depth 1 --branch v10.16 --filter=blob:none --sparse \
  https://github.com/NVIDIA/TensorRT.git .local/third_party/TensorRT
git -C .local/third_party/TensorRT sparse-checkout set include
```

Check or build all routes without importing a sibling repository at runtime:

```bash
python scripts/prepare_ffs_assets.py --check --asset-root .local/ffs
python scripts/prepare_ffs_assets.py --build-tensorrt \
  --asset-root .local/ffs \
  --checkpoint path/to/20-30-48/model_best_bp2_serialize.pth \
  --model-config path/to/20-30-48/cfg.yaml \
  --tensorrt-root .local/third_party/TensorRT
```

Smoke `pytorch`, `tensorrt_single`, `tensorrt_two_stage`, then
`tensorrt_plugin`; production uses only `tensorrt_plugin` FP16. The smoke CLI accepts
`--artifact-dir .local/ffs/artifacts` and
`--plugin-library .local/ffs/build/libffs_gwc_plugin.so`. For each backend, create a
private pipeline YAML from `configs/mapping/ffs_workspace_example.yaml` with that
backend and its checked asset paths, then run the same fresh CameraRig NPZ frame:

```bash
for backend in pytorch tensorrt_single tensorrt_two_stage tensorrt_plugin; do
  python scripts/run_ffs_stereo_frame.py \
    --config ".local/configs/ffs_${backend}.yaml" \
    --input .local/captures/camera_a/frames/frame_000000.npz \
    --output-dir ".local/evidence/ffs-${backend}" --no-show
done
```

## 12. Single-camera XYZ/XYZRGB

```bash
python tools/mapping/run_live_single_camera.py \
  --camera-config .local/camera_a/runtime.yaml \
  --provision .local/camera_a/provision \
  --mapping-config .local/configs/mapping.yaml \
  --ffs-config .local/configs/ffs_tensorrt_plugin.yaml \
  --depth-source ffs_stereo --frames 300 \
  --output .local/evidence/camera_a \
  --report .local/reports/camera_a.json
```

Set `pointcloud.use_rgb: true` for Nx6. Depth points outside the color imager are
explicit black RGB, never fabricated colors.

## 13. Multi-camera acquisition

```bash
python tools/mapping/run_live_rig.py \
  --rig-config .local/configs/live_rig_ffs_rgb.yaml \
  --mapping-config .local/configs/mapping.yaml \
  --frames 1000 --reopen-frames 60 \
  --output .local/evidence/live-rig \
  --report .local/reports/live-rig.json
```

Camera sessions are worker-owned; open/capture/close stay on the same worker thread.

## 14. Frame matching

Live rigs match only `host_receive_timestamp_ns` with the configured maximum skew.
Device timestamps and frame numbers are diagnostics. Buffers are bounded and latest
biased; a frame is never reused across matched sets.

## 15. Raw concatenation

Start from `configs/mapping/raw_rgb_concatenation_example.yaml`. It has fusion OFF and
sampling OFF, exposing dense `/clouds/concatenated` for calibration overlap/debugging.

## 16. Dense RGB fusion

`configs/mapping/dense_rgb_reconstruction_example.yaml` is the recommended profile:
fusion ON at 2.5 mm and sampling OFF. Voxel keys use XYZ only; output XYZ is the voxel
centroid and output RGB is the arithmetic mean. `/clouds/fused` is variable-length
dense XYZRGB. The 2.5 mm default was selected by same-input 2.5/5/10 mm benchmarking:
it materially restored detail without a meaningful fusion-p95 penalty.

Voxel fusion and sampling are different operations:

- Voxel fusion merges overlapping multi-camera observations into voxel centroids.
- Sampling optionally reduces output size.

Recommended dense reconstruction is fusion ON, sampling OFF.

## 17. Crop

The production order is optional camera-frame local crop, workspace transform, one
workspace crop per camera, canonical concatenate, voxel fusion, then optional global
sampling. Crop uses XYZ and preserves RGB columns.

## 18. Optional sampling

`configs/mapping/compact_rgb_reconstruction_example.yaml` applies one global
30,000-point FPS after fusion. It preserves each selected point's RGB. `voxel_fps` and
`voxel_random` remain explicit advanced compatibility modes, but are not defaults for
an already voxel-fused cloud.

## 19. Rerun live visualization

Rerun shows camera RGB/frustums, per-camera workspace clouds, concatenated/fused/
sampled clouds, TSDF entities and scalar metrics. Nx6 uses real RGB; Nx3 uses the
default visualization color. `viewer_point_budget` limits only the packet sent to
Rerun and never modifies the reconstruction tensor.

## 20. Interactive infinite mode

Daily operator command:

```bash
python tools/mapping/run_live_tsdf_mapping.py \
  --rig-config .local/configs/live_rig_ffs_rgb.yaml \
  --tsdf-config .local/configs/tsdf_frozen_ffs.yaml \
  --initial-map .local/maps/static_ffs \
  --interactive
```

It spawns Rerun, requires no `--report`, has no matched-set limit, keeps bounded rolling
statistics, and exits 0 after Ctrl+C/SIGTERM cleanup. Override with `--rerun-connect`,
`--rerun-record`, or `--viewer-point-budget 200000`. Interactive mode is not formal
acceptance. Finite mode remains available with `--matched-sets 300 --report ...`.

## 21. Depth recording

```bash
python tools/mapping/record_live_rig_depth.py \
  --config .local/configs/live_rig_ffs_rgb.yaml \
  --matched-sets 300 --depth-source ffs_stereo \
  --output .local/recordings/rig-depth
```

The recording contains matched per-camera depth, intrinsics, transforms and backend
provenance from the same pass; it does not rerun FFS.

## 22. Offline TSDF

```bash
python tools/mapping/build_tsdf_offline.py \
  --recording .local/recordings/rig-depth \
  --config configs/mapping/tsdf_example.yaml \
  --output .local/maps/static-ffs
python tools/mapping/extract_tsdf_geometry.py \
  --map .local/maps/static-ffs --output .local/evidence/static-ffs
```

Extraction supports raw point cloud, optional crop/sampling, and mesh.

## 23. Live TSDF

Formal finite mapping supplies a same-rig/same-count passing snapshot baseline and a
private report. `build_static` and `guarded_continuous` integrate per-camera depth;
mapper child failure, queue violations, performance regression, or RSS growth blocks
publication.

## 24. Frozen static plus dynamic overlay

`frozen_static` requires `--initial-map`. The TSDF receives no live depth after load;
the current fused RGB overlay continues updating independently. This is the recommended
static-workspace operator composition.

## 25. Map save/load

```bash
python tools/mapping/validate_tsdf_map.py --map .local/maps/static-ffs
python tools/mapping/extract_tsdf_geometry.py \
  --map .local/maps/static-ffs --output .local/evidence/reloaded-map
```

Artifacts include checksums, resolved config, volume, metrics and extracted geometry.

## 26. Benchmarks

```bash
python tools/mapping/benchmark_fusion_voxels.py \
  --rig-config .local/configs/live_rig_ffs_rgb.yaml \
  --mapping-config .local/configs/mapping.yaml --frames 30 \
  --report .local/reports/fusion-voxel-sweep.json
python tools/mapping/benchmark_world_reconstruction.py \
  --rig-config .local/configs/replay_rig.yaml --frames 100 --warmup 10 \
  --report .local/reports/reconstruction.json
```

Timing separates depth inference, RGB mapping, deprojection, local/workspace crop,
transform, concatenate, voxel fusion, optional sampling, `raw_to_world_fused`, TSDF
update and TSDF extraction.

## 27. Validation

```bash
pytest -q
python -m build
python scripts/check_documented_commands.py
python scripts/doctor_reconstruction_env.py --no-hardware
```

Hardware acceptance additionally covers CameraRig provision, all four FFS backends,
dual-camera RGB, Rerun, offline/live TSDF, save/load, resource plateau and camera reopen.

## 28. Troubleshooting

If `/clouds/fused` looks sparse, check in order:

1. `actual_fused_points`;
2. `fusion.voxel_size_m`;
3. `viewer_point_budget`;
4. whether you selected `/clouds/fused` or `/clouds/sampled`.

The viewer budget never modifies the reconstruction tensor. Black points may be valid
depth outside the color FOV. A worker/FFS/mapper/viewer child error is fatal. Never
relax calibration or geometry thresholds to make a run pass.

## 29. Private local artifacts

All serials, runtime YAML, provision bundles, physical captures, checkpoints, engines,
plugins, reports, maps, screenshots and RRD files belong under ignored `.local/`.
Public examples use placeholders. Do not use `PYTHONPATH` or symlinks into an older
clone during reproduction.

## 30. Future 500x700 deployment board

The 500x700 mm board is `DEFERRED` and is not a clean-room gate. Do not infer its ArUco
dictionary or `legacy_pattern` from dimensions or successful corner detection. Future
deployment must use authoritative generator metadata/creator confirmation, or print a
new board with a known specification and reprovision both cameras.
