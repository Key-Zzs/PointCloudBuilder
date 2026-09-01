# PointCloudBuilder

PointCloudBuilder is a fixed multi-camera RGB-D reconstruction system supporting
CameraRig calibration, FFS stereo depth, metric XYZ/XYZRGB reconstruction, workspace
fusion, optional crop/sampling, Rerun visualization, and persistent TSDF mapping.

[中文手册](README_zh-CN.md)

## 1. Overview

The runtime supports `N >= 2` fixed Intel RealSense D435i cameras. The next validated
deployment target is three fixed D435i cameras in one calibrated workspace, using
Fast-FoundationStereo (FFS) TensorRT-plugin FP16 depth, dense XYZRGB voxel fusion, and
an independent Open3D TSDF mapper. Reconstruction tensors, visualization, and
persistent maps remain separate outputs.

The 0.2.0 workflow was reproduced from a fresh clone and isolated environment through
fixed-camera calibration, FFS, dual-camera RGB, Rerun, offline/live TSDF, and map
save/load. Hardware identities and physical evidence remain private under `.local/`.

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

- `N >= 2` fixed RealSense D435i cameras on USB 3 links; the next deployment is three.
- NVIDIA GPU compatible with the selected PyTorch, CUDA and TensorRT packages.
- One known ChArUco target shared by all fixed-camera provisions.
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
which is required to deserialize the official FFS checkpoint metadata. The bootstrap
persists `PYTHONNOUSERSITE=1` in the named Conda environment so packages under
`~/.local` cannot silently override these pins. Reactivate the environment after
rerunning bootstrap. Doctor fails if `cv2` still comes from the user site or its module
version disagrees with the installed wheel.

### 6.1 Moving the same D435i rig to a new computer

This section applies when the camera serials, fixed camera poses, workspace, and
physical calibration target have not changed and only the host is replaced. If any
camera, the workspace, or the target has moved, do not reuse the old extrinsics;
repeat discovery, preflight, and provisioning in Sections 8–10.

#### 6.1.1 Transfer private camera assets

`.local/` is ignored by Git and can contain camera serials, calibration, and local
absolute paths. Transfer it only over trusted encrypted media or a controlled
connection; never commit it. Do not copy the whole old `.local/` tree. Preserve the
relative layout and transfer only what the active rig YAML references:

- `.local/camera_rig/camera_identity_map.json`;
- each camera's runtime YAML and validated provision artifact;
- the target spec/metadata for the unchanged physical target;
- the active production rig, pipeline, and TSDF YAML files;
- the promoted rig-calibration artifact referenced by the rig YAML; and
- only when continuing an old map, the referenced initial map with matching
  calibration provenance.

Old recordings, evidence, logs, FFS smoke output, and old TensorRT Engines are not
required at runtime. After transfer, find old-host absolute paths in YAML/JSON and
replace them with paths relative to the new checkout. Do not edit serials, intrinsics,
extrinsics, or calibration values:

```bash
find .local -type f \( -name '*.yaml' -o -name '*.yml' -o -name '*.json' \) \
  -exec grep -nEH '/home/|/Users/|^[[:space:]]*[A-Za-z]:\\' {} +
camera-rig device list
camera-rig provision validate --artifact .local/camera_rig/camera_a/provision
```

Run `provision validate` for every camera in the rig and confirm that the discovered
devices, identity map, and runtime YAML still identify the same physical D435i units.
If your directory name differs, use the `source.provision_artifact` path from the
production rig YAML. Then run the USB-topology check in Section 8.

#### 6.1.2 Download and verify the official FFS weights

The weights come from the [official NVlabs repository](https://github.com/NVlabs/Fast-FoundationStereo)
and its [official weights folder](https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap?usp=drive_link).
Download the `20-30-48` directory in a browser, or use `gdown` in the activated
`pcb-reconstruction` environment:

```bash
python -m pip install gdown
FFS_DOWNLOAD_DIR="${PWD}/.local/downloads/fast-foundationstereo-weights"
python -m gdown \
  'https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap?usp=drive_link' \
  --folder -O "$FFS_DOWNLOAD_DIR"

FFS_WEIGHT_FILE="$(find "$FFS_DOWNLOAD_DIR" \
  -path '*/20-30-48/model_best_bp2_serialize.pth' -print -quit)"
test -n "$FFS_WEIGHT_FILE"
FFS_WEIGHT_DIR="$(dirname "$FFS_WEIGHT_FILE")"
mkdir -p .local/ffs/artifacts
install -m 0644 "$FFS_WEIGHT_DIR/model_best_bp2_serialize.pth" \
  .local/ffs/artifacts/model_best_bp2_serialize.pth
install -m 0644 "$FFS_WEIGHT_DIR/cfg.yaml" .local/ffs/artifacts/cfg.yaml

printf '%s  %s\n' \
  98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692 \
  .local/ffs/artifacts/model_best_bp2_serialize.pth \
  d45afe99b176454d5aff416edf16c8da6a99579f8f374b927f37907442a7d6bc \
  .local/ffs/artifacts/cfg.yaml | sha256sum -c -
```

For a browser download, manually copy `model_best_bp2_serialize.pth` and `cfg.yaml`
into the same destination and then run the `printf ... | sha256sum -c -` check above.
The checkpoint is loaded with `torch.load(..., weights_only=False)`, so use only the
trusted official files after their hashes pass.

#### 6.1.3 Rebuild TensorRT assets on the new computer

Do not reuse the old computer's `.engine` files or `libffs_gwc_plugin.so`. A default
TensorRT Engine is tied to its build platform, TensorRT version, and GPU compute
capability; see NVIDIA's [Engine Compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html).
Build on the new host with the pinned environment and target GPU. First obtain
matching TensorRT 10.16 C++ headers:

```bash
git clone --depth 1 --branch v10.16 --filter=blob:none --sparse \
  https://github.com/NVIDIA/TensorRT.git .local/third_party/TensorRT
git -C .local/third_party/TensorRT sparse-checkout set include

FFS_CUDA_ARCH="$(python -c \
  'import torch; p=torch.cuda.get_device_capability(); print(f"{p[0]}{p[1]}")')"
printf 'Target CUDA architecture: %s\n' "$FFS_CUDA_ARCH"

python scripts/prepare_ffs_assets.py --build-tensorrt \
  --asset-root .local/ffs \
  --checkpoint .local/ffs/artifacts/model_best_bp2_serialize.pth \
  --model-config .local/ffs/artifacts/cfg.yaml \
  --tensorrt-root .local/third_party/TensorRT \
  --cuda-arch "$FFS_CUDA_ARCH"
python scripts/prepare_ffs_assets.py --check --asset-root .local/ffs
```

Production uses the generated FP16 `tensorrt_plugin` route. To create a pipeline YAML,
copy `configs/mapping/ffs_workspace_example.yaml` under `.local/configs/` and point its
Engine, plugin, manifest, and backend-config fields at the new `.local/ffs/` files;
paths declared from `.local/configs/` can use relative `../ffs/...` paths. Camera
serials and CameraRig calibration are not part of the FFS asset bundle: they continue
to come from the transferred and validated runtime, provision, and rig configuration.
Complete the PyTorch, all three TensorRT-route, and same-fresh-CameraRig-NPZ smokes in
Section 11 before running Doctor below. An asset-check PASS does not replace actual
Engine loading and a camera smoke.

## 7. Doctor

```bash
python scripts/doctor_reconstruction_env.py --no-hardware
python scripts/doctor_reconstruction_env.py --expected-d435i-count 3 \
  --asset-root .local/ffs
```

The full doctor checks Python, Conda, Torch/CUDA/GPU, TensorRT, OpenCV/ArUco,
pyrealsense2, Open3D, Rerun, CameraRig, the configured D435i count, USB 3 descriptors,
and the private FFS bundle. `--no-hardware` remains hardware-independent and the
backward-compatible default expected count is two. The doctor never prints serials.

## 8. Camera discovery

A fresh clone does not contain the Git-ignored `.local/` configuration. Connect only
the three D435i units that belong to this rig. If other RealSense devices are attached,
disconnect them first; never auto-select three devices from a visible set of five.
List devices without opening them:

```bash
camera-rig device list --driver realsense
```

After confirming the three physical identities, create the private identity map. On its
first successful run, the tool assigns `camera_a/b/c` by stable USB physical-port order.
It writes the identity map only when count, model, and USB 3 link checks all PASS; a
failed report does not establish a new identity binding:

```bash
python tools/mapping/check_usb_topology.py \
  --identity-map .local/camera_rig/camera_identity_map.json \
  --expected-count 3 \
  --report .local/reports/usb-topology.json
```

If an earlier five-device run created a wrong identity map, verify it manually, move it
to a private backup location, disconnect non-rig devices, and rerun the command. Never
replace an identity map before confirming the physical camera binding. Serials remain
only in `.local/camera_rig/camera_identity_map.json` and the private YAML generated later.

When an existing physical board has not yet been registered, first generate only the
runtime YAML needed for capture and preflight. This phase needs no target and creates no
provision YAML:

```bash
python scripts/prepare_camera_rig_calibration.py \
  --identity-map .local/camera_rig/camera_identity_map.json \
  --asset-root .local/camera_rig \
  --expected-camera-count 3 \
  --runtime-only \
  --report .local/reports/camera-rig-runtime-preparation.json
```

## 9. Calibration target

Every camera must use a resolved target artifact that exactly matches the same physical
board. The existing 500 x 700 mm deployment board and a newly generated A4 board are
mutually exclusive routes and must never be mixed.

### 9.1 Existing 500 x 700 deployment board (current rig)

The authoritative known metadata is `DICT_4X4_50`, 5 x 7 squares, 100 mm squares, and
75 mm markers. It is not `charuco_a4_v1`; never calibrate this physical board using a
spec from the A4 `target generate` command. `legacy_pattern`, `border_bits`, and
canonical orientation remain evidence-gated.

Use the Section 8 runtime YAML to capture an independent eight-frame artifact from each
final camera. Each output directory must not already exist, and the board must be
clearly visible:

```bash
set -euo pipefail
for camera in camera_a camera_b camera_c; do
  camera-rig capture snapshot \
    --config ".local/camera_rig/${camera}/configs/runtime.yaml" \
    --frames 8 \
    --output ".local/captures/target-id/${camera}"
done
```

Constrain only the dictionary supported by authoritative evidence. Do not guess
`legacy_pattern`, `border_bits`, or orientation, and never substitute `DICT_4X4_100`:

```bash
camera-rig target identify-existing \
  --artifact .local/captures/target-id/camera_a \
  --artifact .local/captures/target-id/camera_b \
  --artifact .local/captures/target-id/camera_c \
  --board-width-mm 500 --board-height-mm 700 \
  --square-length-mm 100 --marker-length-mm 75 \
  --maximum-artifact-frames 8 \
  --authoritative-dictionary DICT_4X4_50 \
  --output .local/reports/charuco_500x700-identification.json

camera-rig target register-existing \
  --identification .local/reports/charuco_500x700-identification.json \
  --target-name charuco_500x700 \
  --target-frame charuco_500x700 \
  --output .local/camera_rig/shared_target/charuco_500x700

camera-rig target inspect \
  --target .local/camera_rig/shared_target/charuco_500x700/target_spec.json
(cd .local/camera_rig/shared_target/charuco_500x700 && sha256sum -c checksums.sha256)
```

If identification returns ambiguity/`PAUSED_FOR_USER_VALIDATION`, resolve it from the
original PDF, generator metadata, or real board-owner evidence before registration.
Never relax the detection gates or select a merely favorable candidate.

### 9.2 Newly generated standard A4 board (only when physically switching boards)

The standard `charuco_a4_v1` is `DICT_5X5_100`, 7 x 5 squares, 30 mm squares, 22 mm
markers, and `legacy_pattern=false`. Run this only when actually printing at 100% scale
and using that new board:

```bash
mkdir -p .local/camera_rig/shared_target
camera-rig target generate \
  --config third_party/CameraRig/configs/targets/charuco_a4_v1.yaml \
  --output .local/camera_rig/shared_target/charuco_a4_v1
camera-rig target inspect \
  --target .local/camera_rig/shared_target/charuco_a4_v1/target_spec.json
(cd .local/camera_rig/shared_target/charuco_a4_v1 && sha256sum -c checksums.sha256)
```

Do not compare `target_spec.json` against a hash hard-coded on another computer. The
resolved spec includes CameraRig/OpenCV versions and generated-file hashes, so different
valid environments can produce different spec SHA values. The authoritative checks are
`camera-rig target inspect` and the artifact's own `checksums.sha256`. Transfer the
complete target artifact directory, never only the JSON.

## 10. Fixed camera calibration

This workflow assumes all three fixed cameras use the same physical board and explicitly
defines `workspace` as that selected artifact's `target_frame` (`charuco_500x700` for
Section 9.1 and `charuco_target` for Section 9.2). The CameraRig fixed-provision contract
requires identity `T_workspace_from_target`. If the workspace is not the physical target
frame, do not fabricate a transform with this workflow; establish the correct
workspace/target contract first.

### 10.1 Generate private per-camera configs

The preparation script reads serials from the confirmed Section 8 identity map without
printing them in the terminal or report. It validates the complete Section 9 target
artifact and generates `configs/runtime.yaml` plus `configs/fixed_provision.yaml` for
every camera:

```bash
PCB_TARGET_SPEC=.local/camera_rig/shared_target/charuco_500x700/target_spec.json
python scripts/prepare_camera_rig_calibration.py \
  --identity-map .local/camera_rig/camera_identity_map.json \
  --target "$PCB_TARGET_SPEC" \
  --asset-root .local/camera_rig \
  --expected-camera-count 3 \
  --workspace-equals-target \
  --report .local/reports/camera-rig-calibration-preparation.json
```

If actually using the newly printed Section 9.2 A4 board, change only
`PCB_TARGET_SPEC` to
`.local/camera_rig/shared_target/charuco_a4_v1/target_spec.json`.

If manually created YAML already differs from the prepared contract, the script stops
without overwriting it. After reviewing the difference, append `--update-existing` to
the same command; the script creates a private `*.bak-<UTC>` backup beside each changed
file before replacement. Then run the config read-only check, which changes no private
config and writes only the requested report:

```bash
PCB_TARGET_SPEC=.local/camera_rig/shared_target/charuco_500x700/target_spec.json
python scripts/prepare_camera_rig_calibration.py \
  --identity-map .local/camera_rig/camera_identity_map.json \
  --target "$PCB_TARGET_SPEC" \
  --asset-root .local/camera_rig \
  --expected-camera-count 3 \
  --workspace-equals-target \
  --check \
  --report .local/reports/camera-rig-calibration-check.json
```

### 10.2 Config and non-hardware input checks

Confirm the configured device and profiles for each camera, then validate the complete
fixed-provision inputs. Dry-run does not open a camera or write a provision artifact:

```bash
set -euo pipefail
PCB_TARGET_SPEC=.local/camera_rig/shared_target/charuco_500x700/target_spec.json
for camera in camera_a camera_b camera_c; do
  camera-rig device inspect \
    --config ".local/camera_rig/${camera}/configs/runtime.yaml" --show-profiles
  camera-rig provision fixed \
    --config ".local/camera_rig/${camera}/configs/fixed_provision.yaml" \
    --output ".local/camera_rig/${camera}/provision" \
    --dry-run
done
```

### 10.3 Physical-board preflight

Keep the cameras, workspace, and board fixed, with the same board clearly visible in
each color image. Capture the strict 60-frame pose-validated preflight for every camera;
stop on any failure and do not relax thresholds:

```bash
set -euo pipefail
PCB_TARGET_SPEC=.local/camera_rig/shared_target/charuco_500x700/target_spec.json
for camera in camera_a camera_b camera_c; do
  camera-rig target preflight \
    --camera-config ".local/camera_rig/${camera}/configs/runtime.yaml" \
    --target "$PCB_TARGET_SPEC" \
    --frames 60 --policy pose_validated \
    --report ".local/reports/${camera}-preflight.json" \
    --overlays ".local/overlays/${camera}"
done
```

### 10.4 Provision and validate

After every preflight passes, provision each camera without moving any camera,
workspace, or board:

```bash
set -euo pipefail
for camera in camera_a camera_b camera_c; do
  camera-rig provision fixed \
    --config ".local/camera_rig/${camera}/configs/fixed_provision.yaml" \
    --output ".local/camera_rig/${camera}/provision"
  camera-rig provision validate \
    --artifact ".local/camera_rig/${camera}/provision"
done
```

Generated provision YAML uses a portable path to the selected target artifact, its
actual SHA-256, and `target.detection_policy: pose_validated`. Never reuse old
extrinsics after a physical coverage, residual, or pose-stability failure, and never
use `--force` to hide a failed gate. `--force` is only for explicitly replacing an
existing CameraRig-owned artifact, which must still pass complete validation.

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
backend and its checked asset paths. Keep these standalone smoke configs at
`pointcloud.use_rgb: false`; they do not have an authoritative IR-to-color transform.
Then run the same fresh CameraRig NPZ frame:
CameraRig snapshot keys `ir_left`, `ir_right`, and `color` are normalized by the
shared offline loader to PCB's canonical `left_ir`, `right_ir`, and `rgb` keys.

```bash
for backend in pytorch tensorrt_single tensorrt_two_stage tensorrt_plugin; do
  python scripts/run_ffs_stereo_frame.py \
    --config ".local/configs/ffs_${backend}.yaml" \
    --input .local/captures/camera_a/frames/frame_000000.npz \
    --output-dir ".local/evidence/ffs-${backend}" --no-show
done
```

For live RGB reconstruction, create a separate
`.local/configs/ffs_tensorrt_plugin_rgb.yaml` from the checked plugin route and set
`pointcloud.use_rgb: true` plus `output_format: xyzrgb`. The live CameraRig integration
supplies the authoritative IR-to-color transform; do not invent one in a standalone
smoke config.

## 12. Single-camera XYZ/XYZRGB

```bash
python tools/mapping/run_live_single_camera.py \
  --camera-config .local/camera_rig/camera_a/configs/runtime.yaml \
  --provision .local/camera_rig/camera_a/provision \
  --mapping-config .local/configs/mapping.yaml \
  --ffs-config .local/configs/ffs_tensorrt_plugin_rgb.yaml \
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
  --acceptance-scope capture_matching \
  --output .local/evidence/live-rig \
  --report .local/reports/live-rig.json
```

Camera sessions are worker-owned; open/capture/close stay on the same worker thread.
`capture_matching` is the CR7 scope: it enforces concurrent capture, complete delivered
matched sets, host-time skew, no frame reuse, and clean lifecycle. It still records
geometry and FFS performance, but CR18 owns their formal benchmark interpretation.

## 14. Frame matching

Live rigs match only `host_receive_timestamp_ns` with the configured maximum skew.
Device timestamps and frame numbers are diagnostics. Buffers are bounded and latest
biased; a frame is never reused across matched sets.

## 15. Raw concatenation

Start from `configs/mapping/raw_rgb_concatenation_example.yaml`. It has fusion OFF and
sampling OFF, exposing dense `/clouds/concatenated` for calibration overlap/debugging.

```bash
python tools/mapping/run_live_reconstruction_profile.py \
  --profile raw --rig-config .local/configs/live_rig_ffs_rgb_raw.yaml \
  --mapping-config .local/configs/mapping_acceptance.yaml --matched-sets 60 \
  --output .local/evidence/raw-rgb --report .local/reports/raw-rgb.json \
  --viewer rerun --rerun-spawn --rerun-record .local/evidence/raw-rgb.rrd
```

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

```bash
python tools/mapping/run_live_reconstruction_profile.py \
  --profile dense --rig-config .local/configs/live_rig_ffs_rgb.yaml \
  --mapping-config .local/configs/mapping_acceptance.yaml --matched-sets 60 \
  --output .local/evidence/dense-rgb --report .local/reports/dense-rgb.json
```

## 17. Crop

The production order is optional camera-frame local crop, workspace transform, one
workspace crop per camera, canonical concatenate, voxel fusion, then optional global
sampling. Crop uses XYZ and preserves RGB columns.

## 18. Optional sampling

`configs/mapping/compact_rgb_reconstruction_example.yaml` applies one global
30,000-point FPS after fusion. It preserves each selected point's RGB. `voxel_fps` and
`voxel_random` remain explicit advanced compatibility modes, but are not defaults for
an already voxel-fused cloud.

```bash
python tools/mapping/run_live_reconstruction_profile.py \
  --profile compact --rig-config .local/configs/live_rig_ffs_rgb_compact.yaml \
  --mapping-config .local/configs/mapping_acceptance.yaml --matched-sets 1 \
  --output .local/evidence/compact-rgb --report .local/reports/compact-rgb.json
```

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
  --input-mode live --rig-config .local/configs/live_rig_ffs_rgb_compact.yaml \
  --frames 60 --warmup 5 \
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

## 30. Existing 500x700 deployment board

The existing physical deployment board is 500 x 700 mm with a 5 x 7 square layout,
100 mm squares, 75 mm markers, and authoritative dictionary `DICT_4X4_50`. This is an
existing-board workflow, not a generated-board workflow; the public known metadata is
in `configs/calibration/charuco_500x700_existing_board.yaml`. Dictionary identity is
resolved, while `legacy_pattern`, `border_bits`, and canonical `squares_x` versus
`squares_y` orientation remain evidence-gated. Physical rotation in a camera image
does not change target geometry. CameraRig visual detection, marker-layout, corner-ID,
and geometry-consistency gates must still pass and must not be weakened.

## 31. Projection models

PCB now preserves CameraRig frame, distortion model, coefficients, and explicit
raw/rectified pixel geometry. Shared projection/deprojection APIs drive the builder,
FFS RGB mapping, calibration, and diagnostics; unsupported model directions fail
closed. Quantitative librealsense parity, the FFS no-double-rectification contract,
and the distinction between model parity and physical intrinsic accuracy are in
[`docs/projection-models.md`](docs/projection-models.md).

## 32. Multi-pose N-camera calibration

CameraRig remains one-physical-camera only. PCB owns gauge-fixed, robust multi-pose
N-camera bundle adjustment, connected partial-visibility graphs, pose-diversity gates,
solve/holdout validation, and explicit candidate-only export under
`pointcloud_builder.rig_calibration`. The operator workflow, mathematics, artifact
contracts, synthetic acceptance, and deferred real third-camera status are documented
in [`docs/multi-pose-multi-camera-calibration.md`](docs/multi-pose-multi-camera-calibration.md).

## 33. Calibration lifecycle and production precedence

```text
CameraRig fixed provision (initializer only)
-> PCB multi-pose candidate
-> exact 2D solve plus holdout validation
-> candidate-only live preview
-> physical N-camera pairwise acceptance
-> promoted PCB rig-calibration deployment
-> snapshot / recording / TSDF / Rerun production outputs
```

`rig_calibration` is optional in a rig YAML. When omitted, the legacy CameraRig fixed
provision supplies workspace geometry. When configured, the PCB deployment is
authoritative for `T_workspace_from_camera` and any identity, CameraBundle hash,
camera-set, frame, or fingerprint mismatch fails without fallback. CameraRig remains
authoritative for K/D, depth scale, stream frames, device identity, and internal stream
extrinsics. For FFS, runtime composes
`T_workspace_from_ir_left = T_workspace_from_color(deployed) @ T_color_from_ir_left`.
The candidate viewer rejects a production-configured rig and always reports
`candidate_only=true`, `production_applied=false`.

Promotion requires exact solution, validation, passed holdout, and real physical 3D
acceptance receipts; reprojection alone is insufficient:

```bash
python tools/calibration/promote_rig_calibration.py \
  --solution .local/calibration/new-workspace/solution.json \
  --validation .local/calibration/new-workspace/validation.json \
  --physical-acceptance .local/calibration/new-workspace/physical_acceptance.json \
  --output .local/calibration/deployment/new-workspace/rig_calibration.json
```

Recordings and TSDF map artifacts store calibration mode, deployment/solution
fingerprints, camera set, workspace frame, and per-camera CameraBundle hashes. An
`--initial-map` with a different deployment fingerprint fails closed; there is no
implicit map migration.

## 34. Existing-board identification and registration

Section 9.1 is the canonical, executable procedure for the current 500 x 700 mm
`DICT_4X4_50` board, including three-camera evidence capture, identification,
registration, and artifact-owned checksum validation. Do not duplicate or relocate
its private paths here. If CameraRig still reports ambiguity in `legacy_pattern`,
`border_bits`, or orientation, stop and resolve it from visual equivalence or a real
authoritative source file. Never pass a descriptive label to `--authoritative-source`,
scan dictionary capacities, or substitute `DICT_4X4_100`.

## 35. Generic N-camera acceptance

The production acceptance layer enumerates all `N choose 2` unordered pairs without
camera-name assumptions. It records overlap, symmetric NN, board/interior and plane
metrics, diagnostic-only residual SE(3), per-camera contribution, overlap graph,
fused count, surface thickness where measurable, and matcher/drop statistics.
Diagnostic ICP is never written back into deployment extrinsics.

```bash
python tools/calibration/evaluate_ncamera_rig_alignment.py \
  --rig-config .local/configs/live_rig_three_camera_candidate.yaml \
  --candidate-solution .local/calibration/new-workspace/solution.json \
  --candidate-validation .local/calibration/new-workspace/validation.json \
  --thresholds configs/calibration/ncamera_physical_acceptance_strict_example.yaml \
  --recording .local/recordings/new-workspace-physical-acceptance \
  --mapping-config .local/configs/new-workspace-mapping.yaml \
  --matched-sets 5 \
  --output .local/reports/new-workspace-ncamera-acceptance.json
```

Candidate mode emits the formal physical-acceptance artifact consumed by promotion.
Preregister/freeze the threshold file before inspecting new data. After promotion,
replace the three candidate arguments with `--rig-calibration` to regression-test the
deployed path; it reuses the exact accepted physical receipt thresholds.

Prefer all A-B, A-C, and B-C pairs passing. A physically non-overlapping pair may be
declared with `--declared-no-overlap camera_a:camera_c` only with physical
justification and only when the remaining accepted-overlap graph stays connected. It
is reported as `NOT_APPLICABLE_NO_OVERLAP`, not as invented NN metrics.

## 36. Three-camera configuration and USB/FFS readiness

Start from `configs/mapping/live_rig_three_camera_example.yaml`. Public files contain
no serials. Each camera needs a private CameraRig runtime YAML and provision artifact;
identical validated FFS profiles may share one FFS pipeline YAML. All three links must
enumerate as USB 3.x. Inspect root hubs and distribute bandwidth across controllers
where practical, without assuming any particular motherboard topology.
For candidate preview/acceptance, copy the example but omit `rig_calibration`; add that
section only after promotion and point it to the exact deployed artifact.

```bash
python scripts/doctor_reconstruction_env.py --no-hardware
python scripts/doctor_reconstruction_env.py \
  --expected-d435i-count 3 --asset-root .local/ffs
python tools/mapping/check_usb_topology.py \
  --identity-map .local/camera_rig/camera_identity_map.json \
  --expected-count 3 --report .local/reports/usb-topology.json
```

N-camera functionality is generic, but three-camera throughput is not inferred from
dual-camera FPS. Benchmark capture FPS, match ratio, p50/p95 processing, GPU memory,
RSS, viewer overhead, and TSDF mapper overhead on the final GPU and USB topology.

## 37. New machine / new workspace checklist

Follow this order; keep cameras fixed after step 12 and stop on every failed gate.

1. Clone with `--recurse-submodules` and update submodules.
2. Create the isolated `pcb-reconstruction` environment.
3. Run the doctor with `--no-hardware`.
4. Prepare or rebuild the FFS TensorRT-plugin assets.
5. Connect three D435i cameras.
6. Discover identities without copying serials into public files.
7. Assign private logical names `camera_a`, `camera_b`, and `camera_c`.
8. Validate USB topology with `--expected-count 3`.
9. Create one private CameraRig runtime YAML per camera.
10. Capture existing-board identity evidence from the 500 x 700 `DICT_4X4_50` board.
11. Identify and register the existing board with the commands in section 34.
12. Deliberately fixture the board at canonical workspace `pose_0`.
13. Run CameraRig `pose_validated` target preflight for every camera.
14. Create initial fixed provisions for every camera while `pose_0` is sufficiently visible.
15. Validate every provision.
16. Create the private three-camera rig config from the public example.
17. Run projection-parity smoke if a profile/model changed.
18. Capture about 24-30 diverse multi-pose target poses.
19. Predeclare about 4-6 final poses as holdout.
20. Solve PCB N-camera bundle adjustment.
21. Validate the exact candidate on the reserved holdout.
22. Run candidate-only live preview.
23. Record and pass generic N-camera pairwise physical acceptance.
24. Promote the exact candidate and acceptance receipts to production.
25. Run raw RGB reconstruction.
26. Run dense XYZRGB fusion with sampling off and 2.5 mm voxels.
27. Measure and configure a new workspace crop.
28. Benchmark real three-camera performance and viewer/mapper overhead.
29. Record new fingerprint-bound depth data.
30. Build a new fingerprint-bound TSDF map.
31. Validate, save, and reload the map.
32. Run interactive Rerun in production mode.

The pose-0 board defines `T_workspace_from_target,pose0 = I`, hence workspace origin
and +X/+Y/+Z. Use mechanical stops, a fixture, tape, or measured references rather
than visual placement. After initial provisioning, move the board through pose 1..M,
never the cameras. Later poses may have partial visibility such as A+B, B+C, A+C, and
A+B+C, but the complete camera-pose graph must remain connected.

## 38. Invalidation rules and deployment status

Recalibrate when a camera moves, a mount loosens, camera set changes, workspace pose-0
changes, physical target geometry changes, CameraBundle changes, or intrinsics/profile
changes. A new physical workspace requires new provisions, observations, solution,
validation, physical acceptance, production deployment, workspace crop, recordings,
and TSDF map. Old artifacts cannot silently become production artifacts in the new
workspace.

Current status: N-camera implementation is `VALIDATED_SYNTHETICALLY` for 2/3/4
cameras; real dual-camera multi-pose production is `VALIDATED`; real three-camera
calibration/reconstruction is `DEFERRED` until camera C and the new workspace exist.
Large-board metadata is `RESOLVED`; real registration is
`DEFERRED_TO_NEW_WORKSPACE`.
