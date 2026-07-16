# Fast-FoundationStereo in PointCloudBuilder

[中文说明](README_zh-CN.md) · [PointCloudBuilder README](../README.md)

This directory is the self-contained Fast-FoundationStereo (FFS) integration
for PointCloudBuilder. FFS replaces only the depth source:

```text
left/right rectified IR -> disparity -> metric depth
                              |
                              v
existing deprojection -> crop -> fixed-size sampling
```

Runtime does not import or read a sibling Fast-FoundationStereo checkout. The
copied source is pinned to commit
`a290ba04c1b3ad1ec41a33974a157b2917b624d4`; provenance is in
[`UPSTREAM_SOURCE.json`](UPSTREAM_SOURCE.json). It remains under the complete
NVIDIA license in [`LICENSE.txt`](LICENSE.txt) and is restricted to
non-commercial research use.

## Backends and local artifacts

| Backend | Required local files | Notes |
| --- | --- | --- |
| `pytorch` | checkpoint + `cfg.yaml` | Reference route; no ONNX/Engine |
| `tensorrt_single` | one ONNX, Engine, same-contract YAML, manifest | External ImageNet normalization |
| `tensorrt_two_stage` | feature/post ONNX + Engines, YAML, manifest | Triton GWC between Engines |
| `tensorrt_plugin` | plugin ONNX + Engine, YAML, manifest, `.so` | `FFSGWCVolume` CUDA plugin |

The following directories are intentionally gitignored:

| Path | Restore method |
| --- | --- |
| `artifacts/model_best_bp2_serialize.pth`, `cfg.yaml` | Download the official `20-30-48` weights |
| `artifacts/*.onnx`, manifests and route YAML | Regenerate with `prepare_ffs_artifacts.py` |
| `artifacts/*.engine` | Rebuild in the target TensorRT/GPU environment |
| `build/libffs_gwc_plugin.so` | Recompile with `build_ffs_plugin.py` |
| `tensorrt_sdk/` | Sparse-clone TensorRT headers |
| `outputs/` | Regenerate by running smoke/visualization commands |

There is no hosted PointCloudBuilder bundle of prebuilt Engines. Do not treat
an Engine copied from another TensorRT/CUDA/GPU stack as a deployment artifact;
rebuild it in the target `dp3` environment.

## Verified target

```text
Python 3.10.20
PyTorch 2.11.0+cu130
TensorRT 10.16.1.11
GPU NVIDIA GeForce RTX 5080 (SM120)
input  [1,3,480,640] x 2
output [1,1,480,640]
max_disp=192, valid_iters=8
```

Always use `PYTHONNOUSERSITE=1`; otherwise user-site packages can override the
validated packages in `dp3`.

## Fresh-clone setup

Run all commands from the PointCloudBuilder repository root. Do not create a
second Python environment.

### 1. Install the package and optional FFS dependencies

```bash
cd ~/workspace/3D-Diffusion-Policy/PointCloudBuilder
export PY=~/miniconda3/envs/dp3/bin/python

PYTHONNOUSERSITE=1 "$PY" -m pip install -e '.[dev,viz]'
PYTHONNOUSERSITE=1 "$PY" -m pip install \
  timm==1.0.28 onnx==1.18.0 onnxscript==0.5.6 \
  imageio opencv-python-headless pyarrow av
```

TensorRT routes additionally require the target TensorRT Python packages:

```bash
PYTHONNOUSERSITE=1 "$PY" -m pip install \
  --extra-index-url https://pypi.nvidia.com/ --no-deps \
  tensorrt_cu13_bindings==10.16.1.11 \
  tensorrt_cu13_libs==10.16.1.11 \
  tensorrt-cu13==10.16.1.11

PYTHONNOUSERSITE=1 "$PY" -c \
  'import torch,tensorrt as trt; print(torch.__version__, torch.version.cuda, trt.__version__, torch.cuda.get_device_name(0))'
```

The plugin build also needs CMake and the CUDA compiler available to `dp3`.

### 2. Download and verify the official checkpoint

The [official FFS repository](https://github.com/NVlabs/Fast-FoundationStereo)
publishes weights through this
[Google Drive folder](https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap?usp=drive_link).
Download the `20-30-48` directory in a browser, or use `gdown`:

```bash
PYTHONNOUSERSITE=1 "$PY" -m pip install gdown
~/miniconda3/envs/dp3/bin/gdown \
  'https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap?usp=drive_link' \
  --folder -O ~/Downloads/fast-foundationstereo-weights

WEIGHT_FILE="$(find ~/Downloads/fast-foundationstereo-weights \
  -path '*/20-30-48/model_best_bp2_serialize.pth' -print -quit)"
test -n "$WEIGHT_FILE"
WEIGHT_DIR="$(dirname "$WEIGHT_FILE")"
mkdir -p ffs_reproduction/artifacts
install -m 0644 "$WEIGHT_DIR/model_best_bp2_serialize.pth" \
  ffs_reproduction/artifacts/model_best_bp2_serialize.pth
install -m 0644 "$WEIGHT_DIR/cfg.yaml" \
  ffs_reproduction/artifacts/cfg.yaml

printf '%s  %s\n' \
  98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692 \
  ffs_reproduction/artifacts/model_best_bp2_serialize.pth \
  d45afe99b176454d5aff416edf16c8da6a99579f8f374b927f37907442a7d6bc \
  ffs_reproduction/artifacts/cfg.yaml | sha256sum -c -
```

The official checkpoint is a trusted pickle. The PyTorch backend deliberately
uses `torch.load(..., weights_only=False)` inside a scoped compatibility loader;
never substitute an untrusted checkpoint.

### 3A. PyTorch-only route

No build is required after the checkpoint and config are present. Continue to
the smoke commands below with `--backend pytorch`.

### 3B. ONNX-only export

On a fresh clone, export fixed-shape ONNX and manifests without building any
Engine:

```bash
PYTHONNOUSERSITE=1 "$PY" scripts/prepare_ffs_artifacts.py \
  --precision fp16 --builder-optimization-level 3 \
  --artifact-suffix fp16_o3 --skip-tensorrt
```

The exporter is explicitly fixed to opset 17, static `480x640`, and
`dynamo=False`. Ordinary ONNX files pass `onnx.checker.check_model`, contract
checks, and the target TensorRT parser before an Engine is accepted.

### 3C. Build all TensorRT routes

Obtain the matching TensorRT C++ headers without storing the entire TensorRT
repository:

```bash
git clone --depth 1 --branch v10.16 --filter=blob:none --sparse \
  https://github.com/NVIDIA/TensorRT.git \
  ffs_reproduction/tensorrt_sdk
git -C ffs_reproduction/tensorrt_sdk sparse-checkout set include
```

Build the SM120 plugin:

```bash
PYTHONNOUSERSITE=1 "$PY" scripts/build_ffs_plugin.py \
  --tensorrt-root ffs_reproduction/tensorrt_sdk \
  --cuda-arch 120
readelf -d ffs_reproduction/build/libffs_gwc_plugin.so | grep RUNPATH
```

The expected RUNPATH is `$ORIGIN`. Then export and build the FP16 artifacts
through the TensorRT Python API:

```bash
PYTHONNOUSERSITE=1 "$PY" scripts/prepare_ffs_artifacts.py \
  --precision fp16 --builder-optimization-level 3 \
  --workspace-gib 8 --artifact-suffix fp16_o3
```

If ONNX-only export was already run, append `--force` because the script
intentionally refuses to overwrite existing derived artifacts. `--force`
replaces only the selected artifact variant; it does not change the trusted
checkpoint.

An independent FP32/o0 diagnostic build is explicit and never relabeled as
FP16:

```bash
PYTHONNOUSERSITE=1 "$PY" scripts/prepare_ffs_artifacts.py \
  --precision fp32 --builder-optimization-level 0 \
  --workspace-gib 8 --artifact-suffix fp32_o0 --force
```

There is no precision or backend fallback. A failed build writes a complete
`*.build_error.json` and leaves the requested route failed.

## Validate after moving or deleting the upstream checkout

First run the test suite and verify that no tracked source contains a machine
absolute path:

```bash
PYTHONNOUSERSITE=1 "$PY" -m pytest -q
git grep -nF "$HOME" || true
```

Run the required v05 frame through every available FP16 route without GUI:

```bash
DATASET=~/.cache/huggingface/lerobot/flexiv_dual_arm_3d/pick_place_20260713_v05
CONFIG=ffs_reproduction/configs/v05_ffs.yaml
OUT=ffs_reproduction/outputs/v05_verify

for BACKEND in pytorch tensorrt_single tensorrt_two_stage tensorrt_plugin
do
  PYTHONNOUSERSITE=1 "$PY" scripts/run_v05_ffs_frame.py \
    --dataset-root "$DATASET" \
    --camera head --global-frame-index 0 \
    --backend "$BACKEND" --builder-config "$CONFIG" \
    --artifact-id fp16_o3 --precision fp16 \
    --builder-optimization-level 3 --workspace-gib 8 \
    --output-dir "$OUT" --no-show
done
```

Each route must produce `left_ir.png`, `right_ir.png`, `disparity.png`,
`depth.png`, validity masks, `raw.ply`, `cropped.ply`, `sampled.ply`,
`metadata.json`, `parity.json`, and `stage_counts.json` under
`$OUT/<backend>/`. `parity.json` must report
`native_depth_used_for_builder: false`.

For a stronger filesystem check, trace one PyTorch run and confirm that the
renamed checkout is never opened:

```bash
strace -f -e trace=file -o /tmp/pointcloudbuilder_ffs_files.strace \
  env PYTHONNOUSERSITE=1 "$PY" scripts/run_v05_ffs_frame.py \
  --dataset-root "$DATASET" --camera head --global-frame-index 0 \
  --backend pytorch --builder-config "$CONFIG" \
  --artifact-id fp16_o3 --precision fp16 \
  --builder-optimization-level 3 --workspace-gib 8 \
  --output-dir "$OUT" --no-show
rg 'Fast-FoundationStereo' /tmp/pointcloudbuilder_ffs_files.strace
```

The final `rg` command should return no matches.

## Visualize each backend

To display the IR/disparity/depth panel and interactive point clouds directly,
run one backend without `--no-show`:

```bash
BACKEND=tensorrt_single  # pytorch | tensorrt_single | tensorrt_two_stage | tensorrt_plugin
PYTHONNOUSERSITE=1 "$PY" scripts/run_v05_ffs_frame.py \
  --dataset-root "$DATASET" --camera head --global-frame-index 0 \
  --backend "$BACKEND" --builder-config "$CONFIG" \
  --artifact-id fp16_o3 --precision fp16 \
  --builder-optimization-level 3 --workspace-gib 8 \
  --output-dir "$OUT"
```

For three simultaneous Open3D windows showing the already generated raw,
cropped, and sampled clouds:

```bash
PYTHONNOUSERSITE=1 "$PY" scripts/view_pointcloud_triplet.py \
  --input-dir "$OUT/$BACKEND"
```

The visualization contract does not apply Open3D radius denoising or zfar
filtering (`denoise_cloud=false`, `zfar=100`, `zfar_applied=false`). Counts for
invalid disparity, `remove_invisible`, z-range, Builder crop, and sampling are
reported separately in `stage_counts.json`.

Compare all TensorRT routes numerically against PyTorch:

```bash
PYTHONNOUSERSITE=1 "$PY" scripts/compare_ffs_backends.py \
  --dataset-root "$DATASET" --camera head --global-frame-index 0 \
  --builder-config "$CONFIG" --artifact-id fp16_o3 \
  --precision fp16 --builder-optimization-level 3 --workspace-gib 8 \
  --output ffs_reproduction/outputs/v05_verify/parity_all.json
```

## Runtime contract

- Input is a rectified, undistorted `480x640` IR1/IR2 pair in raw `0..255`.
- Output disparity is full-resolution float32 in left-image pixels.
- Metric depth is `fx * baseline / disparity`; invalid depth is zero.
- The current calibration gate accepts only the recorded identity/no-op
  rectification contract and rejects non-identity calibration.
- FFS uses IR1 intrinsics. RGB projection is optional and uses IR1-to-color
  extrinsics.
- `depth_source.mode=frame` remains the default and imports no FFS, ONNX,
  TensorRT, Triton, or Open3D modules.
- Engine, config, manifest, shape, precision, normalization, and hashes are
  validated fail-fast; artifacts are never selected ambiguously.

See [`configs/v05_ffs.yaml`](configs/v05_ffs.yaml) for the complete YAML and
[`../FFS_REPRODUCTION_AND_INTEGRATION_REPORT.md`](../FFS_REPRODUCTION_AND_INTEGRATION_REPORT.md)
for the measured build/parity/latency matrix.
