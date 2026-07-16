# Fast-FoundationStereo Reproduction and PointCloudBuilder Integration Report

Date: 2026-07-16

## Scope and environment

This implementation is scoped to the nested git root
`~/workspace/3D-Diffusion-Policy/PointCloudBuilder`. The parent
repository and `~/workspace/Fast-FoundationStereo` were not
modified. It was used as a read-only source during the initial asset import
and for upstream reference evidence. The current export/build/runtime path
uses only the copied vendor tree, local checkpoint/config, and dp3-built
artifacts, so that external checkout may be moved or deleted. No Engine
produced in the `ffs` environment is used by the final runtime.

All final export, TensorRT parsing/building, plugin compilation, inference,
parity, and benchmark commands use:

```text
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python
Python 3.10.20
PyTorch 2.11.0+cu130
CUDA reported by PyTorch: 13.0
TensorRT Python: 10.16.1.11 (package: tensorrt-cu13)
GPU: NVIDIA GeForce RTX 5080, compute capability (12, 0)
```

The copied FFS source is pinned to upstream commit
`a290ba04c1b3ad1ec41a33974a157b2917b624d4`. The checkpoint and config are
copied into the local artifact directory after hash verification:

```text
checkpoint: model_best_bp2_serialize.pth
  size: 62078956 bytes
  sha256: 98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692
model config: cfg.yaml
  sha256: d45afe99b176454d5aff416edf16c8da6a99579f8f374b927f37907442a7d6bc
```

The copied vendor files remain under `ffs_reproduction/LICENSE.txt` and the
upstream non-commercial research-use restriction is retained.

## Implemented contract

`depth_source.mode=frame` remains the default and the existing
PointCloudBuilder public methods and RGB-D path are preserved. The new
`depth_source.mode=ffs_stereo` path accepts raw IR1/IR2 only, uses one shared
FFS depth-source interface, and then reuses the original deprojection, RGB
mapping, crop, and fixed-size sampling stages.

The four explicit backends are:

1. PyTorch reference;
2. ordinary single ONNX/TensorRT;
3. two-stage TensorRT feature + Triton GWC + TensorRT post;
4. TensorRT with the `FFSGWCVolume` CUDA plugin.

There is no implicit backend or precision fallback. `precision` is explicitly
`fp16` or `fp32`; `builder_optimization_level` is `0..5`; and `workspace_gib`
is recorded in the route config and manifest. An FP16 build failure is saved
with its phase and complete traceback. A separately requested FP32 diagnostic
build has a separate artifact id and filenames.

Engine/config resolution is fail-fast and ordered as follows: explicit config,
manifest-declared config, same-stem YAML, then a directory containing exactly
one YAML. Missing or ambiguous candidates fail. Runtime validation covers
backend, shape, max disparity, valid iterations, normalization, precision,
builder resources, I/O names, and SHA-256 entries.

The ordinary ONNX exporter is explicitly pinned to
`opset_version=17`, static `480x640` inputs, and `dynamo=False`. Ordinary graphs
run `onnx.checker.check_model`; the target TensorRT parser is also exercised.
The plugin graph contains the private `FFSGWCVolume` node, so standard ONNX
checking is reported as `skipped_custom_op` while plugin registration and the
target TensorRT parser are required.

The TensorRT Python API is the formal builder path. `trtexec` is not required
or invoked. Successful build records include TensorRT/CUDA/GPU, precision,
workspace, optimization level, build time, Engine size, network I/O names,
shapes and dtypes, activation memory, persistent-memory field when exposed,
parser status, and deserialization status. Failed builds record the same
environment plus the complete error and traceback.

## Upstream reference evidence

These results are retained as an upstream reference baseline only; they are
not the final dp3 artifacts:

```text
environment: ~/miniconda3/envs/ffs
Python 3.12, PyTorch 2.9.1+cu128, TensorRT 10.11.0.33
input: [1,3,480,640] x 2 -> [1,1,480,640]
valid_iters=8, max_disp=192, ONNX opset=17
ONNX: about 77.4 MB, 6309 nodes; TensorRT parser about 15153 layers
```

On that upstream reference, TensorRT FP16 with the default optimization level
failed during tactic/compiler work after about seven minutes with
`costTensor.cpp::indexOfMin Assertion !empty() failed`. The parser, v05 input,
calibration, and SM120 availability were not the failure. TensorRT FP32 with
optimization level 0 built in about 16.26 seconds, produced an 82,636,708-byte
Engine with about 2,626,636,800 bytes of activation memory, and ran v05 head,
episode 0, frame 0 through disparity, metric depth, PLY, and Open3D single
frame output.

The correct status for that reference route is **functional in FP32; FP16
optimization unresolved**. This does not imply that FP16 fails for every
TensorRT version.

## Target dp3 artifact results

Target artifacts were rebuilt in dp3 TensorRT 10.16.1.11. The local CUDA
plugin is `ffs_reproduction/build/libffs_gwc_plugin.so`, size 161,568 bytes,
SHA-256
`04f01d39721bc854f86284f5a4949e1abb2397a7057bbd89e6e5871d8a66e60d`; its
CUDA image was checked for `sm_120`, its TensorRT creator was registered, and
its ELF RUNPATH is the relocatable `$ORIGIN` rather than a repository or conda
absolute path.

The ordinary single ONNX is 77,356,941 bytes, 6,309 nodes, SHA-256
`15ad835b2cf3127b9581ee98d09a4c162e84e954d7e928f7fb55d393c517e9e9`, with
inputs `left_image`, `right_image` and output `disparity`. It passed the ONNX
checker, static-shape/name/opset checks, and the target parser.

### Build and runtime matrix

`parity` means numerical comparison against the PyTorch reference using the
configured thresholds. `v05 point cloud` means the required head/episode 0/
frame 0 smoke produced disparity, metric depth, raw/cropped/sampled PLY, IR
images, and metadata. No performance target was specified by the project, so
the latency column is measurement only rather than a claimed target pass.

| backend | TRT version | precision | opt | build | parser | deserialize | inference | v05 point cloud | parity | latency | notes |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch reference | n/a | fp16 | n/a | n/a | n/a | n/a | pass | pass | reference | host p50 34.56 ms; inference p50 31.29 ms | 629.48 MiB peak in 20/100 benchmark |
| single TRT | 10.16.1.11 | fp32 | 0 | pass | pass | pass | pass | pass | pass | host p50 30.97 ms; inference p50 26.65 ms | independent `fp32_o0` diagnostic, 60,401,564-byte Engine |
| single TRT | 10.16.1.11 | fp16 | 3 | pass | pass | pass | pass | pass | pass | host p50 13.94 ms; inference p50 10.42 ms | `fp16_o3`, 47,097,756-byte Engine |
| two-stage TRT | 10.16.1.11 | fp16 | 3 | pass: feature + post | pass | pass | pass | pass | pass | host p50 14.02 ms; inference p50 10.33 ms | feature 23,202,084 bytes + post 24,102,220 bytes; Triton GWC between stages |
| plugin TRT | 10.16.1.11 | fp16 | 3 | pass | pass with plugin | pass | pass | pass | pass | host p50 13.47 ms; inference p50 9.92 ms | `FFSGWCVolume`, 46,807,444-byte Engine; standard ONNX checker intentionally skipped for private op |
| single TRT upstream reference | 10.11.0.33 | fp32 | 0 | pass | pass | pass | pass | pass | not run | not measured | read-only `~/miniconda3/envs/ffs` baseline |
| single TRT upstream reference | 10.11.0.33 | fp16 | default | fail | parser had succeeded | n/a | n/a | n/a | n/a | n/a | `costTensor.cpp::indexOfMin`; reference status above |

Target FP32 artifact details are recorded in
`ffs_reproduction/artifacts/artifact_manifest_fp32_o0.json`. The target FP16
deployment artifact details are in
`ffs_reproduction/artifacts/artifact_manifest_fp16_o3.json`. The target FP32
single Engine reports 465,467,904 bytes activation memory. FP16 reports
230,709,248 bytes for single, 40,706,048/192,347,648 bytes for two-stage
feature/post, and 279,343,104 bytes for the plugin. Persistent device memory
was not exposed by this TensorRT inspection API and is recorded as null.

## v05 smoke, visualization, and point-count accounting

The authoritative dataset was
`~/.cache/huggingface/lerobot/flexiv_dual_arm_3d/pick_place_20260713_v05`.
The smoke frame is `camera=head`, `episode=0`, `frame=0`, `480x640`,
`max_disp=192`, `valid_iters=8`. The derived input used by the benchmark was
`/tmp/v05_head_frame0_stereo.npz`; the dataset and parent repository were not
modified.

Outputs are separated by artifact variant:

```text
ffs_reproduction/outputs/v05/fp16_o3/{pytorch,tensorrt_single,tensorrt_two_stage,tensorrt_plugin}/
ffs_reproduction/outputs/v05/fp32_o0/{pytorch,tensorrt_single}/
```

Each smoke output contains left/right IR PNGs, disparity and metric-depth
arrays/PNGs, validity masks, raw/cropped/sampled PLYs, metadata, timing, and
stage counts. The visualization contract is `denoise_cloud=false` and
`zfar=100`; zfar is not applied. Therefore far sparse points are not removed by
Open3D radius denoising or zfar before the following accounting.

Representative FP16 counts (small backend-to-backend differences are expected
from precision and tactic selection) are:

| backend | invalid disparity | remove_invisible | z-range removed | raw/deprojection | crop output | sampled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch | 0 | 13,011 | 0 | 294,189 | 294,189 | 1,024 |
| single TRT | 0 | 13,005 | 0 | 294,195 | 294,195 | 1,024 |
| two-stage TRT | 0 | 13,000 | 0 | 294,200 | 294,200 | 1,024 |
| plugin TRT | 0 | 13,015 | 0 | 294,185 | 294,185 | 1,024 |

All four have zero Builder crop removal for this frame. The remaining removal
from the cropped set is sampling (for example 294,189 to 1,024 for PyTorch),
not FFS invalidity and not Builder crop. The generated masks separately show
invalid disparity, remove-invisible, and z-range filtering. Native depth is
used only for a diagnostic comparison and never as the Builder input.

The FP16 v05 diagnostic comparison against native depth is not absolute ground
truth. For PyTorch, native valid ratio was 0.897731, FFS valid ratio 0.957646,
native-overlap ratio 1.0, mean absolute depth difference 0.011601 m, and mean
relative difference 0.011058. The calibration SHA-256 was
`d0622962bd5c83fa0d767395f6f0ee8ae17710c04185c8b3029bb0f5c6ca07ba`.

## Numerical parity

The full v05 parity files are:

```text
ffs_reproduction/outputs/v05/parity_fp16_o3.json
ffs_reproduction/outputs/v05/parity_fp32_o0.json
```

All reported checks passed: finite-positive ratio 1.0 and valid-overlap ratio
1.0. Relative to the PyTorch reference, the FP16 metrics were:

| backend | disparity MAE (px) | disparity p95 (px) | depth MAE (m) | relative depth error |
| --- | ---: | ---: | ---: | ---: |
| single TRT | 0.014144 | 0.032755 | 0.0005081 | 0.0005608 |
| two-stage TRT | 0.013533 | 0.027893 | 0.0005255 | 0.0005635 |
| plugin TRT | 0.014193 | 0.028046 | 0.0005517 | 0.0005925 |

For the independent FP32 diagnostic single route, the corresponding values
were 0.002341 px MAE, 0.003326 px p95, 0.0001021 m depth MAE, and 0.0001004
mean relative depth error; all checks also passed.

## Regression checks

The final dp3 test run collected 74 tests and completed with:

```text
74 passed, 14 warnings
```

This includes the original RGB-D builder/config/deprojection/crop/sampling
tests, CPU/fake FFS integration, four-backend contract checks, missing and
ambiguous config cases, engine/config contract and artifact SHA mismatch cases,
static ONNX shape/name/opset checks, and the target TensorRT parser test.

The warnings are deprecation warnings from the copied upstream PyTorch vendor
code; they do not change the pass result. The remaining known runtime warning
is TensorRT's default-stream `enqueueV3()` synchronization warning. The
benchmark values above include that current synchronization behavior; moving
the runtime to an explicit non-default CUDA stream is a future performance
optimization, not a correctness fallback.

## Reproduction commands

Build the SM120 plugin and all FP16 routes in dp3:

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python \
  scripts/build_ffs_plugin.py --tensorrt-root ffs_reproduction/tensorrt_sdk
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python \
  scripts/prepare_ffs_artifacts.py --precision fp16 \
  --builder-optimization-level 3 --workspace-gib 8.0 \
  --artifact-suffix fp16_o3 --force
```

Build an independent FP32 diagnostic artifact only when explicitly requested:

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python \
  scripts/prepare_ffs_artifacts.py --precision fp32 \
  --builder-optimization-level 0 --workspace-gib 8.0 \
  --artifact-suffix fp32_o0 --force
```

Run the v05 smoke, parity, and benchmark helpers with the corresponding
`--artifact-id`, `--precision`, and `--builder-optimization-level`. The
artifact-specific configs and manifests are selected explicitly; no Engine is
renamed or relabeled across precision variants.
