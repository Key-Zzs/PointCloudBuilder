# FFS reproduction package

This directory contains the self-contained Fast-FoundationStereo integration
used by `PointCloudBuilder`.

The copied vendor tree records upstream commit
`a290ba04c1b3ad1ec41a33974a157b2917b624d4`. File-level SHA-256
provenance is recorded in `UPSTREAM_SOURCE.json`. The copied source remains
under the complete NVIDIA license in `LICENSE.txt` and is restricted to
non-commercial research use.

The runtime contract is fixed to one 480x640 raw IR1/IR2 pair. Four explicit
backends implement the same `infer_disparity([1,3,480,640], [1,3,480,640]) ->
[480,640]` interface:

1. `pytorch`: trusted local checkpoint loaded through the scoped copied vendor
   modules;
2. `tensorrt_single`: ordinary ONNX graph with external ImageNet
   normalization;
3. `tensorrt_two_stage`: TensorRT feature/post graphs with Triton GWC between
   them;
4. `tensorrt_plugin`: raw-input graph using the `FFSGWCVolume` CUDA plugin.

Use the `dp3` interpreter and `PYTHONNOUSERSITE=1`. The artifact
preparation script hash-checks the repository-local checkpoint/config,
exports ONNX through the TensorRT Python builder path, validates manifests,
and refuses an artifact mismatch. `--skip-tensorrt` prepares only the
checkpoint, ONNX, and route manifests. A complete build additionally needs
TensorRT 10.16.1.11 CUDA 13 Python bindings/libs and the TensorRT C++ headers:

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python -m pip install \
  --extra-index-url https://pypi.nvidia.com/ --no-deps \
  tensorrt_cu13_bindings==10.16.1.11 tensorrt_cu13_libs==10.16.1.11
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python -m pip install \
  --extra-index-url https://pypi.nvidia.com/ --no-deps \
  tensorrt-cu13==10.16.1.11
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python \
  scripts/prepare_ffs_artifacts.py --skip-tensorrt
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python \
  scripts/build_ffs_plugin.py --tensorrt-root ffs_reproduction/tensorrt_sdk
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python \
  scripts/prepare_ffs_artifacts.py
```

The default preparation path needs no upstream checkout: it reads
`artifacts/model_best_bp2_serialize.pth` and `artifacts/cfg.yaml`. For a
one-time import into an empty artifact directory, `--source-root` may point to
a trusted checkout. After those two local files exist, moving or deleting that
checkout does not affect export, build, or runtime.

No route falls back to another backend. `depth_source.mode=frame` remains the
default. In `ffs_stereo` mode the calibration gate accepts only identity/no-op
rectified IR input, converts disparity using `fx * baseline / disparity`,
marks invisible/non-finite/out-of-range pixels invalid, and passes metric depth
to the existing builder deprojection/crop/sampling pipeline with effective
depth scale `1.0`.

ONNX export is explicitly pinned to PyTorch's legacy exporter with
`dynamo=False`, opset 17, and static 480x640 inputs. Ordinary ONNX is checked
with `onnx.checker.check_model` and the target TensorRT Python parser; the
custom-plugin graph additionally requires the `FFSGWCVolume` node and parser
registration. Dynamo is not removed unless a complete ONNX/TRT/parity
regression has passed.

TensorRT artifacts carry an explicit variant id, precision, builder optimization
level, workspace, and same-contract YAML/manifest. The default deployment
variant is `fp16_o3`; an FP16 failure is saved with its complete error and does
not silently create FP32. An explicit diagnostic variant is separate, for
example `--precision fp32 --builder-optimization-level 0
--artifact-suffix fp32_o0 --force`. The offline v05 visualization keeps
`denoise_cloud=false`, `zfar=100` without applying zfar filtering and writes
separate masks/counts for invalid disparity, remove-invisible, z-range, crop,
and sampling.
