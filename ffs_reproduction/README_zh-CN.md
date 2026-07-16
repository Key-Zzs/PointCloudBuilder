# PointCloudBuilder 中的 Fast-FoundationStereo

[English](README.md) · [PointCloudBuilder 中文 README](../README_zh-CN.md)

本目录是 PointCloudBuilder 内自包含的 Fast-FoundationStereo（FFS）集成。FFS
只替换深度来源：

```text
左/右已矫正 IR -> disparity -> 米制 depth
                              |
                              v
原有反投影 -> crop -> 固定点数 sampling
```

运行时不会导入或读取同级 Fast-FoundationStereo checkout。复制源码固定在 commit
`a290ba04c1b3ad1ec41a33974a157b2917b624d4`，来源记录见
[`UPSTREAM_SOURCE.json`](UPSTREAM_SOURCE.json)。复制代码继续受
[`LICENSE.txt`](LICENSE.txt) 中完整 NVIDIA license 约束，仅限非商业研究使用。

## 后端和本地 artifact

| 后端 | 必需本地文件 | 说明 |
| --- | --- | --- |
| `pytorch` | checkpoint + `cfg.yaml` | 参考路线，不需要 ONNX/Engine |
| `tensorrt_single` | 单个 ONNX、Engine、同契约 YAML、manifest | 外部 ImageNet normalization |
| `tensorrt_two_stage` | feature/post ONNX + Engine、YAML、manifest | 两个 Engine 之间执行 Triton GWC |
| `tensorrt_plugin` | plugin ONNX + Engine、YAML、manifest、`.so` | `FFSGWCVolume` CUDA plugin |

下列目录按设计不会进入 Git：

| 路径 | 恢复方法 |
| --- | --- |
| `artifacts/model_best_bp2_serialize.pth`、`cfg.yaml` | 下载官方 `20-30-48` 权重 |
| `artifacts/*.onnx`、manifest、路线 YAML | 运行 `prepare_ffs_artifacts.py` 重新生成 |
| `artifacts/*.engine` | 在目标 TensorRT/GPU 环境重新构建 |
| `build/libffs_gwc_plugin.so` | 运行 `build_ffs_plugin.py` 重新编译 |
| `tensorrt_sdk/` | sparse clone TensorRT headers |
| `outputs/` | 重新运行 smoke/可视化命令 |

PointCloudBuilder 没有托管预构建 Engine 下载包。不要把其他
TensorRT/CUDA/GPU 环境生成的 Engine 当作最终部署产物，必须在目标 `dp3` 环境
重建。

## 已验证目标环境

```text
Python 3.10.20
PyTorch 2.11.0+cu130
TensorRT 10.16.1.11
GPU NVIDIA GeForce RTX 5080 (SM120)
输入  [1,3,480,640] x 2
输出  [1,1,480,640]
max_disp=192, valid_iters=8
```

所有命令都必须使用 `PYTHONNOUSERSITE=1`，否则用户 site-packages 可能覆盖 dp3
内已经验证的依赖。

## 全新 clone 配置

以下命令均从 PointCloudBuilder 仓库根目录执行，不创建第二个 Python 环境。

### 1. 安装项目和可选 FFS 依赖

```bash
cd ~/workspace/3D-Diffusion-Policy/PointCloudBuilder
export PY=~/miniconda3/envs/dp3/bin/python

PYTHONNOUSERSITE=1 "$PY" -m pip install -e '.[dev,viz]'
PYTHONNOUSERSITE=1 "$PY" -m pip install \
  timm==1.0.28 onnx==1.18.0 onnxscript==0.5.6 \
  imageio opencv-python-headless pyarrow av
```

TensorRT 路线还需要目标 TensorRT Python 包：

```bash
PYTHONNOUSERSITE=1 "$PY" -m pip install \
  --extra-index-url https://pypi.nvidia.com/ --no-deps \
  tensorrt_cu13_bindings==10.16.1.11 \
  tensorrt_cu13_libs==10.16.1.11 \
  tensorrt-cu13==10.16.1.11

PYTHONNOUSERSITE=1 "$PY" -c \
  'import torch,tensorrt as trt; print(torch.__version__, torch.version.cuda, trt.__version__, torch.cuda.get_device_name(0))'
```

编译 plugin 还要求 dp3 能使用 CMake 和 CUDA 编译器。

### 2. 下载并校验官方 checkpoint

[FFS 官方仓库](https://github.com/NVlabs/Fast-FoundationStereo)通过此
[Google Drive 目录](https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap?usp=drive_link)
发布权重。可以在浏览器中下载 `20-30-48`，也可以使用 `gdown`：

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

官方 checkpoint 是可信 pickle。PyTorch 后端在局部兼容加载器中显式使用
`torch.load(..., weights_only=False)`，不得替换为不可信 checkpoint。

### 3A. 仅 PyTorch 路线

checkpoint 和配置就绪后不需要构建，直接执行下文
`--backend pytorch` smoke。

### 3B. 只导出 ONNX

全新 clone 中可只导出固定 shape ONNX 和 manifest，不构建 Engine：

```bash
PYTHONNOUSERSITE=1 "$PY" scripts/prepare_ffs_artifacts.py \
  --precision fp16 --builder-optimization-level 3 \
  --artifact-suffix fp16_o3 --skip-tensorrt
```

exporter 显式固定 opset 17、静态 `480x640` 和 `dynamo=False`。普通 ONNX 会
经过 `onnx.checker.check_model`、契约检查和目标 TensorRT parser 检查。

### 3C. 构建全部 TensorRT 路线

只拉取匹配的 TensorRT C++ headers，无需保存整个 TensorRT 仓库：

```bash
git clone --depth 1 --branch v10.16 --filter=blob:none --sparse \
  https://github.com/NVIDIA/TensorRT.git \
  ffs_reproduction/tensorrt_sdk
git -C ffs_reproduction/tensorrt_sdk sparse-checkout set include
```

编译 SM120 plugin：

```bash
PYTHONNOUSERSITE=1 "$PY" scripts/build_ffs_plugin.py \
  --tensorrt-root ffs_reproduction/tensorrt_sdk \
  --cuda-arch 120
readelf -d ffs_reproduction/build/libffs_gwc_plugin.so | grep RUNPATH
```

预期 RUNPATH 为 `$ORIGIN`。随后通过 TensorRT Python API 导出并构建 FP16：

```bash
PYTHONNOUSERSITE=1 "$PY" scripts/prepare_ffs_artifacts.py \
  --precision fp16 --builder-optimization-level 3 \
  --workspace-gib 8 --artifact-suffix fp16_o3
```

如果此前已经执行 ONNX-only 导出，必须追加 `--force`，因为脚本按设计拒绝覆盖
已有派生产物。`--force` 只替换所选 artifact variant，不修改可信 checkpoint。

独立 FP32/o0 诊断构建必须显式执行，且绝不会冒充 FP16：

```bash
PYTHONNOUSERSITE=1 "$PY" scripts/prepare_ffs_artifacts.py \
  --precision fp32 --builder-optimization-level 0 \
  --workspace-gib 8 --artifact-suffix fp32_o0 --force
```

系统不存在精度或后端静默 fallback。构建失败会保存完整
`*.build_error.json`，请求路线保持失败状态。

## 上游 checkout 重命名或删除后的验证

先运行测试，并确认 tracked source 不含机器绝对路径：

```bash
PYTHONNOUSERSITE=1 "$PY" -m pytest -q
git grep -nF "$HOME" || true
```

无 GUI 运行 v05 指定帧的全部 FP16 路线：

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

每条路线都应在 `$OUT/<backend>/` 生成 `left_ir.png`、`right_ir.png`、
`disparity.png`、`depth.png`、各类 mask、`raw.ply`、`cropped.ply`、
`sampled.ply`、`metadata.json`、`parity.json` 和 `stage_counts.json`。
`parity.json` 必须记录 `native_depth_used_for_builder: false`。

如需更严格地确认没有访问重命名后的上游 checkout，可追踪一次 PyTorch 运行：

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

最后一条 `rg` 应没有输出。

## 各后端可视化

移除 `--no-show` 即可查看 IR/disparity/depth 面板和交互点云：

```bash
BACKEND=tensorrt_single  # pytorch | tensorrt_single | tensorrt_two_stage | tensorrt_plugin
PYTHONNOUSERSITE=1 "$PY" scripts/run_v05_ffs_frame.py \
  --dataset-root "$DATASET" --camera head --global-frame-index 0 \
  --backend "$BACKEND" --builder-config "$CONFIG" \
  --artifact-id fp16_o3 --precision fp16 \
  --builder-optimization-level 3 --workspace-gib 8 \
  --output-dir "$OUT"
```

同时打开三个 Open3D 窗口，分别显示已经生成的原始、裁剪和采样点云：

```bash
PYTHONNOUSERSITE=1 "$PY" scripts/view_pointcloud_triplet.py \
  --input-dir "$OUT/$BACKEND"
```

可视化契约不执行 Open3D radius denoise 或 zfar 过滤：
`denoise_cloud=false`、`zfar=100`、`zfar_applied=false`。无效 disparity、
`remove_invisible`、z-range、Builder crop 和 sampling 计数分别记录在
`stage_counts.json`。

以 PyTorch 为参考比较全部 TensorRT 路线：

```bash
PYTHONNOUSERSITE=1 "$PY" scripts/compare_ffs_backends.py \
  --dataset-root "$DATASET" --camera head --global-frame-index 0 \
  --builder-config "$CONFIG" --artifact-id fp16_o3 \
  --precision fp16 --builder-optimization-level 3 --workspace-gib 8 \
  --output ffs_reproduction/outputs/v05_verify/parity_all.json
```

## 运行契约

- 输入是已矫正、去畸变的 `480x640` 原始 `0..255` IR1/IR2。
- 输出 disparity 是左图像素单位的全分辨率 float32。
- 米制深度为 `fx * baseline / disparity`，无效深度统一为零。
- 当前标定 gate 只接受已记录的 identity/no-op rectification 契约，非 identity
  标定直接报错。
- FFS 使用 IR1 内参；可选 RGB 投影使用 IR1-to-color 外参。
- `depth_source.mode=frame` 仍是默认值，不导入 FFS、ONNX、TensorRT、Triton 或
  Open3D。
- Engine、配置、manifest、shape、精度、normalization 和 hash 全部 fail-fast
  校验，不会歧义选择 artifact。

完整 YAML 见 [`configs/v05_ffs.yaml`](configs/v05_ffs.yaml)，实机构建、parity 和
latency 矩阵见
[`../FFS_REPRODUCTION_AND_INTEGRATION_REPORT.md`](../FFS_REPRODUCTION_AND_INTEGRATION_REPORT.md)。
