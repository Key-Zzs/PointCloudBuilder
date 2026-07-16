# PointCloudBuilder

PointCloudBuilder 是一个面向机器人学习流程的 RGB-D 转相机坐标系点云模块。训练数据转换和实时部署必须共用同一个 `PointCloudBuilder` 实现和同一套 YAML schema。

当前实现的是第三阶段：raw RGB-D 反投影 + workspace crop + 固定点数采样；同时提供离线工具，用于从本机
D435i 采集一帧 aligned RGB-D、可视化生成的点云，并 benchmark 反投影、裁剪
和采样等基础模块。

## 第三阶段范围

- 从 YAML 读取相机内参。
- 使用 PyTorch tensor 进行 depth 反投影。
- 请求 CUDA 且 CUDA 可用时使用 CUDA；CUDA 不可用时自动回退到 CPU。
- `camera.aligned_depth_to_color: true` 时使用 `color_intrinsics`。
- `camera.aligned_depth_to_color: false` 时使用 `depth_intrinsics`。
- 只有在 depth 已对齐到 color、`pointcloud.use_rgb: true`、`pointcloud.output_format: "xyzrgb"` 且输入 frame 有 `rgb` 时才输出 XYZRGB。
- 过滤 `depth <= 0` 的无效点。
- 从 YAML 读取 workspace crop 范围。
- 对 `N x 3` XYZ 和 `N x 6` XYZRGB 点云按前三列 XYZ 裁剪，并保留 RGB 列。
- 裁剪为空时返回 `0 x C` tensor，不崩溃。
- 对裁剪后的点云采样到固定点数。
- 支持 `fps`、`stride`、`random`、`voxel`、`voxel_random`、`voxel_fps`。
- `N x 6` XYZRGB 点云采样后保留 RGB 列。
- 输入点不足或为空时，根据 `sampling.pad_mode` 重复补齐或补零。
- 通过 `build_stages()` 暴露 raw、cropped、sampled 三个阶段，供离线调试。
- 实时 builder 路径不调用 Open3D、matplotlib 或 GUI 可视化。
- 支持把单帧 RealSense D435i aligned RGB-D 保存为本地 `.npz` 调试样本，并
  根据相机内参自动写出匹配的 YAML 配置。
当 `sampling.enabled: true` 时，高层 `PointCloudBuilder` 输出固定点数的 sampled 点云。
训练默认建议使用 `voxel_random` 或 `fps`；部署默认建议使用 `voxel_random` 或 `voxel_fps`。

## 安装

```bash
conda create -n pointcloud-builder python=3.10 -y
conda run -n pointcloud-builder python -m pip install -e ".[dev]"
```

离线可视化可选依赖：

```bash
conda run -n pointcloud-builder python -m pip install -e ".[viz]"
```

## 核心接口

```python
from pointcloud_builder import PointCloudBuilder

builder = PointCloudBuilder.from_yaml("configs/example_head_aligned.yaml")

# 离线 zarr 转换
pc, meta = builder.from_recorded_frame(frame)

# 实时推理
pc, meta = builder.from_live_frame(frame)
```

`frame` 是一个 mapping，必须包含 `depth`，可选包含 `rgb`：

```python
frame = {
    "depth": depth_image,  # H x W numpy array 或 torch tensor
    "rgb": rgb_image,      # H x W x 3 可选 numpy array 或 torch tensor
    "timestamp": 1.23,
    "global_frame_index": 42,
}
```

`pc` 是固定点数的 `torch.Tensor`，XYZ 时形状为 `num_points x 3`，XYZRGB 时形状为 `num_points x 6`。`meta` 至少包含 `stage`、`aligned_depth_to_color`、`use_rgb`、`num_raw_points`、`num_cropped_points`、`num_sampled_points`、`crop_enabled`、`crop_range`、`crop_empty`、`sampling_enabled`、`sampling_mode`、`target_num_points`、`input_empty`、`padded`、`pad_mode`、`voxel_size`、`device`、`timestamp` 和 `global_frame_index`。

`N` 是过滤 `depth <= 0` 后的有效 depth 像素数量。XYZ 会乘
`camera.depth_scale` 转成米；启用 `pointcloud.output_format: "xyzrgb"` 时，
RGB 会归一化到 `[0, 1]`。
如果裁剪后没有任何点，采样仍会返回固定大小的全零 tensor，不会崩溃。

## 采样模式

- `stride`：按固定间隔选择点，再补齐或截断到 `num_points`。
- `random`：点数足够时无放回随机选择。
- `fps`：基于 XYZ 的 PyTorch farthest point sampling。
- `voxel`：按 XYZ 做 voxel downsample，每个 voxel 保留输入中的第一个点，再补齐或截断到 `num_points`。
- `voxel_random`：先 voxel，再 random 到固定点数。
- `voxel_fps`：先 voxel，再 FPS 到固定点数。

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
  enabled: true
  mode: "voxel_random"
  num_points: 1024
  voxel_size: 0.005
  seed: 42
  deterministic: false
  pad_mode: "repeat"   # repeat | zero
```

## 真实 D435i 单帧采集

这条路径用于验证本机 RealSense 采到的 RGB-D 是否和后续 LeRobot + RGB-D
sidecar 数据形态一致。

`pyrealsense2` 不是本包依赖。相机工具需要在已有 RealSense Python wrapper 的
环境里运行，例如 Flexiv 工作站上的 `dual_arm_teleop` 环境：

```bash
cd PointCloudBuilder
conda run -n dual_arm_teleop python -m pip install -e ".[viz]"
```

先做相机检测：

```bash
conda run -n dual_arm_teleop python \
  tools/camera/detect_realsense.py
```

用 `rs-enumerate-devices` 确认相机 serial，然后采集一帧 depth-to-color aligned
RGB-D：

```bash
conda run -n dual_arm_teleop python \
  tools/camera/capture_d435i_aligned_rgbd.py \
  --serial 344522070241 \
  --width 424 \
  --height 240 \
  --fps 30 \
  --out captures/head_frame_000000.npz \
  --config-out configs/captures/head_aligned.yaml
```

生成的 `.npz` 包含：

```text
rgb: uint8 [H, W, 3]，和 depth 对齐的 RGB
depth: uint16 [H, W]，对齐到 color 像素网格的 depth
rgb_timestamp, depth_timestamp
depth_scale
width, height, fx, fy, cx, cy
```

生成的 YAML 会使用 color intrinsics，因为
`camera.aligned_depth_to_color: true`。`captures/` 下的大文件会被 `.gitignore`
忽略；`configs/captures/` 下的 YAML 可以保留，用于复现实机测试配置。

## 离线可视化

可视化脚本和实时 builder 解耦：

```bash
python scripts/visualize_raw_pointcloud.py \
  --config configs/captures/head_aligned.yaml \
  --input captures/head_frame_000000.npz \
  --output captures/head_raw.ply
```

无图形界面或只想导出 PLY 时：

```bash
python scripts/visualize_raw_pointcloud.py \
  --config configs/captures/head_aligned.yaml \
  --input captures/head_frame_000000.npz \
  --output captures/head_raw.ply \
  --no-show
```

同时可视化 raw 和 cropped 阶段：

```bash
python scripts/visualize_cropped_pointcloud.py \
  --config configs/captures/head_aligned.yaml \
  --input captures/head_frame_000000.npz \
  --raw-output captures/head_raw.ply \
  --output captures/head_cropped.ply
```

同时可视化 raw、cropped、sampled 三个阶段：

```bash
python scripts/visualize_sampled_pointcloud.py \
  --config configs/example_train_voxel_random.yaml \
  --input captures/head_frame_000000.npz \
  --raw-output captures/head_raw.ply \
  --cropped-output captures/head_cropped.ply \
  --output captures/head_sampled.ply
```

## Benchmark

使用真实采集配置中的分辨率和内参 benchmark raw deprojection：

```bash
python scripts/benchmark_deprojection.py \
  --config configs/captures/head_aligned.yaml \
  --iters 1000 \
  --warmup 100
```

benchmark 会输出 p50、p95、mean latency ms、点数、device 和分辨率。

裁剪和采样工具可以单独 benchmark：

```bash
python scripts/benchmark_crop.py \
  --config configs/example_head_aligned.yaml \
  --num-points 307200 \
  --iters 1000 \
  --warmup 100

python scripts/benchmark_sampling.py \
  --num-points 50000 \
  --target-num-points 1024 \
  --iters 100 \
  --warmup 10

python scripts/benchmark_full_pipeline.py \
  --config configs/example_train_voxel_random.yaml \
  --iters 100 \
  --warmup 10
```

## 数据边界

`.npz` 只是一帧调试和可视化格式，不是计划中的 LeRobot 数据集格式。后续集成时，
RGB 应继续保存在 LeRobot video 字段中，depth/IR 建议保存在 zarr 等 sidecar
数组存储里，并通过 `episode_index`、`frame_index` 和 `camera_name` 与 RGB 对齐。
离线转换和实时部署都应复用同一个 PointCloudBuilder 配置和实现，避免训练/推理的
反投影配置不一致。

## 测试

```bash
pytest -q
python scripts/benchmark_deprojection.py --config configs/example_head_aligned.yaml --iters 100 --warmup 10
python scripts/benchmark_crop.py --config configs/example_head_aligned.yaml --num-points 307200 --iters 100 --warmup 10
python scripts/benchmark_sampling.py --num-points 50000 --target-num-points 1024 --iters 20 --warmup 5
python scripts/benchmark_full_pipeline.py --config configs/example_train_voxel_random.yaml --iters 20 --warmup 5
```

## Fast-FoundationStereo 深度源

`depth_source.mode` 默认为 `frame`，因此旧 RGB-D YAML 和原有 builder 公共
接口保持不变。只有在输入原始 480x640 IR1/IR2 时才设置为 `ffs_stereo`。
FFS 层提供四个显式后端：

| backend | 路径 | 输入归一化 |
| --- | --- | --- |
| `pytorch` | 复制到本仓库的 PyTorch 参考模型 | 模型内部 ImageNet |
| `tensorrt_single` | 单个普通 ONNX/TRT engine | builder 外部执行 ImageNet |
| `tensorrt_two_stage` | TRT feature + Triton GWC + TRT post | 模型内部 ImageNet |
| `tensorrt_plugin` | 包含 `FFSGWCVolume` CUDA plugin 的 TRT 路径 | 模型内部 ImageNet |

四个后端之间没有静默 fallback。checkpoint、engine、plugin、manifest、输入
形状、I/O 名称、精度或 artifact hash 缺失/不匹配时会在构造阶段直接失败。
估计器输出全分辨率 disparity、米制 depth、valid mask、标定/模型 provenance
和 timing；随后复用原有 deprojection、RGB 映射、crop 和固定点数 sampling
流程。FFS 点云使用 IR1 内参；只有显式请求 RGB 投影时才使用 IR1-to-color
外参。

TensorRT 的精度是 artifact 的显式属性，同时记录
`builder_optimization_level: 0..5` 和 `workspace_gib`。FP16 构建失败时会保存
完整 traceback，绝不会静默改标为 FP32；只有明确启动独立诊断构建时，才使用
例如 `fp32_o0` 的不同 artifact id。运行时会校验 Engine/config 的 shape、
max_disp、valid_iters、归一化、精度、资源参数和 hash；配置缺失或候选有歧义会
直接失败。

`ffs_reproduction/` 保存固定 commit 的 FFS `core/` 复制品，并在
`UPSTREAM_SOURCE.json` 中记录文件 hash。复制的文件继续受完整
`ffs_reproduction/LICENSE.txt` 中的 NVIDIA license 约束，仅限非商业研究使用，
不能重新标成 PointCloudBuilder 的 Apache 代码。

### FFS 环境和 artifact

所有 FFS 构建/运行都使用指定的 `dp3` Python：

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

准备 checkpoint/config、固定形状 ONNX 和 manifest：

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python \
  scripts/prepare_ffs_artifacts.py --skip-tensorrt
```

默认直接读取仓库内
`ffs_reproduction/artifacts/model_best_bp2_serialize.pth` 和 `cfg.yaml`，不访问
Fast-FoundationStereo checkout。`--source-root` 只用于向空 artifact 目录进行一次性
可信资产导入。
artifact 和 build 目录按要求不会进入 Git；迁移到全新 clone 时，必须另行保留或
恢复 checkpoint、ONNX/Engine、plugin 动态库和 manifest。

已有的 ONNX/engine/manifest 派生产物不会被静默覆盖；只有明确重建时才使用
`--force`。

TensorRT C++ headers 准备好后，显式为 RTX 5080 的 SM120 构建 CUDA plugin：

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python \
  scripts/build_ffs_plugin.py --tensorrt-root ffs_reproduction/tensorrt_sdk
```

随后使用 TensorRT Python API 构建四个部署所需 engine；脚本不会调用
`trtexec`：

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python \
  scripts/prepare_ffs_artifacts.py
```

ONNX exporter 明确固定为 PyTorch legacy exporter：`opset_version=17`、静态
`480x640` shape，并显式传入 `dynamo=False`。普通 single 导出后会执行
`onnx.checker.check_model`、I/O 契约检查，并由目标 TensorRT Python parser
解析后才接受 Engine。将来若重新尝试 Dynamo，必须先完成同一套 ONNX/TRT/parity
全回归，不能直接删除这个固定项。

默认变体为 `fp16_o3`。如果 FP16 在目标 dp3/TRT 版本失败，显式生成独立的
FP32/o0 诊断产物，不覆盖 FP16：

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python \
  scripts/prepare_ffs_artifacts.py --precision fp32 \
  --builder-optimization-level 0 --artifact-suffix fp32_o0 --force
```

### FFS YAML 契约

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

运行 frame 必须提供原始 0..255 的 `left_ir` 和 `right_ir`，形状为
`480x640`（也可为相同范围的灰度 tensor）。当前标定 gate 只接受 v05 的
identity/no-op rectified 契约：两路 IR 内参相同、畸变为零、旋转为 identity、
`IR1 -> IR2 = (-baseline, 0, 0)`。非 identity 标定会直接拒绝，不会在线隐式
执行 rectification。原生 depth 只用于诊断 parity，不会作为 FFS builder 输入。

在线调用仍然使用原有 builder API：

```python
from pointcloud_builder import PointCloudBuilder, StereoIRFrame

builder = PointCloudBuilder.from_yaml("ffs_reproduction/configs/v05_ffs.yaml")
frame = StereoIRFrame(left_ir=left_ir, right_ir=right_ir, timestamp=timestamp)
point_cloud, metadata = builder.from_live_frame(frame)
```

### v05 单帧、可视化、parity 和 benchmark

v05 helper 通过已有的父仓库 reader 读取权威 raw RGB-D sidecar；不会修改父仓库：

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python \
  scripts/run_v05_ffs_frame.py \
  --dataset-root ~/.cache/huggingface/lerobot/flexiv_dual_arm_3d/pick_place_20260713_v05 \
  --camera head --global-frame-index 0 --backend pytorch \
  --builder-config ffs_reproduction/configs/v05_ffs.yaml \
  --output-dir ffs_reproduction/outputs/v05 --no-show
```

`visualize_ffs_stereo_pipeline.py` 会输出 IR PNG、disparity/depth 的 PNG 和
`npy`，以及 invalid disparity、remove-invisible、z-range 的 mask、
raw/cropped/sampled PLY、metadata、timing 和 `stage_counts.json`。离线点云
检查固定为 `denoise_cloud=false`、`zfar=100` 且不执行 zfar 过滤；报告会分别
列出 FFS invalid disparity、`remove_invisible`、z-range、Builder crop 和
sampling 前后点数。
`compare_ffs_backends.py` 以 PyTorch 为参考做 parity，
`benchmark_ffs_backends.py` 执行要求的 warmup 20 / runs 100 CUDA benchmark：

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

`pytest -q` 覆盖 CPU/fake backend 契约测试；TensorRT engine、GPU smoke、parity
和 benchmark 需要对应部署 artifact 就绪后再执行。构建产物和运行输出均被
`.gitignore` 忽略。
