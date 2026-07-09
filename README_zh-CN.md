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
