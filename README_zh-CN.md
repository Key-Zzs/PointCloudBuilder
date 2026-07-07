# PointCloudBuilder

PointCloudBuilder 是一个面向机器人学习流程的 RGB-D 转相机坐标系点云模块。训练数据转换和实时部署必须共用同一个 `PointCloudBuilder` 实现和同一套 YAML schema。

当前实现的是第一阶段：raw RGB-D 反投影。

## 第一阶段范围

- 从 YAML 读取相机内参。
- 使用 PyTorch tensor 进行 depth 反投影。
- 请求 CUDA 且 CUDA 可用时使用 CUDA；CUDA 不可用时自动回退到 CPU。
- `camera.aligned_depth_to_color: true` 时使用 `color_intrinsics`。
- `camera.aligned_depth_to_color: false` 时使用 `depth_intrinsics`。
- 只有在 depth 已对齐到 color、`pointcloud.use_rgb: true`、`pointcloud.output_format: "xyzrgb"` 且输入 frame 有 `rgb` 时才输出 XYZRGB。
- 过滤 `depth <= 0` 的无效点。
- 实时 builder 路径不调用 Open3D、matplotlib 或 GUI 可视化。

裁剪和采样模块保留为后续阶段扩展点，但第一阶段 builder 输出 raw 点云。

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

pc, meta = builder.from_recorded_frame(frame)
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

`pc` 是 `torch.Tensor`，XYZ 时形状为 `N x 3`，XYZRGB 时形状为 `N x 6`。`meta` 至少包含 `stage`、`aligned_depth_to_color`、`use_rgb`、`num_raw_points`、`device`、`timestamp` 和 `global_frame_index`。

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
```

## 离线可视化

可视化脚本和实时 builder 解耦：

```bash
python scripts/visualize_raw_pointcloud.py \
  --config configs/example_head_aligned.yaml \
  --input examples/sample_rgbd.npz
```

## Benchmark

```bash
python scripts/benchmark_deprojection.py \
  --config configs/example_head_aligned.yaml \
  --iters 1000 \
  --warmup 100
```

benchmark 会输出 p50、p95、mean latency ms、点数、device 和分辨率。

## 测试

```bash
pytest -q
python scripts/benchmark_deprojection.py --config configs/example_head_aligned.yaml --iters 100 --warmup 10
```
