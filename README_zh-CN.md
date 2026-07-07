# PointCloudBuilder

PointCloudBuilder 是一个面向机器人学习流程的 RGB-D 转点云模块。它的核心约束是：
训练数据转换和实时部署必须共用同一个 `PointCloudBuilder`，不能在训练脚本和控制代码里各自维护一套反投影、裁剪和采样逻辑。

当前版本先建立可扩展的基础骨架，并提供一个最小可运行的 PyTorch tensor 数据通路：

- 从 YAML 读取相机内参、裁剪范围、采样策略和 aligned depth-to-color 配置。
- 使用 PyTorch tensor 将 depth 反投影为相机坐标系 XYZ 点云。
- aligned depth-to-color 使能时附加 RGB，未使能时只输出 XYZ。
- 按配置的 3D 范围裁剪点云。
- 支持固定点数输出，裁剪为空时补零返回，不会崩溃。
- 提供 stride、random、fps、voxel、voxel_random、voxel_fps 采样模式的基础实现。
- CUDA 可用时默认使用 CUDA，不可用时自动回退到 CPU，接口保持一致。

实时路径不依赖 Open3D。原始点云、裁剪点云、采样点云的分阶段可视化应通过 `visualization.py` 或 `scripts/` 里的离线脚本调用，不能耦合到实时控制路径。

## 安装

```bash
conda create -n pointcloud-builder python=3.10 -y
conda run -n pointcloud-builder python -m pip install -e ".[dev]"
```

如果需要离线 Open3D 可视化，再安装可视化依赖：

```bash
conda run -n pointcloud-builder python -m pip install -e ".[viz]"
```

## 核心接口

```python
from pointcloud_builder import PointCloudBuilder

builder = PointCloudBuilder.from_yaml("configs/example_head_aligned.yaml")

# 离线 zarr 或 recorded frame 转换
pc, meta = builder.from_recorded_frame(frame)

# 实时推理
pc, meta = builder.from_live_frame(frame)
```

`frame` 可以是一个 mapping，至少包含 `depth`，可选包含 `color`：

```python
frame = {
    "depth": depth_image,  # H x W，uint16 毫米深度或 float 深度单位
    "color": color_image,  # H x W x 3，可选，仅 aligned 模式使用
}
```

返回的 `pc` 是固定点数的 `torch.Tensor`：

- 未启用 RGB 时形状为 `(num_points, 3)`。
- 启用 RGB 且输入包含 `color` 时形状为 `(num_points, 6)`。

`meta` 会记录 raw、cropped、sampled 阶段的点数、设备和采样模式。

## 配置

YAML 配置包含：

- 相机内参：`fx`、`fy`、`cx`、`cy`、width、height、depth scale。
- aligned depth-to-color 模式选择。
- 相机坐标系下的裁剪范围。
- 采样模式和固定输出点数。
- 设备策略，`auto` 表示 CUDA 可用时使用 CUDA，否则使用 CPU。

示例配置位于 `configs/`：

- `example_head_aligned.yaml`
- `example_head_depth_raw.yaml`

训练默认建议使用 `voxel_random` 或 `fps`。
部署默认建议使用 `voxel_random` 或 `voxel_fps`。

## 训练和部署共用约束

训练转换和部署推理必须使用同一个 `PointCloudBuilder` 包和同一套 YAML schema。
离线转换可以调用 `from_recorded_frame`，实时推理可以调用 `from_live_frame`，但二者都必须走同一个 builder 实现。

## 测试

```bash
conda run -n pointcloud-builder python -m pytest
```
