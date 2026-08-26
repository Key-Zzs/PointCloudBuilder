# PointCloudBuilder

PointCloudBuilder 是面向机器人学习部署的 RGB-D 几何流水线。它支持单相机点云构建、
CameraRig 固定相机集成、native depth 与 FFS stereo、真实固定多相机并发采集、host-time
匹配、workspace 变换、确定性的 current-snapshot fusion、独立 Rerun 可视化，以及可选的
persistent TSDF mapping。实时策略输入与可视化、地图维护保持解耦。

## CameraRig、workspace 与 rig fusion

仓库在 `third_party/CameraRig` 固定已复核的 `develop` 部署就绪提交。CameraRig 负责单相机
frame、calibration 与 fixed-mount bundle；PointCloudBuilder 只消费稳定的
`camera_rig.api`，并负责 workspace 变换、版本化多相机列表、host timestamp
匹配、当前 snapshot 的确定性 voxel fusion，以及 fusion 后唯一一次全局采样。

Native depth XYZ 的源坐标系是 depth optical frame；FFS XYZ 的源坐标系是
left-IR optical frame。rig pipeline 暴露每相机 camera/workspace、concatenated、
workspace-cropped、fused 与 sampled 阶段及独立 provenance sidecar。面向策略的
current-snapshot 路径不跨时间累计；独立的可选 Open3D 进程从同一次推理得到的逐相机
depth ray 维护 persistent map，绝不把 4096 点策略输入反推成 TSDF。详见
`docs/camera-rig-integration.md`、`docs/offline-rig-orchestration.md` 与
`docs/workspace-fusion.md`。

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

Python 3.10 上已验证并固定的独立 Rerun 与 TSDF 可选依赖：

```bash
python -m pip install -e ".[rerun]"  # rerun-sdk==0.36.3
python -m pip install -e ".[tsdf]"   # open3d==0.19.0
```

## 部署就绪的多相机映射

安装前初始化已复核的 CameraRig pin：

```bash
git submodule update --init --recursive
python -m pip install -e third_party/CameraRig
```

运行 YAML、provision bundle 和报告必须保持私有。双机运行前，先检查 USB 链路/stream
profile，执行不求外参的 target preflight，再让每台固定相机使用同一个 resolved target
完成 provision 并分别验证：

```bash
camera-rig device inspect --config .local/camera_a/runtime.yaml --show-profiles
camera-rig target preflight --camera-config .local/camera_a/runtime.yaml \
  --target .local/target/target_spec.json --frames 60 \
  --policy pose_validated --report .local/reports/camera_a_preflight.json \
  --overlays .local/overlays/camera_a
camera-rig provision fixed ...
camera-rig provision validate ...
```

对 camera B 重复 preflight/provision，然后先运行有界并发采集和 host-time matching，
再启用 fusion：

```bash
python tools/mapping/run_live_rig.py \
  --rig-config .local/configs/live_rig.yaml \
  --mapping-config .local/configs/mapping.yaml \
  --frames 1000 --reopen-frames 60 \
  --output .local/evidence/live_rig \
  --report .local/reports/live_rig.json
```

运行时严格保留两条不同输出路径：

```text
matched cameras -> 逐相机点云 -> snapshot voxel fusion -> 全局采样
matched cameras -> 逐相机 depth + K + T_workspace_from_camera -> 异步 TSDF map
```

第一条是低延迟策略/动态观测；第二条是静态 workspace 历史。生产默认组合为冻结的
TSDF 加当前 fused cloud。`guarded_continuous` 必须显式启用，并在新表面达到配置的
连续固定帧数前屏蔽瞬时像素。
冻结地图完成加载和提取后不再接收 live depth packet；只有独立的 current-snapshot
overlay 继续更新。

不重复执行 FFS 推理，直接记录 native 或 FFS depth，再离线重建：

```bash
python tools/mapping/record_live_rig_depth.py \
  --config .local/configs/live_rig.yaml \
  --matched-sets 300 --depth-source ffs_stereo \
  --output .local/recordings/rig_depth

python tools/mapping/build_tsdf_offline.py \
  --recording .local/recordings/rig_depth \
  --config configs/mapping/tsdf_example.yaml \
  --output .local/maps/static_tsdf
```

在现有 snapshot 验收命令上启用独立 Rerun：

```bash
python tools/mapping/run_live_rig_fusion.py \
  --rig-config .local/configs/live_rig.yaml \
  --mapping-config .local/configs/mapping_acceptance.yaml \
  --matched-sets 300 --output .local/evidence/fusion \
  --report .local/reports/fusion.json \
  --viewer rerun --rerun-spawn \
  --rerun-record .local/rerun/fusion.rrd \
  --viewer-point-budget 30000
```

持续映射使用 `tools/mapping/run_live_tsdf_mapping.py`。Rerun 与 TSDF 分别运行在
spawn 子进程，IPC queue 大小为 1 或 2 并采用 latest-only；消费者落后时丢可视化/
地图更新包，不积压、不阻塞 snapshot 主路径。所有真实配置、采集、depth recording、
map、截图、报告和 `.rrd` 必须留在已忽略的 `.local/`；禁止提交真实 SN、外参、图像、
深度、地图或 RRD。

live map 发布还必须通过 `--snapshot-baseline-report` 提供相同 rig、相同帧数且已经
PASS 的 snapshot-only 报告。CLI 会比较 processed FPS 与端到端 p95；FPS 下降超过
10% 或 p95 增加超过 5 ms 时拒绝发布，并记录 mapper 提交接受/拒绝数、队列与 RSS。
发布还要求至少 32 个子进程 RSS 样本；去掉前 20% warmup 后，首尾四分位中位数增长
不得超过 256 MiB，拟合增长斜率不得超过每 100 帧 5 MiB。
报告还会分别记录帧匹配等待与 snapshot 处理延迟，用于诊断，但不改变端到端门限。

变换统一表示 `T_target_from_source` 并作用于列向量。PCB 保存
`T_workspace_from_camera`；Open3D 接收 synthetic parity 已验证的逆矩阵
`T_camera_from_workspace`。Native TSDF 输入为原始 `uint16` 加设备 scale；FFS 输入为
rectified metric float depth，无效像素严格为 0，scale 为 1。

详细文档：

- [架构](docs/architecture.md)
- [标定部署](docs/deployment-calibration.md)
- [Rerun 可视化](docs/rerun-visualization.md)
- [当前 snapshot 与 persistent map](docs/current-snapshot-vs-persistent-map.md)
- [TSDF 映射](docs/tsdf-mapping.md)
- [TSDF 动态处理](docs/tsdf-dynamic-handling.md)

已知环境债务：`sapien 2.2.1` 声明依赖 `opencv-python`，而当前环境使用 headless OpenCV。
这是加入 Rerun/TSDF 前已有的 `pip check` 警告；本目标不升级 Torch、CUDA、TensorRT
或 OpenCV。

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
  --serial YOUR_DEVICE_SERIAL \
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

FFS 是可选能力。默认 `depth_source.mode=frame`，原生 RGB-D 路径和 Builder
公共 API 不变。`mode=ffs_stereo` 接收已矫正的 `480x640` IR1/IR2，生成米制
深度后继续复用同一套反投影、crop 和 sampling 实现。

可选路线包括 `pytorch`、`tensorrt_single`、`tensorrt_two_stage` 和
`tensorrt_plugin`，不存在静默后端或精度 fallback。复制的 FFS 代码继续受
NVIDIA 非商业研究许可约束。

已验证的可选环境是现有 `dp3`：Python 3.10、PyTorch 2.11/CUDA 13、
TensorRT 10.16.1.11：

```bash
cd ~/workspace/3D-Diffusion-Policy/PointCloudBuilder
export PY=~/miniconda3/envs/dp3/bin/python

PYTHONNOUSERSITE=1 "$PY" -m pip install -e '.[dev,viz]'
PYTHONNOUSERSITE=1 "$PY" -m pip install \
  timm==1.0.28 onnx==1.18.0 onnxscript==0.5.6 \
  imageio opencv-python-headless pyarrow av
```

checkpoint、ONNX、Engine、plugin 动态库和构建输出均不会进入 Git。官方
checkpoint 可以重新下载；ONNX、manifest、Engine 和 plugin 可以在 dp3 中
重新生成。TensorRT Engine 必须在目标 TensorRT/GPU 组合上重新构建。

恢复 checkpoint 后，PyTorch smoke 不需要 TensorRT 构建：

```bash
PYTHONNOUSERSITE=1 "$PY" scripts/run_v05_ffs_frame.py \
  --dataset-root ~/.cache/huggingface/lerobot/flexiv_dual_arm_3d/pick_place_20260713_v05 \
  --camera head --global-frame-index 0 --backend pytorch \
  --builder-config ffs_reproduction/configs/v05_ffs.yaml \
  --artifact-id fp16_o3 --precision fp16 \
  --builder-optimization-level 3 --workspace-gib 8 \
  --output-dir ffs_reproduction/outputs/v05_verify --no-show
```

权重下载、全新 clone 恢复、全部 TensorRT 构建、四路线 smoke/parity，以及同时
显示 raw/cropped/sampled 的三个 Open3D 窗口命令，统一放在以下专门文档：

- [中文 FFS 复现、构建与可视化指南](ffs_reproduction/README_zh-CN.md)
- [English FFS reproduction and deployment guide](ffs_reproduction/README.md)
