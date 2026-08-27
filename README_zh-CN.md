# PointCloudBuilder

PointCloudBuilder 是固定多相机 RGB-D 三维重建系统，支持 CameraRig 标定、FFS 双目深度、
米制 XYZ/XYZRGB、workspace fusion、可选 crop/sampling、Rerun 可视化和持久 TSDF mapping。

[English manual](README.md)

## 1. 概览

生产路线为两台固定 Intel RealSense D435i、同一标定 workspace、Fast-FoundationStereo
（FFS）TensorRT-plugin FP16 深度、dense XYZRGB voxel fusion，以及独立 Open3D TSDF
mapper。重建 tensor、可视化与持久地图是三种独立输出。

## 2. 架构

```text
CameraRig frames -> FFS depth -> camera-frame XYZRGB -> local crop
-> T_workspace_from_camera -> workspace crop -> canonical concatenate
-> voxel centroid fusion -> optional global FPS -> current snapshot

same-pass per-camera depth + K + T_workspace_from_camera
-> independent TSDF process -> extract/crop/sample/mesh -> persistent map
```

TSDF 绝不消费 fused/sampled 点云。Rerun 使用有界 latest-only 子进程队列，不能改变重建
tensor。

## 3. 支持硬件

- 两台通过 USB 3 连接的固定 RealSense D435i。
- 与所选 PyTorch、CUDA、TensorRT 包兼容的 NVIDIA GPU。
- 两台相机共用的一个已知 ChArUco target。
- 已验证部署平台为 Linux。

## 4. 坐标约定

变换命名为 `T_target_from_source`，作用于列向量。Native XYZ 起于
`<camera>/depth_optical`，FFS XYZ 起于 `<camera>/ir_left_optical`。PCB 保存
`T_workspace_from_camera`；Open3D 接收已测试的逆变换 `T_camera_from_workspace`。
XYZ 是以米为单位的 `float32`；RGB 是 `[0,1]` 内的 float RGB。

## 5. Clone

```bash
git clone --branch develop/mapping --recurse-submodules \
  https://github.com/Key-Zzs/PointCloudBuilder.git PointCloudBuilder
cd PointCloudBuilder
git submodule update --init --recursive
```

## 6. 全新环境

标准环境名是 `pcb-reconstruction`；公开安装合同不再依赖 `dp3`。声明式规格文件是
`environment.reconstruction.yml`。

```bash
./scripts/bootstrap_reconstruction_env.sh
conda activate pcb-reconstruction
```

可通过 `PCB_ENV_NAME=my-env` 使用另一个隔离名称。环境固定关键
Python/PyTorch/CUDA/TensorRT/OpenCV ABI，并只安装一个 `cv2` provider：
`opencv-contrib-python-headless==4.14.0.94`。环境还固定安装 OmegaConf，用于反序列化
官方 FFS checkpoint 元数据。

## 7. Doctor

```bash
python scripts/doctor_reconstruction_env.py --no-hardware
python scripts/doctor_reconstruction_env.py --asset-root .local/ffs
```

完整 doctor 检查 Python、Conda、Torch/CUDA/GPU、TensorRT、OpenCV/ArUco、
pyrealsense2、Open3D、Rerun、CameraRig、两台 D435i、USB 3 descriptor 和私有 FFS
bundle，且绝不输出相机序列号。

## 8. 相机发现

```bash
camera-rig device list
camera-rig device inspect --config .local/camera_a/runtime.yaml --show-profiles
python tools/mapping/check_usb_topology.py \
  --identity-map .local/camera_rig/camera_identity_map.json \
  --report .local/reports/usb-topology.json
```

交互确认 identity 后，只把序列号写进已忽略的 `.local/` YAML。

## 9. 标定 target

两台相机使用同一份 target spec。已验证标准 target 为 `charuco_a4_v1`：
`DICT_5X5_100`、7x5 squares、30 mm square、22 mm marker、
`legacy_pattern=false`。使用 CameraRig 生成/检查，按 100% 比例打印，并让
`target_spec.json` 始终跟随实体板。target frame 的 +X 向右、+Y 向上、+Z 向板外。

```bash
mkdir -p .local/target
camera-rig target generate \
  --config third_party/CameraRig/configs/targets/charuco_a4_v1.yaml \
  --output .local/target/charuco_a4_v1
camera-rig target inspect \
  --target .local/target/charuco_a4_v1/target_spec.json
```

若已打印实体板未改变，只需重新生成 artifact；preflight 前须核对 resolved spec 与打印
比例尺。

## 10. 固定相机标定

每台相机 provision 前先执行仅验证姿态的 preflight；camera A/B 必须使用同一
workspace 和 target。

```bash
camera-rig target preflight --camera-config .local/camera_a/runtime.yaml \
  --target .local/target/charuco_a4_v1/target_spec.json \
  --frames 60 --policy pose_validated \
  --report .local/reports/camera_a-preflight.json \
  --overlays .local/overlays/camera_a
camera-rig provision fixed \
  --config .local/camera_a/fixed_provision.yaml \
  --output .local/camera_a/provision
camera-rig provision validate --artifact .local/camera_a/provision
```

camera B 执行同样命令。私有 runtime YAML 从
`third_party/CameraRig/configs/examples/single_camera_contract.yaml` 开始，provision YAML
从 `third_party/CameraRig/configs/examples/fixed_provision_contract.yaml` 开始；只在
`.local/` 写入发现的序列号，让两份 provision 配置指向同一个 target，并设置
`target.detection_policy: pose_validated`，使 provision 使用与必做 preflight 相同的
当前实体板策略。标定残差与姿态稳定性阈值保持不变。

实体 coverage gate 失败后禁止复用旧 extrinsics。

## 11. FFS 设置

FFS 资产是外部私有文件。把官方 `20-30-48` checkpoint 与 `cfg.yaml` 放到
`.local/ffs/artifacts/`。预期 SHA-256：

```text
model_best_bp2_serialize.pth  98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692
cfg.yaml                       d45afe99b176454d5aff416edf16c8da6a99579f8f374b927f37907442a7d6bc
```

来源是 [NVlabs 官方仓库](https://github.com/NVlabs/Fast-FoundationStereo)及其
[官方 weights 目录](https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap?usp=drive_link)。
用浏览器下载 `20-30-48`，再把上述两个命名文件手工复制到目标目录。TensorRT C++
header 可在不依赖旧 clone 的前提下获取：

```bash
git clone --depth 1 --branch v10.16 --filter=blob:none --sparse \
  https://github.com/NVIDIA/TensorRT.git .local/third_party/TensorRT
git -C .local/third_party/TensorRT sparse-checkout set include
```

使用以下入口检查或构建全部 route：

```bash
python scripts/prepare_ffs_assets.py --check --asset-root .local/ffs
python scripts/prepare_ffs_assets.py --build-tensorrt \
  --asset-root .local/ffs \
  --checkpoint path/to/20-30-48/model_best_bp2_serialize.pth \
  --model-config path/to/20-30-48/cfg.yaml \
  --tensorrt-root .local/third_party/TensorRT
```

依次 smoke `pytorch`、`tensorrt_single`、`tensorrt_two_stage`、
`tensorrt_plugin`；生产只使用 FP16 `tensorrt_plugin`。Smoke CLI 支持
`--artifact-dir .local/ffs/artifacts` 和
`--plugin-library .local/ffs/build/libffs_gwc_plugin.so`。每个 backend 从
`configs/mapping/ffs_workspace_example.yaml` 创建私有 pipeline YAML，写入该 backend
及已检查的资产路径。Standalone smoke 配置必须保持 `pointcloud.use_rgb: false`，
因为它没有权威 IR→color 外参。然后对同一个全新 CameraRig NPZ frame 运行：
共享 offline loader 会把 CameraRig snapshot 的 `ir_left`、`ir_right`、`color`
规范化为 PCB canonical `left_ir`、`right_ir`、`rgb` 键。

```bash
for backend in pytorch tensorrt_single tensorrt_two_stage tensorrt_plugin; do
  python scripts/run_ffs_stereo_frame.py \
    --config ".local/configs/ffs_${backend}.yaml" \
    --input .local/captures/camera_a/frames/frame_000000.npz \
    --output-dir ".local/evidence/ffs-${backend}" --no-show
done
```

Live RGB 重建应从已检查的 plugin route 单独创建
`.local/configs/ffs_tensorrt_plugin_rgb.yaml`，设置 `pointcloud.use_rgb: true` 和
`output_format: xyzrgb`。Live CameraRig 集成会提供权威 IR→color 外参；不得在
standalone smoke 配置中编造外参。

## 12. 单相机 XYZ/XYZRGB

```bash
python tools/mapping/run_live_single_camera.py \
  --camera-config .local/camera_a/runtime.yaml \
  --provision .local/camera_a/provision \
  --mapping-config .local/configs/mapping.yaml \
  --ffs-config .local/configs/ffs_tensorrt_plugin_rgb.yaml \
  --depth-source ffs_stereo --frames 300 \
  --output .local/evidence/camera_a \
  --report .local/reports/camera_a.json
```

`pointcloud.use_rgb: true` 输出 Nx6。落在 color imager 视场外的 depth 点使用显式黑色，
不会伪造颜色。

## 13. 多相机采集

```bash
python tools/mapping/run_live_rig.py \
  --rig-config .local/configs/live_rig_ffs_rgb.yaml \
  --mapping-config .local/configs/mapping.yaml \
  --frames 1000 --reopen-frames 60 \
  --acceptance-scope capture_matching \
  --output .local/evidence/live-rig \
  --report .local/reports/live-rig.json
```

Camera session 由 worker 独占，open/capture/close 始终在同一 worker thread。
`capture_matching` 是 CR7 scope：强制并发采集、完整交付 matched sets、host-time
skew、无帧复用和干净 lifecycle。它仍记录 geometry 与 FFS performance，但其正式
benchmark 解释由 CR18 负责。

## 14. 帧匹配

Live rig 只使用 `host_receive_timestamp_ns` 和配置的最大 skew 匹配。设备 timestamp 与
frame number 仅用于诊断。Buffer 有界且偏向最新帧；同一帧不会被复用于多个 matched set。

## 15. Raw concatenation

从 `configs/mapping/raw_rgb_concatenation_example.yaml` 开始。它关闭 fusion 与 sampling，
通过 dense `/clouds/concatenated` 检查标定重合和调试问题。

```bash
python tools/mapping/run_live_reconstruction_profile.py \
  --profile raw --rig-config .local/configs/live_rig_ffs_rgb_raw.yaml \
  --mapping-config .local/configs/mapping_acceptance.yaml --matched-sets 60 \
  --output .local/evidence/raw-rgb --report .local/reports/raw-rgb.json \
  --viewer rerun --rerun-spawn --rerun-record .local/evidence/raw-rgb.rrd
```

## 16. Dense RGB fusion

推荐 profile 是 `configs/mapping/dense_rgb_reconstruction_example.yaml`：2.5 mm fusion
开启、sampling 关闭。Voxel key 只使用 XYZ；输出 XYZ 是 voxel centroid，输出 RGB 是
算术均值。`/clouds/fused` 是变长 dense XYZRGB。2.5 mm 来自同输入 2.5/5/10 mm benchmark：
它明显恢复细节，同时没有有意义的 fusion p95 代价。

Voxel fusion 与 sampling 不是同一个操作：

- Voxel fusion 把多相机重叠 observation 合并成 voxel centroid。
- Sampling 只用于可选的输出尺寸缩减。

推荐 dense reconstruction 为 fusion ON、sampling OFF。

```bash
python tools/mapping/run_live_reconstruction_profile.py \
  --profile dense --rig-config .local/configs/live_rig_ffs_rgb.yaml \
  --mapping-config .local/configs/mapping_acceptance.yaml --matched-sets 60 \
  --output .local/evidence/dense-rgb --report .local/reports/dense-rgb.json
```

## 17. Crop

生产顺序是：可选 camera-frame local crop、workspace transform、逐相机唯一一次 workspace
crop、canonical concatenate、voxel fusion、可选全局 sampling。Crop 仅按 XYZ 判断并保留
RGB 列。

## 18. 可选 sampling

`configs/mapping/compact_rgb_reconstruction_example.yaml` 在 fusion 后执行一次全局
30,000 点 FPS，并保留所选点的 RGB。`voxel_fps` 与 `voxel_random` 作为显式高级兼容模式
保留，但不是已 voxel-fused cloud 的默认值。

```bash
python tools/mapping/run_live_reconstruction_profile.py \
  --profile compact --rig-config .local/configs/live_rig_ffs_rgb_compact.yaml \
  --mapping-config .local/configs/mapping_acceptance.yaml --matched-sets 1 \
  --output .local/evidence/compact-rgb --report .local/reports/compact-rgb.json
```

## 19. Rerun 实时可视化

Rerun 展示相机 RGB/frustum、逐相机 workspace、concatenated/fused/sampled cloud、TSDF
entity 和 scalar metric。Nx6 使用真实 RGB；Nx3 使用默认颜色。`viewer_point_budget` 只限制
发给 Rerun 的 packet，绝不修改重建 tensor。

## 20. Interactive 无限模式

日常 operator 命令：

```bash
python tools/mapping/run_live_tsdf_mapping.py \
  --rig-config .local/configs/live_rig_ffs_rgb.yaml \
  --tsdf-config .local/configs/tsdf_frozen_ffs.yaml \
  --initial-map .local/maps/static_ffs \
  --interactive
```

它自动启动 Rerun，不要求 `--report`，没有 matched-set 上限，统计窗口有界，并在
Ctrl+C/SIGTERM 清理后以 0 退出。可用 `--rerun-connect`、`--rerun-record` 或
`--viewer-point-budget 200000` 覆盖默认。Interactive 不是正式 acceptance；finite mode
继续支持 `--matched-sets 300 --report ...`。

## 21. Depth recording

```bash
python tools/mapping/record_live_rig_depth.py \
  --config .local/configs/live_rig_ffs_rgb.yaml \
  --matched-sets 300 --depth-source ffs_stereo \
  --output .local/recordings/rig-depth
```

Recording 保存 same-pass matched 逐相机 depth、intrinsics、transform 与 backend
provenance，不会再次运行 FFS。

## 22. Offline TSDF

```bash
python tools/mapping/build_tsdf_offline.py \
  --recording .local/recordings/rig-depth \
  --config configs/mapping/tsdf_example.yaml \
  --output .local/maps/static-ffs
python tools/mapping/extract_tsdf_geometry.py \
  --map .local/maps/static-ffs --output .local/evidence/static-ffs
```

Extraction 支持 raw point cloud、可选 crop/sampling 与 mesh。

## 23. Live TSDF

正式 finite mapping 必须提供相同 rig、相同帧数且 PASS 的 snapshot baseline，并写私有
report。`build_static` 和 `guarded_continuous` 集成逐相机 depth；mapper child 错误、队列
违规、性能回归或 RSS 增长都会阻断发布。

## 24. Frozen static + dynamic overlay

`frozen_static` 必须提供 `--initial-map`。TSDF load 后不接收 live depth；当前 fused RGB
overlay 继续独立更新。这是静态 workspace 的推荐 operator 组合。

## 25. Map save/load

```bash
python tools/mapping/validate_tsdf_map.py --map .local/maps/static-ffs
python tools/mapping/extract_tsdf_geometry.py \
  --map .local/maps/static-ffs --output .local/evidence/reloaded-map
```

Artifact 包含 checksum、resolved config、volume、metric 和提取 geometry。

## 26. Benchmark

```bash
python tools/mapping/benchmark_fusion_voxels.py \
  --rig-config .local/configs/live_rig_ffs_rgb.yaml \
  --mapping-config .local/configs/mapping.yaml --frames 30 \
  --report .local/reports/fusion-voxel-sweep.json
python tools/mapping/benchmark_world_reconstruction.py \
  --rig-config .local/configs/replay_rig.yaml --frames 100 --warmup 10 \
  --report .local/reports/reconstruction.json
```

Timing 分开记录 depth inference、RGB mapping、deprojection、local/workspace crop、
transform、concatenate、voxel fusion、可选 sampling、`raw_to_world_fused`、TSDF update
与 TSDF extraction。

## 27. Validation

```bash
pytest -q
python -m build
python scripts/check_documented_commands.py
python scripts/doctor_reconstruction_env.py --no-hardware
```

硬件验收还覆盖 CameraRig provision、四个 FFS backend、双机 RGB、Rerun、offline/live
TSDF、save/load、资源 plateau 与 camera reopen。

## 28. Troubleshooting

如果 `/clouds/fused` 看起来稀疏，按顺序检查：

1. `actual_fused_points`；
2. `fusion.voxel_size_m`；
3. `viewer_point_budget`；
4. 当前选择的是 `/clouds/fused` 还是 `/clouds/sampled`。

Viewer budget 不修改重建 tensor。黑点可能是 color FOV 外的有效 depth。Worker、FFS、
mapper 或 viewer child 错误都是 fatal。禁止放松标定/几何阈值来制造 PASS。

## 29. Private local artifacts

全部序列号、runtime YAML、provision bundle、实体 capture、checkpoint、engine、plugin、
report、map、截图与 RRD 必须位于已忽略的 `.local/`。Public example 使用 placeholder。
复现时禁止用 `PYTHONPATH` 或 symlink 指向旧 clone。

## 30. Future 500x700 deployment board

500x700 mm board 状态为 `DEFERRED`，不是 clean-room gate。禁止根据尺寸或成功 corner
detection 推断 ArUco dictionary/`legacy_pattern`。未来部署必须获得权威 generator metadata/
制作者确认，或者打印已知 spec 的新板并重新 provision 两台相机。
