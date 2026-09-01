# PointCloudBuilder

PointCloudBuilder 是固定多相机 RGB-D 三维重建系统，支持 CameraRig 标定、FFS 双目深度、
米制 XYZ/XYZRGB、workspace fusion、可选 crop/sampling、Rerun 可视化和持久 TSDF mapping。

[English manual](README.md)

## 1. 概览

Runtime 支持 `N >= 2` 台固定 Intel RealSense D435i。下一次经过验证的部署目标是同一
标定 workspace 内的三台固定 D435i，采用 Fast-FoundationStereo（FFS）
TensorRT-plugin FP16 深度、dense XYZRGB voxel fusion，以及独立 Open3D TSDF mapper。
重建 tensor、可视化与持久地图是三种独立输出。

0.2.0 工作流已从全新 clone 和隔离环境完成复现，覆盖固定相机标定、FFS、双相机 RGB、
Rerun、offline/live TSDF 及地图 save/load。硬件 identity 与实体证据仍只保存在私有
`.local/` 下。

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

- `N >= 2` 台通过 USB 3 连接的固定 RealSense D435i；下一次部署目标为三台。
- 与所选 PyTorch、CUDA、TensorRT 包兼容的 NVIDIA GPU。
- 全部固定相机共用的一个已知 ChArUco target。
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
官方 FFS checkpoint 元数据。bootstrap 会把 `PYTHONNOUSERSITE=1` 持久写入指定 Conda
环境，避免 `~/.local` 中的包静默覆盖上述版本；重新运行 bootstrap 后要重新激活环境。
若 `cv2` 仍来自 user site，或模块版本与已安装 wheel 不一致，Doctor 会直接 FAIL。

### 6.1 同一批 D435i 迁移到新电脑

本节适用于相机序列号、固定安装位姿、workspace 和实体标定板均未改变，仅更换主机的
情况。若移动过任一相机、workspace 或标定板，不得复用旧外参；按第 8–10 节重新发现、
preflight 和 provision。

#### 6.1.1 迁移私有相机资产

`.local/` 被 Git 忽略，其中可能包含相机序列号、标定和本地绝对路径。只通过可信的
加密介质或受控连接迁移，不要提交到仓库。不要整包复制旧 `.local/`；保留相对目录结构，
只迁移当前 rig YAML 实际引用的内容：

- `.local/camera_rig/camera_identity_map.json`；
- 每台相机的 runtime YAML 和已验证 provision artifact；
- 与未改变实体板对应的 target spec/metadata；
- 当前生产 rig/pipeline/TSDF YAML；
- rig YAML 引用的已提升 rig-calibration artifact；
- 只有在继续旧地图时，才迁移命令所引用且标定 provenance 一致的 initial map。

旧 recordings、evidence、logs、FFS smoke 输出和旧 TensorRT Engine 都不是运行必需项。
迁移后检查 YAML/JSON 中的旧主机绝对路径并改成新 checkout 内的相对路径；不要修改序列号、
内参、外参或标定数值：

```bash
find .local -type f \( -name '*.yaml' -o -name '*.yml' -o -name '*.json' \) \
  -exec grep -nEH '/home/|/Users/|^[[:space:]]*[A-Za-z]:\\' {} +
camera-rig device list
camera-rig provision validate --artifact .local/camera_rig/camera_a/provision
```

对 rig 中每台相机分别执行 `provision validate`，并确认发现的设备、identity map 与 runtime
YAML 仍指向同一实体 D435i。若目录名不同，使用生产 rig YAML 的
`source.provision_artifact` 路径。随后按第 8 节运行 USB topology 检查。

#### 6.1.2 下载并校验官方 FFS 权重

FFS 权重来自 [NVlabs 官方仓库](https://github.com/NVlabs/Fast-FoundationStereo) 的 [官方 weights 目录](https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap?usp=drive_link)。
在网页中下载 `20-30-48` 目录，或在已经激活的 `pcb-reconstruction` 环境中使用
`gdown`：

```bash
python -m pip install gdown
FFS_DOWNLOAD_DIR="${PWD}/.local/downloads/fast-foundationstereo-weights"
python -m gdown \
  'https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap?usp=drive_link' \
  --folder -O "$FFS_DOWNLOAD_DIR"

FFS_WEIGHT_FILE="$(find "$FFS_DOWNLOAD_DIR" \
  -path '*/20-30-48/model_best_bp2_serialize.pth' -print -quit)"
test -n "$FFS_WEIGHT_FILE"
FFS_WEIGHT_DIR="$(dirname "$FFS_WEIGHT_FILE")"
mkdir -p .local/ffs/artifacts
install -m 0644 "$FFS_WEIGHT_DIR/model_best_bp2_serialize.pth" \
  .local/ffs/artifacts/model_best_bp2_serialize.pth
install -m 0644 "$FFS_WEIGHT_DIR/cfg.yaml" .local/ffs/artifacts/cfg.yaml

printf '%s  %s\n' \
  98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692 \
  .local/ffs/artifacts/model_best_bp2_serialize.pth \
  d45afe99b176454d5aff416edf16c8da6a99579f8f374b927f37907442a7d6bc \
  .local/ffs/artifacts/cfg.yaml | sha256sum -c -
```

网页下载时，将其中的 `model_best_bp2_serialize.pth` 和 `cfg.yaml` 手工复制到同一目标
目录，再执行上面的 `printf ... | sha256sum -c -`。checkpoint 由
`torch.load(..., weights_only=False)` 加载，因此只能使用上述经过哈希校验的可信官方文件。

#### 6.1.3 在新电脑重建 TensorRT 资产

不要从旧电脑复用 `.engine` 或 `libffs_gwc_plugin.so`。默认 TensorRT Engine 绑定构建时的
平台、TensorRT 版本和 GPU compute capability，具体规则见 [NVIDIA Engine Compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)；
新主机应使用已固定的环境和目标 GPU 重新构建。先获取匹配的 TensorRT 10.16 C++ headers：

```bash
git clone --depth 1 --branch v10.16 --filter=blob:none --sparse \
  https://github.com/NVIDIA/TensorRT.git .local/third_party/TensorRT
git -C .local/third_party/TensorRT sparse-checkout set include

FFS_CUDA_ARCH="$(python -c \
  'import torch; p=torch.cuda.get_device_capability(); print(f"{p[0]}{p[1]}")')"
printf 'Target CUDA architecture: %s\n' "$FFS_CUDA_ARCH"

python scripts/prepare_ffs_assets.py --build-tensorrt \
  --asset-root .local/ffs \
  --checkpoint .local/ffs/artifacts/model_best_bp2_serialize.pth \
  --model-config .local/ffs/artifacts/cfg.yaml \
  --tensorrt-root .local/third_party/TensorRT \
  --cuda-arch "$FFS_CUDA_ARCH"
python scripts/prepare_ffs_assets.py --check --asset-root .local/ffs
```

生产使用生成的 FP16 `tensorrt_plugin` route。不要手工抄写 Engine、plugin、manifest 或
backend-config 路径；第 11 节的 `prepare_ffs_pipeline_configs.py` 会从已检查的
`.local/ffs/` bundle 自动生成全部私有 pipeline YAML。相机序列号和 CameraRig 标定不属于
FFS asset bundle，继续由迁移并验证过的 runtime/provision/rig 配置提供。按第 11 节完成
PyTorch、三条 TensorRT route 和同一个全新 CameraRig NPZ frame 的 smoke 后，再运行下方
Doctor；资产检查 PASS 不能替代实际 Engine 加载和相机 smoke。

## 7. Doctor

```bash
python scripts/doctor_reconstruction_env.py --no-hardware
python scripts/doctor_reconstruction_env.py --expected-d435i-count 3 \
  --asset-root .local/ffs
```

完整 doctor 检查 Python、Conda、Torch/CUDA/GPU、TensorRT、OpenCV/ArUco、
pyrealsense2、Open3D、Rerun、CameraRig、配置的 D435i 数量、USB 3 descriptor 和私有
FFS bundle。`--no-hardware` 仍不接触硬件，向后兼容的默认期望数量为两台；doctor
绝不输出序列号。

## 8. 相机发现

新 clone 不包含被 Git 忽略的 `.local/` 配置。先只连接属于当前 rig 的三台 D435i；若主机
还连接着其他 RealSense，先断开它们，不得从五台设备中自动挑选三台。只读列出设备：

```bash
camera-rig device list --driver realsense
```

确认三台设备的物理 identity 后生成私有 identity map。首次建立时按 USB physical port 的
稳定排序命名为 `camera_a/b/c`；命令只在数量、型号和 USB 3 link 全部 PASS 时才写 identity
map，失败报告不会建立新的身份绑定：

```bash
python tools/mapping/check_usb_topology.py \
  --identity-map .local/camera_rig/camera_identity_map.json \
  --expected-count 3 \
  --report .local/reports/usb-topology.json
```

若之前误用五台设备建立过 identity map，先人工核对，再把错误文件移动到私有备份位置，
断开非 rig 设备并重新运行上面的命令；不要在未确认物理相机身份时覆盖映射。序列号只保留
在 `.local/camera_rig/camera_identity_map.json` 和稍后自动生成的私有 YAML 中。

现有实体板尚未注册时，先只生成 capture/preflight 所需的 runtime YAML；该阶段不需要
target，也不会生成 provision YAML：

```bash
python scripts/prepare_camera_rig_calibration.py \
  --identity-map .local/camera_rig/camera_identity_map.json \
  --asset-root .local/camera_rig \
  --expected-camera-count 3 \
  --runtime-only \
  --report .local/reports/camera-rig-runtime-preparation.json
```

## 9. 标定 target

全部相机必须使用与同一实体板精确匹配的 resolved target artifact。现有 500 x 700 mm
部署板和新生成的 A4 板是两条互斥路线，不得混用。

### 9.1 现有 500 x 700 部署板（当前 rig 使用）

该实体板的权威已知元数据是 `DICT_4X4_50`、5 x 7 squares、100 mm square、75 mm
marker。它不是 `charuco_a4_v1`，禁止运行 A4 `target generate` 后拿生成的 spec 标定该
实体板。`legacy_pattern`、`border_bits` 和 canonical orientation 仍由多相机图像证据决定。

使用第 8 节生成的 runtime YAML，从每台最终相机采集独立的 8 帧证据；输出目录必须尚不
存在，实体板应清晰可见：

```bash
set -euo pipefail
for camera in camera_a camera_b camera_c; do
  camera-rig capture snapshot \
    --config ".local/camera_rig/${camera}/configs/runtime.yaml" \
    --frames 8 \
    --output ".local/captures/target-id/${camera}"
done
```

只把已有证据支持的 dictionary 作为权威约束；不要猜 `legacy_pattern`、`border_bits` 或
orientation，也不要把 `DICT_4X4_50` 换成 `DICT_4X4_100`：

```bash
camera-rig target identify-existing \
  --artifact .local/captures/target-id/camera_a \
  --artifact .local/captures/target-id/camera_b \
  --artifact .local/captures/target-id/camera_c \
  --board-width-mm 500 --board-height-mm 700 \
  --square-length-mm 100 --marker-length-mm 75 \
  --maximum-artifact-frames 8 \
  --authoritative-dictionary DICT_4X4_50 \
  --output .local/reports/charuco_500x700-identification.json

camera-rig target register-existing \
  --identification .local/reports/charuco_500x700-identification.json \
  --target-name charuco_500x700 \
  --target-frame charuco_500x700 \
  --output .local/camera_rig/shared_target/charuco_500x700

camera-rig target inspect \
  --target .local/camera_rig/shared_target/charuco_500x700/target_spec.json
(cd .local/camera_rig/shared_target/charuco_500x700 && sha256sum -c checksums.sha256)
```

若 identify 返回 ambiguity/`PAUSED_FOR_USER_VALIDATION`，必须用原 PDF、生成器元数据或板主
提供的真实证据解决后再注册；不能降低检测门槛或选择“看起来能过”的候选。

### 9.2 新生成的标准 A4 板（仅在实际改用并打印新板时）

标准 `charuco_a4_v1` 是 `DICT_5X5_100`、7 x 5 squares、30 mm square、22 mm marker、
`legacy_pattern=false`。只有实际按 100% 比例打印并使用该新板时才运行：

```bash
mkdir -p .local/camera_rig/shared_target
camera-rig target generate \
  --config third_party/CameraRig/configs/targets/charuco_a4_v1.yaml \
  --output .local/camera_rig/shared_target/charuco_a4_v1
camera-rig target inspect \
  --target .local/camera_rig/shared_target/charuco_a4_v1/target_spec.json
(cd .local/camera_rig/shared_target/charuco_a4_v1 && sha256sum -c checksums.sha256)
```

不得把 `target_spec.json` 与旧电脑写死的哈希比较。resolved spec 包含 CameraRig/OpenCV
版本和生成文件哈希；不同的有效环境可能得到不同 spec SHA。权威检查是
`camera-rig target inspect` 加 artifact 自带的 `checksums.sha256`。迁移时必须复制整个
target artifact 目录，不能只复制 JSON。

## 10. 固定相机标定

以下流程假设三台固定相机使用同一实体板，并且把 `workspace` 明确定义为所选 artifact 的
`target_frame`（第 9.1 节为 `charuco_500x700`，第 9.2 节为 `charuco_target`）；CameraRig
fixed-provision 契约要求 `T_workspace_from_target` 为单位矩阵。若你的 workspace 不是实体板
坐标系，不能使用此流程伪造变换，必须先建立正确的 workspace/target 合同。

### 10.1 自动生成每台相机的私有配置

脚本从第 8 节确认过的 identity map 读取序列号，不在终端或报告中输出序列号；它校验第 9
节完整 target artifact，并为每台相机生成 `configs/runtime.yaml` 和
`configs/fixed_provision.yaml`：

```bash
PCB_TARGET_SPEC=.local/camera_rig/shared_target/charuco_500x700/target_spec.json
python scripts/prepare_camera_rig_calibration.py \
  --identity-map .local/camera_rig/camera_identity_map.json \
  --target "$PCB_TARGET_SPEC" \
  --asset-root .local/camera_rig \
  --expected-camera-count 3 \
  --workspace-equals-target \
  --report .local/reports/camera-rig-calibration-preparation.json
```

若实际改用第 9.2 节新打印的 A4 板，只把 `PCB_TARGET_SPEC` 改成
`.local/camera_rig/shared_target/charuco_a4_v1/target_spec.json`。

若之前手工创建的 YAML 与自动合同不同，脚本默认停止且不覆盖。核对差异后，可在同一命令
追加 `--update-existing`；脚本会先在原目录生成 `*.bak-<UTC>` 私有备份，再替换冲突文件。
随后执行配置只读检查（除写入指定 report 外，不修改私有配置）：

```bash
PCB_TARGET_SPEC=.local/camera_rig/shared_target/charuco_500x700/target_spec.json
python scripts/prepare_camera_rig_calibration.py \
  --identity-map .local/camera_rig/camera_identity_map.json \
  --target "$PCB_TARGET_SPEC" \
  --asset-root .local/camera_rig \
  --expected-camera-count 3 \
  --workspace-equals-target \
  --check \
  --report .local/reports/camera-rig-calibration-check.json
```

### 10.2 配置与非硬件输入检查

逐台确认 runtime 绑定的设备和 profiles，再让 CameraRig 校验完整 provision 输入。dry-run
不打开相机，也不写 provision artifact：

```bash
set -euo pipefail
PCB_TARGET_SPEC=.local/camera_rig/shared_target/charuco_500x700/target_spec.json
for camera in camera_a camera_b camera_c; do
  camera-rig device inspect \
    --config ".local/camera_rig/${camera}/configs/runtime.yaml" --show-profiles
  camera-rig provision fixed \
    --config ".local/camera_rig/${camera}/configs/fixed_provision.yaml" \
    --output ".local/camera_rig/${camera}/provision" \
    --dry-run
done
```

### 10.3 实体板 preflight

保持相机、workspace 和实体板固定，让同一实体板清晰出现在每台相机的 color 画面中。逐台
采集严格的 60 帧 pose-validated preflight；任一相机失败就停止，不要降低阈值：

```bash
set -euo pipefail
PCB_TARGET_SPEC=.local/camera_rig/shared_target/charuco_500x700/target_spec.json
for camera in camera_a camera_b camera_c; do
  camera-rig target preflight \
    --camera-config ".local/camera_rig/${camera}/configs/runtime.yaml" \
    --target "$PCB_TARGET_SPEC" \
    --frames 60 --policy pose_validated \
    --report ".local/reports/${camera}-preflight.json" \
    --overlays ".local/overlays/${camera}"
done
```

### 10.4 Provision 与验证

全部 preflight PASS 后，在相机、workspace 和实体板均未移动的情况下逐台 provision：

```bash
set -euo pipefail
for camera in camera_a camera_b camera_c; do
  camera-rig provision fixed \
    --config ".local/camera_rig/${camera}/configs/fixed_provision.yaml" \
    --output ".local/camera_rig/${camera}/provision"
  camera-rig provision validate \
    --artifact ".local/camera_rig/${camera}/provision"
done
```

脚本生成的 provision YAML 使用指向所选 target artifact 的可移植相对路径、实际 target
SHA-256 和 `target.detection_policy: pose_validated`。实体 coverage、残差或姿态稳定性 gate
失败后禁止复用旧 extrinsics，也禁止使用 `--force` 掩盖失败；`--force` 只用于明确替换一个
已存在且由 CameraRig 管理的 artifact，并且仍须通过完整验证。

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

先检查全部 route：

```bash
python scripts/prepare_ffs_assets.py --check --asset-root .local/ffs
```

若顶层 `passed` 为 `true`，资产已经完整，禁止紧接着再次运行 `--build-tensorrt`，也不要加
`--force`。只有全新主机尚未生成派生资产或明确决定完整重建时，才按第 6.1.3 节使用真实的
`.local/ffs/artifacts/model_best_bp2_serialize.pth`、`.local/ffs/artifacts/cfg.yaml` 和自动检测
的 `FFS_CUDA_ARCH` 构建；`path/to/...` 只是占位文本，不能原样执行。构建脚本默认拒绝覆盖
已有 ONNX、Engine 和 manifest，这是保护已有有效资产，不是构建故障。

依次 smoke `pytorch`、`tensorrt_single`、`tensorrt_two_stage`、
`tensorrt_plugin`；生产只使用 FP16 `tensorrt_plugin`。资产检查 PASS 后，自动生成四份
standalone smoke YAML 和 live RGB plugin YAML：

```bash
python scripts/prepare_ffs_pipeline_configs.py \
  --asset-root .local/ffs \
  --output-dir .local/configs \
  --camera-name camera_a
```

命令生成 `.local/configs/ffs_{pytorch,tensorrt_single,tensorrt_two_stage,tensorrt_plugin}.yaml`
以及 `.local/configs/ffs_tensorrt_plugin_rgb.yaml`，并以相对于声明 YAML 的 `../ffs/...`
路径绑定已经检查的资产。默认拒绝覆盖已有配置；只有明确要重新生成这五个私有 YAML 时才可
加 `--force`。四份 standalone smoke 配置保持 `pointcloud.use_rgb: false`，因为它们没有
权威 IR→color 外参；RGB 配置只供读取 provision bundle 的 live CameraRig 集成使用。

第 9.1 节已采集的同一新鲜 CameraRig snapshot 位于
`.local/captures/target-id/camera_a/frames/frame_000000.npz`。先确认输入存在，再运行四条
route；共享 offline loader 会把 snapshot 的 `ir_left`、`ir_right`、`color` 规范化为 PCB
canonical `left_ir`、`right_ir`、`rgb` 键：

```bash
test -f .local/captures/target-id/camera_a/frames/frame_000000.npz

for backend in pytorch tensorrt_single tensorrt_two_stage tensorrt_plugin; do
  if ! python scripts/run_ffs_stereo_frame.py \
      --config ".local/configs/ffs_${backend}.yaml" \
      --input .local/captures/target-id/camera_a/frames/frame_000000.npz \
      --output-dir ".local/evidence/ffs-${backend}" --no-show; then
    echo "${backend}: FFS smoke FAIL"
    break
  fi
done
```

这些 standalone 输出只证明对应模型 route 能在当前 GPU 上加载和推理，不构成标定、尺度
或实体几何验收。生成的 `.local/configs/ffs_tensorrt_plugin_rgb.yaml` 已设置
`pointcloud.use_rgb: true` 和 `output_format: xyzrgb`；live CameraRig 集成会从已验证的
provision bundle 提供权威 IR 内参、双目 baseline 和 IR→color 外参，不得在 standalone
smoke 配置中编造这些值。

## 12. 单相机 XYZ/XYZRGB

首次进入本节前，用已确认的 identity map、三份 passed provision 和第 11 节生成的 RGB
plugin 配置创建 workspace mapping 与 live rig 私有配置：

```bash
python scripts/prepare_live_reconstruction_configs.py \
  --identity-map .local/camera_rig/camera_identity_map.json \
  --camera-rig-root .local/camera_rig \
  --ffs-config .local/configs/ffs_tensorrt_plugin_rgb.yaml \
  --output-dir .local/configs \
  --expected-camera-count 3
```

它生成 `.local/configs/mapping.yaml`、`mapping_acceptance.yaml`、
`live_rig_ffs_rgb.yaml`、`live_rig_ffs_rgb_raw.yaml` 和
`live_rig_ffs_rgb_compact.yaml`。如果 workspace 专属的 `mapping.yaml` 已存在，默认保留它；
其余已有文件只有内容完全一致才接受，禁止用 `--force` 静默覆盖测量过的 crop、plane ROI 或
rig 配置。这里故意不写 `rig_calibration`：当前只使用三份 fixed provision；只有完成后续
multi-pose validation 与 promotion 后才能添加正式 deployment。

下面的命令打开 `camera_a`，用 production TensorRT plugin 从左右 IR 推理 300 帧 FFS
深度，使用 provision 中的内参、双目 baseline、IR→color 和 workspace 变换生成 XYZRGB，
检查标定板 `z=0` 平面、帧连续性、超时和 open/close/reopen lifecycle，并保存 PLY、PNG 和
JSON evidence。标定板必须继续固定在 provision 时的 workspace pose-0：

```bash
python tools/mapping/run_live_single_camera.py \
  --camera-config .local/camera_rig/camera_a/configs/runtime.yaml \
  --provision .local/camera_rig/camera_a/provision \
  --mapping-config .local/configs/mapping.yaml \
  --ffs-config .local/configs/ffs_tensorrt_plugin_rgb.yaml \
  --depth-source ffs_stereo --frames 300 \
  --output .local/evidence/camera_a \
  --report .local/reports/camera_a.json
```

`pointcloud.use_rgb: true` 输出 Nx6。落在 color imager 视场外的 depth 点使用显式黑色，
不会伪造颜色。该命令不会弹出交互窗口，也不需要连接实体显示器；Matplotlib 使用 `Agg`，
Open3D 使用 offscreen renderer 写 PNG。SSH/headless 主机仍需可用的 NVIDIA EGL/OpenGL
runtime；若出现 EGL/OpenGL context 错误，应修复 headless rendering 环境，不能把它误判成
相机或几何 PASS。

## 13. 多相机采集

下面的命令同时打开 `camera_a/b/c`，每台使用自己的 runtime、passed provision 和共享的
FFS plugin 配置。它采集并交付 1000 个按 `host_receive_timestamp_ns` 匹配的 frame sets，
然后重新打开全部相机再验证 60 帧。`capture_matching` scope 正式执行相机采集、时间匹配、
无帧复用以及 worker/buffer lifecycle gates；它会记录几何和 FFS 性能，但此时不把它们作为
正式通过条件，也不等价于后续 multi-pose 跨相机标定。

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
benchmark 解释由 CR18 负责。该命令只用 Matplotlib `Agg` 保存 PLY、PNG 和 JSON，不启动
Open3D/Rerun 交互查看器，不需要实体显示器；通过 SSH 可直接运行。后续明确带
`--viewer rerun --rerun-spawn` 或 `--interactive` 的命令才会启动交互 viewer。

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
  --input-mode live --rig-config .local/configs/live_rig_ffs_rgb_compact.yaml \
  --frames 60 --warmup 5 \
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

## 30. Existing 500x700 deployment board

现有实体部署板为 500 x 700 mm、5 x 7 squares、100 mm square、75 mm marker，权威
dictionary 为 `DICT_4X4_50`。这是 existing-board 流程，不是 generated-board 流程；公开
已知元数据位于 `configs/calibration/charuco_500x700_existing_board.yaml`。Dictionary
identity 已解决，但 `legacy_pattern`、`border_bits` 和 canonical `squares_x` / `squares_y`
orientation 仍必须由证据决定。相机图像中的实体旋转不改变 target geometry。CameraRig
视觉检测、marker layout、corner ID 与 geometry consistency gate 仍须全部通过，禁止放松。

## 31. Projection model

PCB 现在完整保留 CameraRig 的 frame、distortion model、coefficients 与显式
raw/rectified pixel geometry。Builder、FFS RGB mapping、calibration 与 diagnostic 共用
projection/deprojection API；不支持的 model direction 会 fail closed。librealsense 定量
parity、FFS 无 double-rectification 契约，以及 model parity 与实体 intrinsic accuracy 的
区别见 [`docs/projection-models.md`](docs/projection-models.md)。

## 32. Multi-pose N-camera calibration

CameraRig 仍严格只表示一台实体相机。PCB 在 `pointcloud_builder.rig_calibration` 中拥有
gauge-fixed robust multi-pose N-camera bundle adjustment、connected partial-visibility
graph、pose-diversity gate、solve/holdout validation 与显式 candidate-only export。Operator
流程、数学模型、artifact contract、synthetic acceptance 及 real 第三相机 deferred 状态见
[`docs/multi-pose-multi-camera-calibration.md`](docs/multi-pose-multi-camera-calibration.md)。

## 33. Calibration lifecycle and production precedence

```text
CameraRig fixed provision（仅作 initializer）
-> PCB multi-pose candidate
-> 精确 2D solve + holdout validation
-> candidate-only live preview
-> 实体 N-camera pairwise acceptance
-> promoted PCB rig-calibration deployment
-> snapshot / recording / TSDF / Rerun production outputs
```

Rig YAML 中的 `rig_calibration` 可选。省略时保持旧 CameraRig fixed-provision workspace
geometry；配置后 PCB deployment 对 `T_workspace_from_camera` 具有最高优先级，identity、
CameraBundle hash、camera set、frame 或 fingerprint 任一不匹配都直接失败，绝不 fallback。
CameraRig 继续负责 K/D、depth scale、stream frame、device identity 与内部 stream
extrinsics。FFS runtime 组合：
`T_workspace_from_ir_left = T_workspace_from_color(deployed) @ T_color_from_ir_left`。
Candidate viewer 拒绝已配置 production deployment 的 rig，且始终报告
`candidate_only=true`、`production_applied=false`。

Promotion 必须精确绑定 solution、validation、通过的 holdout 与真实实体 3D acceptance；
只有 reprojection PASS 不够：

```bash
python tools/calibration/promote_rig_calibration.py \
  --solution .local/calibration/new-workspace/solution.json \
  --validation .local/calibration/new-workspace/validation.json \
  --physical-acceptance .local/calibration/new-workspace/physical_acceptance.json \
  --output .local/calibration/deployment/new-workspace/rig_calibration.json
```

Recording 和 TSDF map artifact 保存 calibration mode、deployment/solution fingerprint、
camera set、workspace frame 与逐相机 CameraBundle hash。`--initial-map` 的 deployment
fingerprint 不同会 fail closed；不存在隐式 map migration。

## 34. Existing-board identification and registration

第 9.1 节是当前 500 x 700 mm `DICT_4X4_50` 实体板的唯一规范可执行流程，已经包含三相机
证据采集、identify、register 和 artifact 自带校验和验证；这里不再复制命令或另设私有路径。
若 CameraRig 对 `legacy_pattern`、`border_bits` 或 orientation 仍报告 ambiguity，必须用视觉
equivalence 或真实权威 source 文件解决后再继续。不得把描述性标签传给
`--authoritative-source`，禁止扫描 dictionary capacity，也禁止替换成 `DICT_4X4_100`。

## 35. Generic N-camera acceptance

Production acceptance 层按 `N choose 2` 自动枚举全部无序 pair，不限制 camera name。报告
overlap、symmetric NN、board/interior、plane、diagnostic-only residual SE(3)、逐相机贡献、
overlap graph、fused count、可测 surface thickness，以及 matcher/drop statistics。
Diagnostic ICP 永不写回 deployment extrinsics。

```bash
python tools/calibration/evaluate_ncamera_rig_alignment.py \
  --rig-config .local/configs/live_rig_three_camera_candidate.yaml \
  --candidate-solution .local/calibration/new-workspace/solution.json \
  --candidate-validation .local/calibration/new-workspace/validation.json \
  --thresholds configs/calibration/ncamera_physical_acceptance_strict_example.yaml \
  --recording .local/recordings/new-workspace-physical-acceptance \
  --mapping-config .local/configs/new-workspace-mapping.yaml \
  --matched-sets 5 \
  --output .local/reports/new-workspace-ncamera-acceptance.json
```

Candidate mode 输出 promotion 所消费的正式 physical-acceptance artifact。必须在查看新数据
前 preregister/freeze threshold 文件。Promotion 后，用 `--rig-calibration` 替换三个
candidate 参数，即可 regression-test deployed path；此时复用精确 accepted physical receipt
中的 thresholds。

优先要求 A-B、A-C、B-C 全部 PASS。只有在实体上确实无重叠且有说明时，才可使用
`--declared-no-overlap camera_a:camera_c`；剩余 accepted-overlap graph 仍须 connected。
该 pair 报 `NOT_APPLICABLE_NO_OVERLAP`，不能编造 NN metrics。

## 36. Three-camera configuration and USB/FFS readiness

从 `configs/mapping/live_rig_three_camera_example.yaml` 开始。Public 文件不得含 serial。
每台相机需要私有 CameraRig runtime YAML 与 provision artifact；相同且已验证的 FFS profile
可以共用一个 FFS pipeline YAML。三条链路都必须枚举为 USB 3.x。检查 root hub，并在可行
时把带宽分散到不同 controller/root hub，但不得硬编码主板 topology。
Candidate preview/acceptance 阶段复制 example 后须省略 `rig_calibration`；只有 promotion
完成后才添加该 section，并指向精确 deployed artifact。

```bash
python scripts/doctor_reconstruction_env.py --no-hardware
python scripts/doctor_reconstruction_env.py \
  --expected-d435i-count 3 --asset-root .local/ffs
python tools/mapping/check_usb_topology.py \
  --identity-map .local/camera_rig/camera_identity_map.json \
  --expected-count 3 --report .local/reports/usb-topology.json
```

N-camera 功能是 generic，但不能从双相机 FPS 推断三相机吞吐。必须在最终 GPU/USB topology
上重新测 capture FPS、match ratio、p50/p95 processing、GPU memory、RSS、viewer overhead
与 TSDF mapper overhead。

## 37. New machine / new workspace checklist

严格按以下顺序执行；第 12 步后相机保持固定，任一 gate 失败就停止。

1. 使用 `--recurse-submodules` clone，并更新 submodule。
2. 创建隔离的 `pcb-reconstruction` 环境。
3. 使用 `--no-hardware` 运行 doctor。
4. 准备或重建 FFS TensorRT-plugin assets。
5. 连接三台 D435i。
6. 发现 identity，不把 serial 写入 public 文件。
7. 分配私有逻辑名 `camera_a`、`camera_b`、`camera_c`。
8. 用 `--expected-count 3` 验证 USB topology。
9. 为每台相机创建私有 CameraRig runtime YAML。
10. 从 500 x 700 `DICT_4X4_50` 现有板采集 identity 证据。
11. 使用第 34 节命令 identify/register existing board。
12. 把板精确 fixture 在 canonical workspace `pose_0`。
13. 对每台相机运行 CameraRig `pose_validated` target preflight。
14. 在所有相机充分看到 `pose_0` 时创建逐相机 initial fixed provision。
15. 验证每份 provision。
16. 从 public example 创建私有三相机 rig config。
17. 若 profile/model 有变化，运行 projection-parity smoke。
18. 采集约 24-30 个 diverse multi-pose target pose。
19. 预先指定最后约 4-6 个 pose 为 holdout。
20. 求解 PCB N-camera bundle adjustment。
21. 用保留 holdout 验证精确 candidate。
22. 运行 candidate-only live preview。
23. 记录并通过 generic N-camera pairwise physical acceptance。
24. 把精确 candidate 与 acceptance receipts promote 到 production。
25. 运行 raw RGB reconstruction。
26. 在 sampling off、2.5 mm voxel 下运行 dense XYZRGB fusion。
27. 测量并配置新的 workspace crop。
28. Benchmark 真实三相机性能及 viewer/mapper overhead。
29. 记录新的 fingerprint-bound depth data。
30. 构建新的 fingerprint-bound TSDF map。
31. 验证、保存并重新加载 map。
32. 在 production mode 下运行 interactive Rerun。

Pose-0 board 定义 `T_workspace_from_target,pose0 = I`，即 workspace origin 与
+X/+Y/+Z。应使用 mechanical stop、fixture、tape 或测量基准，不依赖目测。Initial
provision 完成后，只移动 pose 1..M 的板，绝不移动相机。后续 pose 可为 A+B、B+C、A+C、
A+B+C 等 partial visibility，但完整 camera-pose graph 必须 connected。

## 38. Invalidation rules and deployment status

相机移动、mount 松动、camera set 改变、workspace pose-0 改变、实体 target geometry 改变、
CameraBundle 改变或 intrinsics/profile 改变时必须重标定。新的实体 workspace 必须重新生成
provision、observation、solution、validation、physical acceptance、production deployment、
workspace crop、recording 与 TSDF map。旧 artifact 不能静默成为新 workspace 的 production
artifact。

当前状态：N-camera implementation 已对 2/3/4 cameras `VALIDATED_SYNTHETICALLY`；真实
dual-camera multi-pose production 为 `VALIDATED`；real three-camera calibration/
reconstruction 在 camera C 和新 workspace 可用前为 `DEFERRED`。Large-board metadata 为
`RESOLVED`；real registration 为 `DEFERRED_TO_NEW_WORKSPACE`。
