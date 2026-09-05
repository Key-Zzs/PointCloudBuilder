# Bootstrap 与 production 资格

固定三相机路线明确分成两个不同权限等级。

`BOOTSTRAP_QUALIFIED` 属于单台相机的 CameraRig CameraBundle。它要求真实 A4 target
metrology、通过 raw/detection/物理 PnP/uncertainty/repeatability，并有独立 native metric
depth 的平面、法向和尺度证据。Bundle 保留 factory K/D、内部 stream 外参、depth scale、
设备身份和单姿态 workspace initializer；其权限必须是 `bootstrap_only` 且
`production_authoritative=false`。

`PRODUCTION_QUALIFIED` 只属于 PCB rig deployment。Promotion 前必须精确绑定三台相机的
bootstrap authority、预注册的 30 姿态（6 个未参与训练的 holdout）、每相机
factory-vs-diagnostic-refit intrinsic health PASS、联合求解与 holdout PASS，以及全部必需
A-B、A-C、B-C overlap 的三维物理验收 PASS。所有回执均用 hash 绑定；缺失或不匹配立即
fail closed。

Structured residual 只保留为 diagnostic，通用 hard-gate 状态固定为
`NOT_SUPPORTED_DUE_TO_PLANAR_IDENTIFIABILITY_LIMIT`。它不能授予 bootstrap 资格，也不能
覆盖 target metrology、native depth、holdout 或三维物理证据。

未配置并验证 deployment 时，PCB 明确报告 `production_applied=false`。Promotion 前只有显式
candidate physical-acceptance recording 路线可使用 bootstrap geometry；production recording
与 mapping 必须拒绝该状态。

## 下一阶段：双腕相机（仅设计）

固定 rig promotion 后，以部署的 `T_workspace_from_camera_a/b/c` 作为 world anchor。每条手臂
估计 `T_workspace_from_base` 与 `T_ee_from_wrist_camera`；机器人 FK 提供
`T_base_from_ee(q)`，target observation 提供 `T_wrist_camera_from_target`。核心方程为：

```text
T_workspace_from_base @ T_base_from_ee(q) @ T_ee_from_wrist_camera
= T_workspace_from_target @ inverse(T_wrist_camera_from_target)
```

双臂 baselink 安装误差必须估计，不能直接采用 CAD 假设。本阶段不实现 eye-in-hand、
robot-world/hand-eye 或 dual-arm joint BA。

延后阶段的 frame 集合明确为：workspace/world `W`，左右机器人基座 `B_L`、`B_R`，左右
末端 `E_L`、`E_R`，以及左右腕部相机光学 frame `C_wL`、`C_wR`。未来阶段的未知量为
`T_W_from_B_L`、`T_W_from_B_R`、`T_E_L_from_C_wL` 与 `T_E_R_from_C_wR`；机器人侧权威
输入仅为 FK 变换 `T_B_L_from_E_L(q_L)` 与 `T_B_R_from_E_R(q_R)`。当前阶段不得估计或
发布任何 wrist transform。
