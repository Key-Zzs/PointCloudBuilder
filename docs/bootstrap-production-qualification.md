# Bootstrap and production qualification

The fixed three-camera route has two deliberately different authority levels.

`BOOTSTRAP_QUALIFIED` belongs to one CameraRig CameraBundle. It requires genuine A4
target metrology, passed raw/detection/physical-PnP/uncertainty/repeatability checks,
and independent native metric-depth plane, normal, and scale evidence. The bundle
retains factory K/D, internal stream extrinsics, depth scale, device identity, and a
single-pose workspace initializer. Its authority says `bootstrap_only` and
`production_authoritative=false`.

`PRODUCTION_QUALIFIED` belongs only to a PCB rig deployment. Before promotion, PCB
requires the exact three bootstrap authorities, a preregistered 30-pose capture with
six untouched holdouts, passed per-camera factory-vs-diagnostic-refit intrinsic health,
passed joint solution and holdout validation, and passed 3-D physical acceptance for
all required A-B, A-C, and B-C overlaps. Every receipt is hash-bound. A missing or
mismatched receipt fails closed.

The structured residual is retained only as a diagnostic. Its universal hard-gate
status is `NOT_SUPPORTED_DUE_TO_PLANAR_IDENTIFIABILITY_LIMIT`; it can neither qualify
a bootstrap nor override target metrology, native depth, holdout, or physical 3-D
evidence.

Without a configured, validated deployment, PCB reports `production_applied=false`.
Only the explicit candidate physical-acceptance recording route may consume bootstrap
geometry before promotion. Production recording and mapping must reject it.

## Next stage: dual wrist cameras (design only)

After fixed-rig promotion, use deployed `T_workspace_from_camera_a/b/c` as the world
anchor. For each arm, estimate `T_workspace_from_base` and
`T_ee_from_wrist_camera`; robot FK supplies `T_base_from_ee(q)`, and target detection
supplies `T_wrist_camera_from_target`. The governing equation is:

```text
T_workspace_from_base @ T_base_from_ee(q) @ T_ee_from_wrist_camera
= T_workspace_from_target @ inverse(T_wrist_camera_from_target)
```

Dual-arm base installation error must be estimated, not assumed from CAD. This stage
does not implement eye-in-hand, robot-world/hand-eye, or dual-arm joint bundle
adjustment.

The deferred frame set is explicit: workspace/world `W`; left/right robot bases
`B_L`, `B_R`; left/right end effectors `E_L`, `E_R`; and left/right wrist-camera
optical frames `C_wL`, `C_wR`. The future arm-specific unknowns are
`T_W_from_B_L`, `T_W_from_B_R`, `T_E_L_from_C_wL`, and `T_E_R_from_C_wR`;
robot-owned inputs are only the FK transforms `T_B_L_from_E_L(q_L)` and
`T_B_R_from_E_R(q_R)`. No wrist transform is estimated or published in this current stage.
