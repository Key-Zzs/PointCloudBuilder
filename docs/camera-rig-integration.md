# CameraRig v1 integration

PointCloudBuilder pins CameraRig `v1.0.0` as `third_party/CameraRig`. Consumer code
imports only `camera_rig.api`; PointCloudBuilder's core import remains independent of
CameraRig.

`CameraRigFrameAdapter` binds every CameraFrame to the CameraBundle's camera name and
serial, then maps one synchronized CameraFrame to raw `rgb`, `depth`,
`left_ir`, and `right_ir` arrays plus per-stream timing metadata. It does not align,
filter, recolor, or rescale the arrays. Color is RGB, native depth remains raw `uint16`,
and IR remains `uint8`.

`calibration_from_camera_bundle` accepts only a passed fixed-mount CameraBundle. It
keeps every transform as `T_target_from_source`, preserves source and target frame
names, and resolves inverse or multi-hop paths deterministically. Duplicate edges,
conflicting alternate paths/cycles, disconnected geometry, or reversed use fails closed.

`create_native_builder` derives depth scale, depth/color intrinsics, internal
extrinsics, and the workspace transform from the bundle. Native XYZ is defined in the
depth optical frame. `create_ffs_builder` derives left/right IR intrinsics,
distortion, `T_ir_right_from_ir_left`, positive baseline norm,
`T_color_from_ir_left`, and `T_workspace_from_ir_left` from the same bundle. The
strict identity-rectified gate remains authoritative and no YAML calibration value can
override it. FFS XYZ is defined in the left-IR optical frame. RGB projection remains
disabled for M1-M6 acceptance.
