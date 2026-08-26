# Workspace geometry

PointCloudBuilder represents deprojected clouds as `FramedPointCloud`. Native raw depth
is deprojected in the bundle's depth optical frame using the bundle's depth scale and
depth intrinsics. `transform_point_cloud` requires a frame-matching
`T_target_from_source` and evaluates `xyz @ R.T + t` on the tensor's existing device;
for XYZRGB tensors, only XYZ changes.

`SingleCameraWorkspacePipeline` exposes `camera_raw`, `camera_cropped`,
`camera_sampled`, `workspace_raw`, `workspace_cropped`, and `workspace_sampled`.
Workspace crop occurs after the source-to-workspace transform. M2 permits the legacy
single-camera sampling stage; multi-camera fusion disables per-camera sampling and
samples globally after fusion.

`ExpectedPlaneRegion` is target-independent. It selects a workspace XYZ region and
reports point count, signed Z bias, median/P95 absolute Z, RMSE, fitted normal, and
normal-angle error. Calibration-target detection is not a runtime dependency.
