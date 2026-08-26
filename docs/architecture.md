# Mapping architecture

PointCloudBuilder keeps policy observation and persistent geometry separate after one
matched camera set. `RigFrameProcessor` performs each native/FFS inference once. The
same resolved depth feeds both deprojection and `RigDepthFrameSet`; TSDF never consumes
`concatenated`, `fused`, or globally sampled point clouds.

```text
CameraRig worker A/B -> host-time matcher -> RigFrameProcessor
                                           |-> current workspace clouds -> fusion -> policy sample
                                           |-> CPU RigDepthFrameSet -> latest-only TSDF process
current result -> bounded packet -> latest-only Rerun process
```

Camera workers own device sessions. The mapping process owns CUDA FFS and snapshot
fusion. Optional Rerun and Open3D imports occur only in their spawned child processes.
Both IPC queues have capacity one or two, evict the oldest item, publish drop/RSS
telemetry, and have finite close/terminate behavior.
Freeze, unfreeze, and reset are queue barriers: pre-command depth packets are discarded
before acknowledgment, so stale observations cannot cross a lifecycle transition.

Frames use column vectors and `T_target_from_source`. CameraRig provides
`T_workspace_from_camera`; Open3D receives the tested world-to-camera inverse. Rerun
uses `ParentFromChild` for camera pose and cancels that parent transform for per-camera
clouds already expressed in workspace coordinates.

Real artifacts are never package inputs. Private camera YAML, provisions, captures,
recordings, maps, screenshots, reports, and RRD files stay below ignored `.local/`.

See the existing [CameraRig integration](camera-rig-integration.md),
[live acquisition](live-rig-acquisition.md), and [workspace fusion](workspace-fusion.md)
guides for the snapshot side of the boundary.
