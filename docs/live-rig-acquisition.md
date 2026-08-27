# Live rig acquisition

`camera_rig_live` extends the strict `pointcloud-builder.rig.v1` source union. Each
enabled camera names one private CameraRig runtime YAML and one fixed provision
bundle. The loader verifies camera name, serial equality, distinct device identities,
`copy_frames=true`, and a common fixed-calibration parent frame before acquisition.
Private device identities and calibration artifacts must remain outside Git.

Each camera session is created, opened, captured, and closed by its own non-daemon
worker thread. Workers publish copied `CameraFrame` objects into capacity-limited
latest-biased buffers. A shared condition makes multi-buffer snapshot and consumption
atomic. Overflow explicitly evicts the oldest frame; shutdown and error propagation
wake all bounded waits and join every worker.

The matcher uses `time.monotonic_ns()` host-receive timestamps only. It does not compare
device clocks or cross-device frame numbers. The reference camera's oldest buffered
frame is paired with the nearest bracketing candidate from every other camera under
`maximum_skew_ms`. A selected frame and every older frame in that camera's buffer are
consumed together, so a frame cannot be reused. Matcher waits, buffers, skew summaries,
and worker telemetry are bounded scalar state and never retain an unbounded frame or
tensor history.

After matching, `RigFrameProcessor` applies the same M6 path to each camera in canonical
name order: per-camera native or FFS deprojection, fixed transform to workspace,
workspace crop, optional deterministic snapshot voxel fusion, and one global sampling
step. Fusion is current-frame-set only; live acquisition does not create a persistent
map, TSDF, temporal voxel state, or cross-time accumulated cloud.

`configs/mapping/live_rig_example.yaml` documents the portable public schema. Each
camera may opt into RGB point features with `pointcloud.use_rgb: true`. Native depth
and FFS left-IR depth are projected into the calibrated color camera with the CameraRig
bundle's intrinsics and `T_color_from_depth`; color remains attached through workspace
transforms, crop, voxel fusion, and global sampling. The default is `false` for
backward-compatible XYZ output. RGB mode requires the color stream and fails closed if
it is absent.

A private hardware run can be accepted with:

```bash
python tools/mapping/run_live_rig.py \
  --rig-config path/to/private/live_rig.yaml \
  --mapping-config path/to/private/mapping.yaml \
  --frames 1000 \
  --reopen-frames 60 \
  --output path/to/private/output \
  --report path/to/private/report.json
```

The tool produces a private structured report, capture/match/latency timelines, and
bounded best/middle-sequence/worst workspace snapshots. FFS throughput and end-to-end
latency gates are evaluated without changing the selected backend.
