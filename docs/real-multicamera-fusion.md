# Real multi-camera snapshot fusion

Real-scene acceptance consumes an M7 matched `RigFrameSet` through the existing M6
`RigFrameProcessor`. Per-camera sampling remains disabled. Each camera is deprojected
and transformed independently, the configured workspace crop is applied, deterministic
5 mm voxel fusion operates on the current matched snapshot, and one 4096-point global
sampler runs after fusion. No stage retains a map or accumulates clouds across time.

`pointcloud_builder.fusion.real_metrics` provides target-agnostic board, overlap,
contribution, and supported-cube measurements. Board statistics use the interior
ChArUco region. Cross-camera consistency voxelizes each camera independently to 5 mm
and reports both nearest-neighbor directions plus symmetric median, p95, and RMSE.
Contribution distinguishes point share from the fraction of output voxels touched by
each camera.

Cube extraction removes the fitted support-plane band, keeps the 10--120 mm height
range, creates 26-neighbour components on a 5 mm voxel grid, and scores cube-like
components against a nominal 70 mm side. All robust extents use one centroid per
occupied voxel so image-density differences cannot bias the result. XY size uses a yaw
search with a robust 5--95% rectangle. Height is exactly `p99(z) - p01(z)`, and support
gap is `p01(z)` from the workspace plane. Length/width retain XY semantics and height
remains the third dimension. An unresolved close-score candidate fails closed after
writing visual evidence instead of silently selecting among tied objects.
The same-object gate compares every route's center with the fused candidate. Joint
observation counts 5 mm occupied voxels inside the fused oriented box only after each
camera's resolved plane-removal threshold, so board points cannot satisfy the gate.

Interference controls compare camera-a-only, camera-b-only, and simultaneous capture
without changing emitter, laser, exposure, gain, firmware, or hardware-sync settings.
A valid-ratio drop over 0.10 or a board-p95 increase over 2x is advisory. A simultaneous
valid ratio below 0.50 or board p95 above 40 mm is a blocking gross failure.

One formal run is executed with:

```bash
python tools/mapping/run_live_rig_fusion.py \
  --rig-config path/to/private/ffs_fusion_rig.yaml \
  --mapping-config path/to/private/mapping.yaml \
  --matched-sets 300 \
  --output path/to/private/formal_run \
  --report path/to/private/formal_run.json
```

Two independent invocations reopen both CameraRig sessions. The aggregate tool compares
cube dimensions, board residual, overlap residual, and worker failures while also
evaluating native-depth and FFS interference controls. Each run keeps five independent
snapshot results distributed across its 300 matched sets; clouds are never accumulated
across snapshots. The aggregate also binds the already-passed M7 dual-camera FFS report,
including its 15 Hz and 66.8 ms performance gates. Input schema, camera aliases, depth
routes, frame counts, and distinct A/B report paths are validated before aggregation.
Reports and all real PLY/PNG evidence must remain in ignored private storage.
