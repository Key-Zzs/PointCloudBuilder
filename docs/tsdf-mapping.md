# TSDF mapping

Install the validated backend without changing Torch/CUDA/TensorRT/OpenCV:

```bash
python -m pip install -e ".[tsdf]"  # open3d==0.19.0
```

`configs/mapping/tsdf_example.yaml` uses schema `pointcloud-builder.tsdf.v1`.
Geometry-only Open3D integration still allocates `Float32[1] tsdf`, `UInt16[1]
weight`, and dummy `UInt16[3] color`. With 50,000 blocks at resolution 16, attributes
alone are 2,457,600,000 bytes before hash-map/runtime overhead; size the deployment
host accordingly.

Production persistent mapping uses `integration.source: ffs_stereo`; the recommended
backend is the validated TensorRT plugin. Missing FFS artifacts are fatal and never
select native depth implicitly. Native TSDF is retained only as an optional diagnostic
baseline and may report `DEGRADED_GEOMETRY` without changing production status.

Native observations retain raw `uint16` and device meters-per-unit. Open3D divides by
its `depth_scale`, so the backend passes the reciprocal. FFS observations are
rectified float meters, invalid=0, scale=1, in the left-IR optical frame. Non-identity
unrectified distortion is rejected. The Open3D extrinsic is explicitly
`inverse(T_workspace_from_camera)`; synthetic tests show the opposite direction fails.

Recording and offline reconstruction:

```bash
python tools/mapping/record_live_rig_depth.py \
  --config .local/configs/live_rig.yaml --matched-sets 300 \
  --depth-source ffs_stereo --output .local/recordings/ffs_production

python tools/mapping/build_tsdf_offline.py \
  --recording .local/recordings/ffs_production \
  --config configs/mapping/tsdf_example.yaml \
  --output .local/maps/ffs_static
```

An FFS recording is publishable only when its checksummed manifest binds every camera
to `tensorrt_plugin`, precision, artifact ID, and pipeline-config SHA. Production
acceptance verifies that the map source receipt matches that manifest. `--native-map`
is optional and reports `NOT_RUN` when omitted; it never blocks the FFS verdict.

Postprocessing occurs only after the volume extracts a raw workspace point cloud:

```text
VoxelBlockGrid -> raw_point_cloud -> crop_workspace_cloud -> sample_point_cloud
               -> full extracted mesh (independent; not cropped)
```

The optional `postprocess.crop` and `postprocess.sampling` sections reuse the same
`CropConfig`, `SamplingConfig`, workspace crop, and sampler as current snapshots.
Without `postprocess`, both stages are disabled. Artifacts expose `point_cloud_raw.ply`,
`point_cloud_cropped.ply`, and `point_cloud_sampled.ply`; empty and undersized clouds
follow the existing repeat/zero padding rules.

Timing is split into coordinate/block activation, volume integration, point extraction,
mesh extraction, post-crop, and post-sampling. `raw_to_tsdf_update_ms` ends at the map
update; `map_to_sampled_cloud_ms` starts from an existing map. Mesh time is excluded
from all point-cloud totals.

For live publication, first produce a passing snapshot-only report with the same rig
config and matched-set count, then supply it to the asynchronous mapper:

```bash
python tools/mapping/run_live_tsdf_mapping.py \
  --rig-config .local/configs/live_rig.yaml \
  --tsdf-config .local/configs/tsdf_build.yaml \
  --matched-sets 300 \
  --snapshot-baseline-report .local/reports/snapshot_only.json \
  --recording-output .local/recordings/live_source \
  --map-output .local/maps/live_static \
  --report .local/reports/live_tsdf.json
```

For a cold `build_static` acceptance, `--build-warmup-sets N` records and integrates
an explicit prebuild interval, drains it through a lifecycle barrier, and then starts
the unchanged N-set FPS/p95/RSS window. Both counts remain in the same checksummed
source recording. This prevents normal empty-map block allocation from being mislabeled
as a steady-state leak. Frame-stride eligibility is applied before the wall-clock
update limiter, so the two controls cannot accidentally multiply and starve updates.
When full mesh plus deterministic FPS extraction costs hundreds of milliseconds,
keep `maximum_mesh_hz` below the update rate; the production example uses 0.2 Hz.

Map publication fails closed if acquisition is unclean, the mapper child fails or
integrates nothing, FPS loss exceeds 10%, or end-to-end snapshot p95 grows by more
than 5 ms. It also fails if fewer than 32 child-RSS samples exist or, after discarding
the first 20% as warmup, RSS has either more than 256 MiB quartile-median growth or a
fitted slope above 5 MiB per 100 frames.

Both the snapshot baseline and live report retain p50/p95 summaries for frame-match
wait and snapshot processing separately. These are diagnostic decompositions of the
unchanged end-to-end latency gate, not alternative ways to pass it.

In `frozen_static`, the mapper loads and extracts the initial map before acquisition,
then receives no per-frame depth packets: the map is read-only and the current fused
cloud remains the separate dynamic overlay. Child RSS is still sampled once per
matched set. `build_static` and `guarded_continuous` continue to receive the original
per-camera depth observations through the bounded latest-only queue.

Recordings and map artifacts use temporary sibling directories, atomic rename,
canonical relative paths, exact file-set validation, and SHA-256 receipts. A map holds
the native `volume.npz`, point/mesh PLY, resolved config, metrics, source receipt, and
screenshots directory. Publication reloads the volume and checks active blocks,
attribute shapes and TSDF/weight statistics, geometry counts, and sampled symmetric
distance.

Lifecycle is create, integrate, freeze, explicit unfreeze, save/load, extract, full
reset, close. Open3D 0.19 has no validated atomic AABB weight reset, so local AABB
invalidation raises `FeatureNotSupportedError` instead of pretending to succeed.

Open3D 0.19 cannot reload a native VoxelBlockGrid with zero allocated keys. Publishing
a fully reset map therefore allocates one documented origin block whose weights remain
all zero. It extracts no points or triangles (`active_weight_voxels = 0`) and is a
serialization sentinel, not retained scene geometry.
