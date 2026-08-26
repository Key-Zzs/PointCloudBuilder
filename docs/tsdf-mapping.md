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

Native observations retain raw `uint16` and device meters-per-unit. Open3D divides by
its `depth_scale`, so the backend passes the reciprocal. FFS observations are
rectified float meters, invalid=0, scale=1, in the left-IR optical frame. Non-identity
unrectified distortion is rejected. The Open3D extrinsic is explicitly
`inverse(T_workspace_from_camera)`; synthetic tests show the opposite direction fails.

Recording and offline reconstruction:

```bash
python tools/mapping/record_live_rig_depth.py \
  --config .local/configs/live_rig.yaml --matched-sets 300 \
  --depth-source native --output .local/recordings/native

python tools/mapping/build_tsdf_offline.py \
  --recording .local/recordings/native \
  --config configs/mapping/tsdf_example.yaml \
  --output .local/maps/native_static
```

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
