# Rerun visualization

Install only the official distribution verified for Python 3.10:

```bash
python -m pip install -e ".[rerun]"  # rerun-sdk==0.36.3
```

The mapping process does not import the SDK. It copies 320-pixel-wide bounded CPU
NumPy previews, poses, intrinsics, scalar metrics, and 20k-50k selected points into a strict
`VisualizationPacket`. Selection happens on the tensor's source device before CPU
transfer. A spawned logger owns the SDK and a latest-only queue.

```bash
python tools/mapping/run_live_rig_fusion.py ... \
  --viewer rerun --rerun-spawn --viewer-point-budget 30000

python tools/mapping/run_live_rig_fusion.py ... \
  --viewer rerun --rerun-record .local/rerun/session.rrd
```

`--rerun-connect URL`, spawn, recording-only, and viewer-plus-file tee are supported.
Rerun 0.36 `set_sinks` is used for simultaneous gRPC and `FileSink`; sequential
`save`/`connect` calls would overwrite sinks. Files must end in `.rrd`.

Stable entities include `/world/workspace_axes`, `/world/target`, both `/rig/camera_*`
RGB/frustum/cloud trees, snapshot cloud stages, `/map/tsdf_points_raw`,
`/map/tsdf_points_cropped`, `/map/tsdf_points_sampled`, `/map/tsdf_mesh`, dynamic overlay,
raycast depth, dynamic mask, and `/metrics/*`. Timelines are `matched_set_index` and
`host_time_seconds`. Static map geometry is logged only when its map revision changes;
the current dynamic overlay remains time-varying. If latest-only eviction removes the
first packet for a static revision, its bounded geometry is carried into the replacement
packet until the child consumes it.

The three point entities can be toggled independently in the UI. The mesh remains the
full extracted TSDF mesh; toggling a cropped point entity does not imply mesh cropping.
Visual acceptance compares raw/cropped/sampled shape, crop bounds, handedness, and
metric scale while leaving private screenshots and `.rrd` recordings under `.local/`.

Telemetry reports produced/dropped/logged packets, maximum queue depth, child peak RSS,
child error, and producer p95. Acceptance is p95 producer overhead at most 2 ms or no
more than 5% processed-FPS loss. Rerun 0.36 can render deterministic evidence from an
`.rrd` without a desktop window using `rerun --headless --screenshot-to`; interactive
GUI review remains available when camera/view selection needs manual adjustment.
