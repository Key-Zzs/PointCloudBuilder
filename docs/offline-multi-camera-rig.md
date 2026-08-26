# Offline multi-camera rig orchestration

The PCB-owned `pointcloud-builder.rig.v1` contract lists independent single-camera
CameraRig replay/provision artifacts. CameraRig remains responsible for each camera's
pixels and calibration; PCB matches frames, runs each native or FFS builder, transforms
every cloud into the declared workspace, and concatenates in canonical camera-name
order. M5 does not voxel-fuse the result.

The loader rejects unknown fields, duplicate or empty camera lists, missing runtime
bundles, frame-count mismatches, camera identity mismatches, and bundle parent frames
that differ from `output_frame`. `exact_index` matches artifact indices.
`nearest_host_timestamp` compares only host receive timestamps, reports the reference
camera, signed per-camera deltas, maximum skew, and unmatched cameras; sensor clocks
from different physical cameras are never assumed comparable.

Per-camera sampling is disabled inside the rig. The M5 concatenated cloud can use one
deterministic global sampler, while metadata retains every pre-sampling count for the
M6 no-early-sampling regression contract.

`tools/mapping/run_synthetic_rig.py` analytically ray-renders a plane and cuboid from
three distinct pinhole poses, validates 1/2/3-camera operation and YAML order
invariance, and emits a red/blue/green workspace overlay. Its synthetic identities and
paths contain no hardware or private calibration data.
