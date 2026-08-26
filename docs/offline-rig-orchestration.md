# Offline rig orchestration

`pointcloud-builder.rig.v1` is a strict PCB-owned list of single-camera sources. Each
entry names its CameraRig replay and provision artifacts, native or FFS depth mode,
pipeline configuration, local crop, and enabled state. CameraRig itself remains a
single-camera library pinned at `v1.0.0`; PCB owns cross-camera matching and workspace
composition.

The loader rejects unknown fields, empty or duplicate camera lists, missing runtime
bundles, camera/frame identity mismatches, and `output_frame` values that differ from a
bundle's fixed parent frame. `exact_index` matches equal artifact indices.
`nearest_host_timestamp` uses host-receive timestamps only, reports the reference
camera, signed per-camera deltas, maximum skew and unmatched cameras, and never assumes
that sensor clocks from different physical cameras are comparable.

Each camera independently follows CameraFrame to native/FFS depth, local camera crop,
and workspace transform. Results are canonically ordered by camera name, so reversing
YAML order does not exchange camera identity or change concatenated metadata. One,
two, and three-camera rigs share the same contract.
