# Deterministic workspace fusion

M6 fuses one matched `RigFrameSet` and does not accumulate state across time. The exact
stage order is per-camera deprojection, per-camera local crop, workspace transform,
workspace crop, canonical concatenation, voxel fusion, then one global sampling step.
Per-camera samplers are disabled, and the provenance sidecar proves that fusion input
count equals the sum of per-camera pre-sampling counts.

Voxel keys use:

```text
floor((xyz - origin) / voxel_size_m)
```

This definition handles negative coordinates. Keys and point values are canonically
sorted before aggregation. Each voxel emits the arithmetic XYZ centroid and, for Nx6
input, arithmetic RGB mean. Deterministic sampling uses the configured fixed seed.

The sidecar reports total input points, output voxels, per-camera input and unique-voxel
counts, multi-camera voxel count, and aligned per-voxel source-camera and point counts.
Point tensors do not embed camera IDs. `RigBuildResult` exposes per-camera camera-frame
and workspace clouds plus concatenated, workspace-cropped, fused, and sampled stages.

Supported boundary cases include a one-camera fallback, empty individual or all-empty
clouds, unequal point counts, Nx3/Nx6, negative workspace coordinates, CPU/CUDA, and
camera-order permutations. A zero-camera rig is rejected by configuration.
