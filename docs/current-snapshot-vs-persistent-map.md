# Current snapshot versus persistent map

These products answer different questions and must not be substituted for one another.

| Property | Current snapshot | Persistent TSDF |
|---|---|---|
| Input | Current per-camera point clouds | Per-camera depth pixels, K, pose |
| Time | One matched set | Many matched sets |
| Output | Fused/current 4096-point policy tensor | Static mesh/dense points/volume |
| Dynamic objects | Preserved | Excluded or guarded |
| Update owner | Mapping main process | Independent Open3D process |

The snapshot sequence is per-camera deprojection, workspace transform, crop, voxel
fusion, then one global sampler. It remains authoritative for robot-policy input and
the current dynamic overlay.

The map sequence integrates each camera separately. It preserves each pixel ray,
metric depth, stream intrinsics, and `T_workspace_from_camera`. It never accepts an
already fused or globally sampled point cloud. The recommended production mode loads
a validated static map, freezes it, and continues to display the current fused cloud
as `/map/dynamic_overlay` without writing that cloud into the map.

Use `build_static` only when the scene is controlled and stationary. Use
`guarded_continuous` only when its residual/persistence behavior has been validated for
the fixed cameras. Moving-camera SLAM, loop closure, pose graphs, robot FK, semantics,
and collision planning are outside this implementation.
