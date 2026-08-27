# Current snapshot versus persistent map

These products answer different questions and must not be substituted for one another.

| Property | Current snapshot | Persistent TSDF |
|---|---|---|
| Input | Current per-camera point clouds | Per-camera depth pixels, K, pose |
| Time | One matched set | Many matched sets |
| Output | Fused/cropped/sampled current tensor | Raw/cropped/sampled points, full mesh, volume |
| Dynamic objects | Preserved | Excluded or guarded |
| Update owner | Mapping main process | Independent Open3D process |

The snapshot sequence is per-camera deprojection, workspace transform, crop, voxel
fusion, then one global sampler. It remains authoritative for robot-policy input and
the current dynamic overlay.

```text
Current snapshot:
raw -> FFS -> deproject -> workspace -> fuse -> crop -> sample

Persistent:
raw -> FFS -> metric depth -> TSDF update
                                |
                             extract
                                |
                              crop
                                |
                            sampling
```

The current implementation retains its already accepted M8 crop/fusion execution
semantics; the scenario benchmark reports actual ordering rather than silently changing
geometry. The frozen-input benchmark compares reconstruction only, reconstruction plus
crop, and reconstruction plus crop/sampling with identical frames, transforms, depth
backend, device, warmup, and iteration count.
The report binds each camera to a checksummed pipeline config and directly measures
the canonical combined sequential camera stage. Its enabled/disabled overhead run
alternates execution order and disables builder, single-camera, and rig timers together.

The map sequence integrates each camera separately. It preserves each pixel ray,
metric depth, stream intrinsics, and `T_workspace_from_camera`. It never accepts an
already fused or globally sampled point cloud. The recommended production mode loads
a validated static map, freezes it, and continues to display the current fused cloud
as `/map/dynamic_overlay` without writing that cloud into the map.

Production persistent depth is FFS TensorRT plugin. Native TSDF is an optional
diagnostic baseline, not a fallback or production gate. Map update timing and map
extraction timing are always reported separately because extraction is not per-frame.

Use `build_static` only when the scene is controlled and stationary. Use
`guarded_continuous` only when its residual/persistence behavior has been validated for
the fixed cameras. Moving-camera SLAM, loop closure, pose graphs, robot FK, semantics,
and collision planning are outside this implementation.
