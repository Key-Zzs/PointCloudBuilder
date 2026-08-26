# TSDF dynamic handling

`frozen_static` is the default production mode: load a validated static TSDF, keep it
read-only, and overlay the current M8 fused cloud. `build_static` integrates a bounded
stationary sequence and then freezes. Neither mode writes the dynamic overlay into the
static map.

`guarded_continuous` is opt-in for fixed cameras. The mapper raycasts predicted depth
for each camera and compares observed minus predicted depth per pixel:

- residual within the configured threshold is background-consistent;
- a large short-lived or moving residual is masked by writing depth zero;
- a new surface must remain within `consistency_tolerance_m` for
  `persistence_frames` before integration;
- disappearance of a transient foreground reveals and reintegrates consistent static
  background.

The default thresholds are 20 mm residual, 10 consecutive frames, and 10 mm temporal
consistency. Reports count consistent, transient, persistent, newly persistent, and
integrated pixels plus residual percentiles. State is independent per camera and is
cleared by full map reset.

Synthetic hard-gate sequences cover a transient cube, a moving cube, disappearance,
and a new fixed cube. An end-to-end Open3D test filters the sequence, integrates the
result into a real VoxelBlockGrid, and requires zero extracted geometry in the
transient-only target, recovery of the revealed background, and complete mapped
coverage of the persistent target after continued fixed frames. Filtered integration
starts on the preregistered persistence frame. Real user-directed
cube motion is useful evidence but is not substituted for this automatic gate and is
not reported as PASS when it was not performed.

The async mapper caps updates at 5/10 Hz and mesh extraction at 1 Hz by configuration.
Its latest-only queue drops old frame sets under load. Snapshot mapping remains
independent; acceptance requires no more than 10% FPS loss or a producer-side latency
increase proxy at most 5 ms, bounded queue depth, and a measured child-RSS plateau.
The RSS gate requires at least 32 samples, discards the first 20% as warmup, and then
requires both last-minus-first quartile median growth at most 256 MiB and a fitted
slope at most 5 MiB per 100 frames.
