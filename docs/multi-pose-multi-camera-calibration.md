# Multi-pose, multi-camera calibration

## Architecture boundary

CameraRig remains strictly single-camera. One CameraRig, CameraSession, CameraBundle,
or FixedMountCalibration represents one physical device. CameraRig owns factory
intrinsics/distortion, internal stream extrinsics, target detection, single-camera
PnP, and fixed-mount artifacts.

PointCloudBuilder owns `pointcloud_builder.rig_calibration`: pose grouping, the
multi-camera observation graph, N-camera initialization, joint bundle adjustment,
cross-camera quality, candidate export, and future diagnostic comparison. No
diagnostic ICP transform is used as a calibration input or written to production.

## Frames and objective

Transforms use column vectors and the unambiguous `T_target_from_source` convention.
For fixed camera `i`, target pose `j`, and target corner `k`:

```text
X_workspace = T_workspace_from_target[j] X_target[k]
X_camera = inverse(T_workspace_from_camera[i]) X_workspace
u_hat = project_points(X_camera, projection_model[i])
```

The solver minimizes `rho(||u_observed - u_predicted||^2)` per accepted 2-D corner;
Huber/Cauchy weights are applied to the corner block, not independently to x and y.
Intrinsics and distortion are fixed in v1; they are not selected or adjusted by
model performance. SE(3) uses a rotation vector plus translation, never 16 free
matrix entries. Every output rotation is revalidated for orthonormality and
determinant +1.

## Gauge fixing and pose 0

The canonical target pose is fixed:

```text
T_workspace_from_target[pose_0] = I
```

Thus workspace equals target pose 0. In a future physical acquisition, place the
board at the canonical workspace location for pose 0. Cameras remain fixed while the
board moves for later poses.

## Versioned observation artifact

`pointcloud-builder.rig-calibration-observations.v1` stores:

- target identity, physical/logical camera identities, and CameraBundle hashes;
- one complete projection-model snapshot per camera;
- optional existing `T_workspace_from_camera` initial guesses;
- pose-group IDs;
- per-camera CameraRig target corners, IDs, timestamp, quality, and solve/holdout split.

A `point_id` must resolve to one identical target-frame 3-D coordinate across every
camera and pose. Missing explicit `quality.passed=true` evidence and inconsistent
target geometry fail before optimization.

The solver depends on corner arrays, not images. Physical images, identities, bundles,
and reports remain under ignored `.local/` paths.

## Partial visibility and connectivity

The observation graph is bipartite: camera nodes on one side and target-pose nodes on
the other. A camera need not see every pose, and no pose need be visible to every
camera. The graph must be connected. A detached camera or pose component fails before
optimization with:

```text
DISCONNECTED_CALIBRATION_GRAPH
```

PnP estimates `T_camera_from_target` on each observed edge. Existing fixed provisions
are optional camera initial guesses, not constraints. Missing camera or target
initializers propagate through the connected graph, so the anchor camera need not see
every target pose.

## Diversity and degeneracy

Preflight reports and gates:

- image coverage per camera;
- target translation span;
- camera-relative target depth span;
- board-normal angular span;
- yaw and pitch spans;
- minimum corner and per-camera observation counts.

Repeated copies of one pose, tiny central coverage, fronto-parallel-only motion,
insufficient corners, insufficient camera observations, and disconnected graphs fail
closed. The stable status is `INSUFFICIENT_POSE_DIVERSITY`; repeated video frames are
not relabeled as distinct physical poses.

## Optimizer and quality report

The default optimizer is `scipy.optimize.least_squares` with Huber loss and a recorded
1 px scale; Cauchy is also supported. No corner is silently deleted. Frame-level
rejection is not performed by v1. CameraRig target quality is checked first, then the
robust loss contains pixel outliers.

`RigCalibrationSolution` records explicit camera/pose/observation/corner counts,
source identities and hashes, graph connectivity, diversity,
initial/final reprojection, per-camera and per-pose metrics, initial-to-candidate
camera correction magnitude, termination, function evaluations, Jacobian rank,
condition number, robust loss, and gauge anchor. The solution is candidate-only and
cannot update a CameraBundle in place.

## Solve, validate, and export

```bash
python tools/calibration/solve_multicamera_multipose.py \
  --observations .local/calibration/observations.json \
  --config configs/calibration/multipose_rig_example.yaml \
  --output .local/calibration/solution.json

python tools/calibration/validate_multicamera_multipose.py \
  --solution .local/calibration/solution.json \
  --observations .local/calibration/observations.json \
  --output .local/calibration/validation.json

python tools/calibration/export_rig_calibration.py \
  --solution .local/calibration/solution.json \
  --validation .local/calibration/validation.json \
  --output-root .local/calibration/fixed-mount-candidates
```

Export requires a passed validation artifact cryptographically bound to the exact
solution and writes new per-camera candidate artifacts carrying their source bundle
identity/hash. It never edits an existing CameraBundle or changes `/clouds/fused`.
All real observation, solution, validation, and candidate CLI paths fail closed unless
their resolved paths are below the repository-root `.local/` directory.

Solve/holdout pose splits are explicit. Each holdout target pose is initialized from
per-camera PnP and then jointly refined as one six-DoF nuisance pose against all of
that pose's observations while every candidate camera pose remains fixed. All corners
are scored with linear loss and no rejection. This prevents presenting optimizer
training residual as the only validation evidence. Each holdout pose must be observed
by at least two cameras; fitting and scoring a pose from one camera alone is rejected
as self-validation.

## Future physical capture

The operator workflow captures stationary matched sets:

```bash
python tools/calibration/capture_multicamera_target_poses.py \
  --rig-config .local/configs/live_rig.yaml \
  --target .local/target/target_spec.json \
  --pose-count 24 \
  --holdout-pose-count 4 \
  --min-corners-per-observation 6 \
  --output .local/calibration/observations.json
```

Use 15 to 30 poses spanning center, left, right, top, bottom, near, far, positive and
negative yaw, positive and negative pitch, and combined rotations. Holdout poses are
predeclared as the final N captures and are never used by the optimizer. For each
prompt: move the board, stop it, then capture. Connected partial visibility is
expected and supported.

## Synthetic acceptance and real status

```bash
python tools/calibration/run_synthetic_acceptance.py \
  --output .local/reports/rig-calibration-synthetic-acceptance.json
```

The independent OpenCV observation oracle covers 2-camera Gaussian noise, perturbed
initial extrinsics, 3-camera full/partial visibility, disconnected failure, 4-camera
generic smoke, both required camera permutations, pose order invariance, gauge-invariant
odd/even split stability, same-pose/tiny/fronto-parallel
degeneracy, robust outliers, and nonzero Brown radial/tangential distortion. A pinhole
solver on the nonzero-distortion scene must fail its frozen gate.

The existing 300-set static capture is checked with
`validate_static_capture_pose_diversity.py`; all frames remain one pose group and must
return `INSUFFICIENT_POSE_DIVERSITY`.

Without a connected third physical camera:

```text
REAL_CAMERA_C_AVAILABLE = NO
REAL_3_CAMERA_VALIDATION = DEFERRED
REAL_MULTIPOSE_PHYSICAL_VALIDATION = DEFERRED
```

These deferrals do not block a passed generic N-camera implementation.

## Candidate-only cross-camera re-diagnosis

`rig_calibration.diagnostics.candidate_T_workspace_from_geometry_source` composes a
candidate color-frame camera pose with the authoritative, frame-explicit CameraRig
internal transform for native depth or FFS IR-left geometry. This lets the existing cross-camera
diagnostic rebuild before/candidate clouds without overwriting production transforms.
The adapter checks source and target frame names, and its generated override contract
binds the solution fingerprint, target identity, and per-camera source bundle
identity/hash while marking it candidate-only with `production_applied=false`.

Use one immutable diagnostic capture for both baselines and write each analysis to a
separate directory:

```bash
python tools/mapping/diagnose_cross_camera_alignment.py capture \
  --rig-config .local/configs/cross_camera_alignment_ffs.yaml \
  --matched-sets 300 \
  --output .local/diagnostics/real_dual_multipose_v1

python tools/mapping/diagnose_cross_camera_alignment.py analyze \
  --input .local/diagnostics/real_dual_multipose_v1 \
  --mapping-config .local/configs/m2_native.yaml \
  --analysis-output .local/diagnostics/real_dual_multipose_v1/production \
  --allow-missing-cube

python tools/mapping/diagnose_cross_camera_alignment.py analyze \
  --input .local/diagnostics/real_dual_multipose_v1 \
  --mapping-config .local/configs/m2_native.yaml \
  --analysis-output .local/diagnostics/real_dual_multipose_v1/candidate \
  --candidate-solution .local/calibration/real_dual_multipose_v1/solution.json \
  --candidate-validation .local/calibration/real_dual_multipose_v1/validation_refined_v2.json \
  --allow-missing-cube
```

Candidate analysis fails closed unless the validation is passed, its holdout status
is `PASS`, its solution fingerprint matches, and the recorded CameraBundle identities,
hashes, source frames, and workspace frame still match. Analysis never modifies the
recording or production CameraBundles.

`--allow-missing-cube` selects explicit overlap-only analysis and skips the cube
detector. The resulting manifest records cube acceptance as `NOT_RUN`; it cannot
be used as complete 3D physical acceptance. Omit the flag for a formal board-plus-cube
run.

To inspect a passed candidate continuously on the current live scene without
modifying any CameraBundle, run:

```bash
python tools/mapping/view_live_candidate_calibration.py \
  --rig-config .local/configs/cross_camera_alignment_ffs.yaml \
  --candidate-solution .local/calibration/real_dual_multipose_v1/solution.json \
  --candidate-validation .local/calibration/real_dual_multipose_v1/validation_refined_v2.json
```

This operator-only viewer applies the candidate geometry in memory, runs until
Ctrl-C, and reports `production_applied=false`. It does not publish a map or an
accepted production calibration.
