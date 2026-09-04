# Deployment calibration

CameraRig owns target identity, physical dimensions, detection policy, pose solve, and
the atomic fixed-camera provision. PointCloudBuilder consumes the provision only after
CameraRig validates its exact file set and calibration quality.

For a generated custom board, declare page/board dimensions in a CameraRig target v2
YAML, generate the PDF/artifact, print at 100% scale, measure the physical square and
marker sizes, and run the pose-free preflight before provisioning. For an existing
board, capture up to eight representative frames from each distinct camera and run:

```bash
camera-rig target identify-existing \
  --artifact .local/captures/camera_a \
  --artifact .local/captures/camera_b \
  --board-width-mm BOARD_WIDTH --board-height-mm BOARD_HEIGHT \
  --square-length-mm SQUARE --marker-length-mm MARKER \
  --output .local/target/identification.json
```

An ambiguous dictionary/layout result is `PAUSED_FOR_USER_VALIDATION`, not a usable
target. Supply authoritative PDF/generator metadata, rerun identification, then use
`camera-rig target register-existing` and `camera-rig target preflight`. Registration
freezes dictionary, legacy layout, border bits, orientation, dimensions, OpenCV
version, and detection policy. The historical `uncertainty_validated_v1` profile is currently
`HOLD`, not released. Use it only for `target preflight` and `provision preflight`, retaining
private evidence with `--evidence-root`. CameraRig blocks canonical CameraBundle/provision
publication for this profile even when its candidate numerical checks pass.
The additive `uncertainty_validated_v2` structured successor is also `HOLD`: development evidence
does not satisfy the 5% false-reject and false-accept Wilson bounds, so the frozen holdout remains
unopened. Retained repeats can be evaluated with
`camera-rig calibration evaluate-model-counterfactuals`; the output is baseline sensitivity,
not ground-truth pose bias, and cannot release or mutate a provision.

The uncertainty candidate preflight uses exactly 60 selected frames, requires at least 12 successful
poses, at least 50% pose coverage, translation p95 at most 10 mm, and mandatory native
depth-plane PASS. It is not release authority while the preset is HOLD. Short frame counts exist
only through injected tests. Real serials,
images, overlays, measurements, and transforms remain under CameraRig/PCB `.local/`.

Under `uncertainty_validated`, coverage and image span remain operator guidance and
target-size/visual diagnostics, not hard pose-accuracy gates. Low coverage does not guarantee
PASS; it only cannot reject an otherwise qualified observation by itself. At least 12 corners,
the corner fraction and marker-pixel floor, physical PnP, per-frame/final pose uncertainty,
scaled condition, IPPE ambiguity, temporal stability, reprojection, split-half stability, and
native depth remain hard gates. CameraRig raw-stream validation is unchanged and runs first;
raw discontinuity, timeout, timestamp/sync, dtype/shape, or empty-stream failures are
`NOT_APPLICABLE_TO_POSE_OBSERVABILITY`.

The current structured-gate experiment uses the already fixed A4 board for real A/B/C repeats.
Synthetic validation includes 500×700-equivalent geometry, but
`REAL_500X700_STRUCTURED_GATE_VALIDATION=DEFERRED`. Do not move cameras, board, or robot; do not
start multipose or overwrite canonical provisions during this gate-closure stage.

CameraRig now proves render/geometry equivalence before folding candidates that differ
only in `legacy_pattern`; it still fails closed for non-equivalent candidates, wrong
authoritative dictionaries, orientation conflicts, or marker-layout mismatch.

The current 500 x 700 mm physical board has authoritative known metadata
`DICT_4X4_50`, 5 x 7 squares, 100 mm squares, and 75 mm markers. It must use the
existing-board identification and registration route; it is not the generated A4
target. Legacy layout, border bits, and orientation remain evidence-gated. The 500×700 board has no
real structured-gate PASS in this stage. Fixed provisioning remains blocked; a future version
would require an authenticated release loader, an adequate preregistered split, and passing
unopened holdout evidence before it could create an explicitly hash-bound `RELEASED` preset.
