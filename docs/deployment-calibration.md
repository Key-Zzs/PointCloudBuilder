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
version, and detection policy. Provision each camera independently with
`camera-rig provision fixed`, then validate both provision roots.

The release preflight uses exactly 60 selected frames, requires at least 12 successful
poses, at least 50% pose coverage, translation p95 at most 10 mm, and mandatory native
depth-plane PASS. Short frame counts exist only through injected tests. Real serials,
images, overlays, measurements, and transforms remain under CameraRig/PCB `.local/`.

The current 500 x 700 mm physical board remains unresolved because several ChArUco
candidate dictionaries/layouts are visually indistinguishable. No target registration,
reprovision, or physical calibration PASS is claimed until authoritative metadata is
provided.
