# Projection models

## Ownership and compatibility

CameraRig remains the authority for one physical camera's factory stream calibration.
Each CameraRig `CameraIntrinsics` supplies image size, `K`, distortion model,
coefficients, and optical frame. PointCloudBuilder copies that complete contract into
its `CameraIntrinsics` projection model; it never silently drops `D`, the model name,
the frame, or the pixel geometry.

Legacy PointCloudBuilder YAML containing only `width`, `height`, `fx`, `fy`, `cx`, and
`cy` remains valid. It is interpreted explicitly as an ideal, rectified pinhole image:

```yaml
distortion_model: none
distortion_coeffs: []
pixel_geometry: rectified
frame: ""
```

CameraRig factory streams are adapted as `pixel_geometry: raw`, including the case
where the device reports a zero-coefficient model. A rectified model must use
`distortion_model: none` and zero coefficients; inconsistent combinations fail closed.

## Projection support matrix

| CameraRig / RealSense model | XYZ to pixel | Pixel + depth to XYZ | Backend decision |
|---|---:|---:|---|
| `none` | yes | yes | closed-form pinhole |
| `brown-conrady` | yes | yes | exact librealsense ten-step deprojection path |
| `modified-brown-conrady` | yes | identity only | librealsense forward parity; non-identity deprojection fails closed |
| `inverse-brown-conrady` | yes | yes | explicit librealsense API-compatible paths |
| `ftheta` | yes | yes | librealsense-compatible formula |
| `kannala-brandt4` | yes | yes | librealsense-compatible formula and iterative inverse |

No model is approximated with `cv2.projectPoints` merely because it resembles Brown
Conrady. OpenCV is an independent oracle only for the established standard
Brown-Conrady mapping. RealSense parity uses `rs2_project_point_to_pixel` and
`rs2_deproject_pixel_to_point` directly.

## APIs

```python
from pointcloud_builder import deproject_pixels, project_points

deprojected = deproject_pixels(pixels_px, depth_m, projection_model)
projected = project_points(points_camera, projection_model)
```

`project_points` returns pixels plus finite, positive-depth, and image-bounds masks.
It does not perform visibility, z-buffering, occlusion reasoning, or RGB sampling.
Those concerns remain separate so an edge color artifact is not mislabeled as a
projection-model error.

The main builder and cross-camera diagnostic call this shared API. There is no second
private pinhole formula in `builder.py`.

## FFS rectification contract

FFS geometry is expressed in the left-IR optical frame. The current realtime FFS
contract accepts only an identity/no-op stereo rectification: equal left/right `K`,
zero factory coefficients, identity relative rotation, and horizontal
`(-baseline, 0, 0)` translation. The FFS adapter creates a derived projection model
with:

```text
frame = <camera>/ir_left_optical
pixel_geometry = rectified
distortion_model = none
distortion_coeffs = []
```

`ResolvedDepth.intrinsics` is this rectified model. The image and model therefore
agree, and the builder does not undistort the FFS image a second time. A non-identity
stereo contract still fails closed and requires a separately validated offline
rectifier; the realtime TensorRT engine is unchanged.

## IR-left to color mapping

The geometry-only RGB chain is:

```text
pixel/depth in rectified IR-left
-> P_ir_left
-> T_color_from_ir_left
-> P_color
-> color pixel
```

The color stream's own raw distortion model is applied by `project_points`. Nearest
sampling happens only after projection. Visibility and occlusion remain explicitly
unevaluated unless a separate consumer implements them.

## Quantitative parity

Run one private report per physical camera:

```bash
python tools/calibration/audit_projection_parity.py \
  --bundle .local/camera_rig/camera_a/provision/camera_bundle.json \
  --runtime-config .local/camera_rig/camera_a/runtime.yaml \
  --camera-label camera_a \
  --output .local/reports/projection-parity-camera-a.json
```

The audit evaluates a 20 by 15 image grid plus the exact principal point, image center,
and edge midpoints at 0.25, 1.0, and 3.0 m for every factory stream, reports
p50/p95/p99/max and error versus radius, checks applicable
deprojection and round-trip directions, and evaluates the full IR-left-to-color
chain. The frozen API gate is p95 at most 0.10 px and max at most 0.25 px. A measured
reference floor is reported rather than hidden by changing the threshold.

The librealsense reference is built directly from the source CameraRig intrinsics,
not from the adapted PCB model. A separate exact source-to-adapter equality gate checks
K, D, model, frame, and raw pixel geometry. The RGB reference likewise obtains the
source color model and transform independently, while the PCB side uses the derived
FFS rectified left-IR model. Thus an adapter mutation cannot self-confirm parity.

If a user-owned pipeline already holds the device, `--allow-busy-connected` may bind
the private runtime identity to a currently connected device without interrupting
that pipeline. It records dedicated capture as deferred; it does not fabricate a
fresh capture pass.

`MODEL_PARITY=PASS` proves only that PCB executes the model stored by CameraRig.
Factory intrinsic physical accuracy is a separate result and remains
`NOT_FULLY_VALIDATED` without real multi-pose intrinsic evidence.
