# Changelog

## 0.2.0 - 2026-08-27

### Added

- CameraRig fixed-camera integration with explicit frame-aware transforms.
- Native and Fast-FoundationStereo depth, including reproducible PyTorch and TensorRT routes.
- Concurrent host-time-matched multi-camera acquisition and deterministic workspace fusion.
- Dense XYZRGB reconstruction with voxel-centroid geometry and mean RGB aggregation.
- Optional post-fusion FPS compact output and raw concatenation diagnostics.
- Process-isolated Rerun visualization with bounded latest-only queues.
- Persistent Open3D TSDF build, freeze, overlay, extraction, save and load workflows.
- Interactive until-interrupted TSDF/Rerun operator mode with bounded statistics.
- Dedicated reconstruction environment, doctor, FFS asset checker and command validation.
- Clean-room reproduction from a fresh clone through fixed-camera calibration, all four
  FFS routes, dual-camera RGB reconstruction, Rerun, offline/live TSDF, and save/load.

### Changed

- Dense reconstruction now defaults to sampling disabled; compact output uses FPS instead of a second voxel pass.
- Public reconstruction profiles use the benchmark-selected 2.5 mm fusion voxel.
- The full environment uses only `opencv-contrib-python-headless` as its OpenCV wheel.

### Privacy

- Real serials, captures, calibration bundles, model assets, maps, reports, screenshots and RRD files remain under ignored `.local/` paths.
