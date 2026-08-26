# Live single-camera workspace pipeline

`CameraRigLiveSource` owns one stable-API `CameraSession`: open once, capture many
frames synchronously, and close once. It has no background queue. A source can be
opened again after close, creating a fresh session through the same validated
single-camera config.

`LiveSingleCameraWorkspacePipeline` supports native depth and bundle-derived FFS while
sharing the same CameraConfig, CameraBundle, and workspace transform. Each frame reports
capture, adapter, depth inference, deprojection, local crop, workspace transform,
workspace crop, sampling, and end-to-end timings. Acceptance reports also include frame
continuity, stale/duplicate counts, plane geometry, process/GPU memory, balanced
open/close counts, and a reopen run.

Hardware configs, serials, provision artifacts, frames, screenshots, logs, and timing
reports remain under `.local/` and are never published.
