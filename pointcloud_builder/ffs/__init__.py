"""Optional Fast-FoundationStereo integration.

This module intentionally contains no TensorRT, ONNX, Open3D, or checkpoint
imports.  Heavy dependencies are loaded only after ``depth_source.mode`` is
set to ``ffs_stereo`` and a backend is constructed.
"""

from pointcloud_builder.ffs.types import FFSDepthResult, StereoIRFrame

__all__ = ["FFSDepthResult", "StereoIRFrame"]
