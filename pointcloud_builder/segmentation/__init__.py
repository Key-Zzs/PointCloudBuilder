"""Optional instance-segmentation interfaces; no model is imported by default."""

from .base import InstanceSegmentationBackend
from .prompt_provider import ManifestPromptProvider, PromptSpec
from .sidecar import SegmentationSidecar, decode_rle, encode_rle
from .types import InstanceMask, SegmentationProvenance

__all__ = [
    "InstanceMask",
    "InstanceSegmentationBackend",
    "ManifestPromptProvider",
    "PromptSpec",
    "SegmentationProvenance",
    "SegmentationSidecar",
    "decode_rle",
    "encode_rle",
]
