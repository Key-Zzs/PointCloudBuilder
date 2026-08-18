"""Opt-in SAM3 video adapter kept outside the default import path."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .base import InstanceSegmentationBackend
from .prompt_provider import PromptSpec
from .types import InstanceMask, SegmentationProvenance


class SAM3VideoBackend(InstanceSegmentationBackend):
    """Adapter boundary for the official SAM3 video/Object Multiplex API.

    It intentionally does not fall back to independent image segmentation.  A
    release-specific adapter must be installed in the isolated SAM environment
    and expose the documented ``segment_video`` callable supplied here.
    """

    def __init__(self, *, segment_video: object, provenance: SegmentationProvenance) -> None:
        if not callable(segment_video):
            raise TypeError("segment_video must be the official SAM3 video API callable")
        self._segment_video = segment_video
        self._provenance = provenance

    @property
    def provenance(self) -> SegmentationProvenance:
        return self._provenance

    def segment_episode(self, *, episode_index: int, frame_indices: Iterable[int], rgb_frames: Iterable[np.ndarray], prompts: Iterable[PromptSpec]) -> list[InstanceMask]:
        records = self._segment_video(episode_index=episode_index, frame_indices=list(frame_indices), rgb_frames=list(rgb_frames), prompts=list(prompts))
        if not isinstance(records, list) or not all(isinstance(item, InstanceMask) for item in records):
            raise RuntimeError("SAM3 adapter must return InstanceMask records with persistent video track IDs")
        return records
