"""Optional segmentation backend protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np

from .prompt_provider import PromptSpec
from .types import InstanceMask, SegmentationProvenance


class InstanceSegmentationBackend(ABC):
    """Video-aware instance segmentation backend.

    Implementations must preserve their native persistent track ID and may not
    silently choose one candidate when ``expected_instances`` is violated.
    """

    @property
    @abstractmethod
    def provenance(self) -> SegmentationProvenance:
        """Return fully resolved code/checkpoint/prompt/source provenance."""

    @abstractmethod
    def segment_episode(
        self,
        *,
        episode_index: int,
        frame_indices: Iterable[int],
        rgb_frames: Iterable[np.ndarray],
        prompts: Iterable[PromptSpec],
    ) -> list[InstanceMask]:
        """Run one video pipeline over a complete episode."""
