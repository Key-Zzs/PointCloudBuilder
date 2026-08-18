"""Model-independent instance-mask records used by Stage 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class InstanceMask:
    frame_index: int
    episode_index: int
    track_id: str
    concept_id: str
    binary_mask: np.ndarray
    bbox: tuple[int, int, int, int]
    score: float
    prompt_type: Literal["text", "visual_exemplar", "manifest", "dataset_metadata", "vlm"]
    prompt_value: str
    valid: bool

    def __post_init__(self) -> None:
        if self.binary_mask.ndim != 2:
            raise ValueError("InstanceMask.binary_mask must be H x W")
        if self.binary_mask.dtype != np.bool_:
            object.__setattr__(self, "binary_mask", self.binary_mask.astype(bool, copy=False))
        if len(self.bbox) != 4:
            raise ValueError("InstanceMask.bbox must have four values")


@dataclass(frozen=True)
class SegmentationProvenance:
    execution: str
    code_commit: str
    checkpoint_id: str
    checkpoint_sha256: str
    prompt_config_sha256: str
    source_dataset_sha256: str
