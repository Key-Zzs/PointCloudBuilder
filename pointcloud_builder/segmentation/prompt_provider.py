"""Configuration-backed concept prompts without task-specific core literals."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(frozen=True)
class PromptSpec:
    concept_id: str
    value: str
    expected_instances: int | None
    prompt_type: Literal["text", "visual_exemplar", "manifest", "dataset_metadata", "vlm"] = "text"


class ManifestPromptProvider:
    """Load a declarative prompt manifest suitable for open-vocabulary data."""

    def __init__(self, prompts: tuple[PromptSpec, ...], *, config_hash: str) -> None:
        self._prompts = prompts
        self.config_hash = config_hash

    @property
    def prompts(self) -> tuple[PromptSpec, ...]:
        return self._prompts

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ManifestPromptProvider":
        raw_bytes = Path(path).read_bytes()
        raw = yaml.safe_load(raw_bytes)
        if not isinstance(raw, dict) or not isinstance(raw.get("concepts"), list):
            raise ValueError("prompt config must contain a concepts list")
        prompts: list[PromptSpec] = []
        seen: set[str] = set()
        for item in raw["concepts"]:
            if not isinstance(item, dict):
                raise ValueError("each prompt concept must be a mapping")
            concept = str(item.get("concept_id", "")).strip()
            value = str(item.get("text", item.get("value", ""))).strip()
            if not concept or not value:
                raise ValueError("prompt concept requires non-empty concept_id and text")
            if concept in seen:
                raise ValueError(f"duplicate concept_id: {concept}")
            seen.add(concept)
            expected = item.get("expected_instances")
            expected_value = int(expected) if expected is not None else None
            if expected_value is not None and expected_value <= 0:
                raise ValueError("expected_instances must be positive when specified")
            kind = str(item.get("prompt_type", "text")).lower()
            if kind not in {"text", "visual_exemplar", "manifest", "dataset_metadata", "vlm"}:
                raise ValueError(f"unsupported prompt_type: {kind}")
            prompts.append(PromptSpec(concept, value, expected_value, kind))  # type: ignore[arg-type]
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return cls(tuple(prompts), config_hash=hashlib.sha256(canonical).hexdigest())
