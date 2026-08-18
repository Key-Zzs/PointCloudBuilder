"""Explicit JSON cache for episode-level support-plane models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .types import SupportPlane


@dataclass(frozen=True)
class SupportPlaneCache:
    plane: SupportPlane
    episode_index: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": "support_plane_v1", "episode_index": self.episode_index, "plane": self.plane.to_dict()}


def save_support_plane(path: str | Path, plane: SupportPlane, *, episode_index: int | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(SupportPlaneCache(plane, episode_index).to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def load_support_plane(path: str | Path, *, config_hash: str | None = None) -> SupportPlane:
    target = Path(path)
    value = json.loads(target.read_text(encoding="utf-8"))
    if value.get("schema_version") != "support_plane_v1":
        raise ValueError(f"unsupported support-plane cache schema: {value.get('schema_version')!r}")
    plane = SupportPlane.from_dict(value["plane"])
    if config_hash is not None and plane.config_hash != config_hash:
        raise ValueError("support-plane cache config hash does not match requested configuration")
    return plane
