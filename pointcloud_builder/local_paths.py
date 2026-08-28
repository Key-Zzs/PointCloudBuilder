"""Fail-closed containment for private repository-local artifacts."""

from __future__ import annotations

from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def require_repo_local_path(path: str | Path, *, label: str) -> Path:
    """Resolve ``path`` and require containment below this repository's ``.local``."""

    local_root = (_REPOSITORY_ROOT / ".local").resolve()
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(local_root):
        raise ValueError(f"{label} must remain under repository .local: {local_root}")
    return resolved
