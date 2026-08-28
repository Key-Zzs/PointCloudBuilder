from __future__ import annotations

from pathlib import Path

import pytest

from pointcloud_builder.local_paths import require_repo_local_path


def test_repository_local_path_uses_repository_root_not_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    expected = (repository_root / ".local" / "reports" / "result.json").resolve()
    monkeypatch.chdir(tmp_path)

    assert require_repo_local_path(expected, label="test artifact") == expected
    with pytest.raises(ValueError, match="repository .local"):
        require_repo_local_path(
            tmp_path / ".local" / "decoy.json", label="test artifact"
        )


def test_repository_local_path_rejects_resolved_traversal() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="repository .local"):
        require_repo_local_path(
            repository_root / ".local" / ".." / "escape.json",
            label="test artifact",
        )
