"""Shared fail-closed checks for versioned mapping artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

_CHECKSUM = re.compile(r"([0-9a-f]{64})  ([^\n]+)")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(root: Path, relative_files: list[str]) -> None:
    normalized = _validated_relative_files(relative_files)
    lines = [f"{sha256_file(root / item)}  {item}" for item in normalized]
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_checksums(root: str | Path) -> dict[str, str]:
    artifact = Path(root)
    checksum_path = artifact / "checksums.sha256"
    if (
        not artifact.is_dir()
        or not checksum_path.is_file()
        or checksum_path.is_symlink()
    ):
        raise ValueError("artifact root/checksums.sha256 must be real files")
    recorded: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        match = _CHECKSUM.fullmatch(line)
        if match is None:
            raise ValueError("checksums.sha256 has an invalid line")
        digest, relative = match.groups()
        _validated_relative_files([relative])
        if relative in recorded:
            raise ValueError("checksums.sha256 contains a duplicate path")
        recorded[relative] = digest
    actual = sorted(
        path.relative_to(artifact).as_posix()
        for path in artifact.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    if actual != sorted(recorded):
        raise ValueError("artifact exact file set differs from checksums.sha256")
    for relative, expected in recorded.items():
        candidate = artifact / relative
        if candidate.is_symlink() or sha256_file(candidate) != expected:
            raise ValueError(f"artifact checksum mismatch: {relative}")
    return recorded


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON artifact member must be an object")
    _reject_absolute_strings(value)
    return value


def artifact_member(root: str | Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise ValueError("artifact member reference must be a string")
    _validated_relative_files([relative])
    artifact = Path(root).resolve()
    candidate = (artifact / relative).resolve()
    if candidate.parent != artifact and artifact not in candidate.parents:
        raise ValueError("artifact member escapes the artifact root")
    return candidate


def _validated_relative_files(values: list[str]) -> list[str]:
    normalized = sorted(values)
    if len(normalized) != len(set(normalized)):
        raise ValueError("artifact file list contains duplicates")
    for value in normalized:
        path = Path(value)
        if (
            not value
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != value
            or value == "checksums.sha256"
        ):
            raise ValueError(f"artifact path must be canonical and relative: {value!r}")
    return normalized


def _reject_absolute_strings(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _reject_absolute_strings(item)
    elif isinstance(value, list):
        for item in value:
            _reject_absolute_strings(item)
    elif isinstance(value, str) and (
        value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        raise ValueError("artifact JSON must not contain absolute paths")
