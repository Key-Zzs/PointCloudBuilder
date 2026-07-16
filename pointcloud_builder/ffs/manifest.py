"""Machine-readable FFS artifact manifest validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(
    path: str | Path,
    *,
    backend: str,
    height: int,
    width: int,
    max_disp: int,
    valid_iters: int,
    precision: str,
    normalization_contract: str,
    artifact_paths: Iterable[str | Path] = (),
    input_names: Iterable[str] = (),
    output_names: Iterable[str] = (),
    config_path: str | Path | None = None,
    builder_optimization_level: int | None = None,
    workspace_gib: float | None = None,
) -> dict[str, Any]:
    """Load and reject any manifest that does not match the run contract."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"FFS artifact manifest is required: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"FFS manifest must be a JSON object: {manifest_path}")
    expected = {
        "backend": backend,
        "height": int(height),
        "width": int(width),
        "max_disp": int(max_disp),
        "valid_iters": int(valid_iters),
        "precision": precision,
        "normalization_contract": normalization_contract,
    }
    if builder_optimization_level is not None:
        expected["builder_optimization_level"] = int(builder_optimization_level)
    if workspace_gib is not None:
        expected["workspace_gib"] = float(workspace_gib)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"FFS manifest mismatch for {key}: manifest={manifest.get(key)!r}, configured={value!r}"
            )
    for key, names in (("input_names", list(input_names)), ("output_names", list(output_names))):
        if names and list(manifest.get(key, ())) != names:
            raise ValueError(
                f"FFS manifest mismatch for {key}: manifest={manifest.get(key)!r}, configured={names!r}"
            )
    expected_fp16 = precision == "fp16"
    for record_name in ("engine", "feature_engine", "post_engine"):
        record = manifest.get(record_name)
        if isinstance(record, Mapping) and "fp16" in record and bool(record["fp16"]) != expected_fp16:
            raise ValueError(
                f"FFS manifest mismatch for {record_name}.fp16: "
                f"manifest={record['fp16']!r}, configured_precision={precision!r}"
            )
    artifact_values = tuple(Path(value).expanduser().resolve() for value in artifact_paths)
    for artifact_path in artifact_values:
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Manifest artifact is missing: {artifact_path}")
        recorded = _recorded_hash(manifest, artifact_path)
        if recorded is None:
            raise ValueError(f"FFS manifest has no SHA-256 entry for required artifact: {artifact_path}")
        if recorded != sha256_file(artifact_path):
            raise ValueError(f"Artifact SHA-256 mismatch for {artifact_path}")
    if artifact_values:
        resolved_config = resolve_artifact_config_path(
            artifact_values[0],
            explicit_config_path=config_path,
            manifest_path=manifest_path,
        )
        _validate_config_contract(
            resolved_config,
            expected=expected,
            input_names=list(input_names),
            output_names=list(output_names),
            manifest=manifest,
        )
        manifest["resolved_config_path"] = str(resolved_config)
    return dict(manifest)


def resolve_artifact_config_path(
    engine_path: str | Path,
    *,
    explicit_config_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> Path:
    """Resolve an engine's contract file without guessing between candidates.

    The order is explicit config, a manifest-declared config, same-stem YAML,
    and finally a directory containing exactly one YAML.  A missing config or
    more than one fallback candidate is an error by design.
    """

    engine = Path(engine_path).expanduser().resolve()
    if explicit_config_path is not None:
        return _require_config_file(Path(explicit_config_path).expanduser().resolve())
    manifest_file = Path(manifest_path).expanduser().resolve() if manifest_path is not None else None
    if manifest_file is not None:
        if not manifest_file.is_file():
            raise FileNotFoundError(f"FFS artifact manifest is required: {manifest_file}")
        with manifest_file.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        declared = manifest.get("config_path") if isinstance(manifest, Mapping) else None
        if declared:
            declared_path = Path(str(declared)).expanduser()
            if not declared_path.is_absolute():
                declared_path = manifest_file.parent / declared_path
            return _require_config_file(declared_path.resolve())
    same_stem = [engine.with_suffix(suffix) for suffix in (".yaml", ".yml") if engine.with_suffix(suffix).is_file()]
    if len(same_stem) == 1:
        return same_stem[0]
    if len(same_stem) > 1:
        raise ValueError(f"Multiple same-stem FFS configs found for {engine}: {same_stem}")
    directory_candidates = sorted(
        path for suffix in ("*.yaml", "*.yml") for path in engine.parent.glob(suffix) if path.is_file()
    )
    if len(directory_candidates) == 1:
        return directory_candidates[0].resolve()
    if len(directory_candidates) > 1:
        raise ValueError(
            f"FFS engine config is ambiguous for {engine}; candidates={directory_candidates}"
        )
    raise FileNotFoundError(
        f"No FFS config found for engine {engine}; provide depth_source.ffs.config_path"
    )


def _require_config_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"FFS engine config does not exist: {path}")
    return path


def _validate_config_contract(
    path: Path,
    *,
    expected: Mapping[str, Any],
    input_names: list[str],
    output_names: list[str],
    manifest: Mapping[str, Any],
) -> None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except Exception as exc:
        raise ValueError(f"Could not parse FFS engine config {path}: {exc}") from exc
    if not isinstance(config, Mapping):
        raise ValueError(f"FFS engine config must be a mapping: {path}")
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"FFS engine/config mismatch for {key}: config={config.get(key)!r}, configured={value!r}"
            )
    if input_names and list(config.get("input_names", ())) != input_names:
        raise ValueError(
            f"FFS engine/config mismatch for input_names: config={config.get('input_names')!r}, configured={input_names!r}"
        )
    if output_names and list(config.get("output_names", ())) != output_names:
        raise ValueError(
            f"FFS engine/config mismatch for output_names: config={config.get('output_names')!r}, configured={output_names!r}"
        )
    for key in ("backend", "height", "width", "max_disp", "valid_iters", "precision", "normalization_contract"):
        if config.get(key) != manifest.get(key):
            raise ValueError(
                f"FFS manifest/config mismatch for {key}: manifest={manifest.get(key)!r}, config={config.get(key)!r}"
            )


def _recorded_hash(manifest: Mapping[str, Any], path: Path) -> str | None:
    artifacts = manifest.get("artifacts", ())
    if isinstance(artifacts, Mapping):
        value = artifacts.get(path.name) or artifacts.get(str(path))
        if isinstance(value, Mapping):
            value = value.get("sha256")
        return str(value) if value else None
    if isinstance(artifacts, list):
        for entry in artifacts:
            if not isinstance(entry, Mapping):
                continue
            name = entry.get("path") or entry.get("name")
            if name and (Path(str(name)).name == path.name or str(name) == str(path)):
                value = entry.get("sha256")
                return str(value) if value else None
    return None
