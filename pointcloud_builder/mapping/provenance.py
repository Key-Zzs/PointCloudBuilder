"""Privacy-safe production backend provenance for rig recordings and benchmarks."""

from __future__ import annotations

from typing import Any

from pointcloud_builder.config import load_config as load_builder_config
from pointcloud_builder.mapping.validation import sha256_file


def rig_backend_provenance(config: Any) -> dict[str, dict[str, object]]:
    """Bind each enabled camera to its depth implementation without private paths."""

    result: dict[str, dict[str, object]] = {}
    for camera in config.enabled_cameras:
        value: dict[str, object] = {"depth_source": camera.depth.mode}
        if camera.pipeline_config is not None:
            builder_config = load_builder_config(camera.pipeline_config)
            ffs = builder_config.depth_source.ffs
            value.update(
                {
                    "backend": None if ffs is None else ffs.backend,
                    "precision": None if ffs is None else ffs.precision,
                    "artifact_id": None if ffs is None else ffs.artifact_id,
                    "pipeline_config_sha256": sha256_file(camera.pipeline_config),
                }
            )
        result[camera.name] = value
    return result


def validate_production_ffs_provenance(
    provenance: object, camera_names: list[str] | tuple[str, ...]
) -> dict[str, dict[str, object]]:
    """Fail closed unless all cameras are bound to the TensorRT plugin."""

    if not isinstance(provenance, dict) or sorted(provenance) != sorted(camera_names):
        raise ValueError("backend provenance must exactly cover recording cameras")
    checked: dict[str, dict[str, object]] = {}
    for name in sorted(camera_names):
        value = provenance.get(name)
        if not isinstance(value, dict):
            raise ValueError("camera backend provenance must be a mapping")
        expected_keys = {
            "depth_source",
            "backend",
            "precision",
            "artifact_id",
            "pipeline_config_sha256",
        }
        if set(value) != expected_keys:
            raise ValueError("camera backend provenance has an invalid field set")
        if value["depth_source"] != "ffs_stereo":
            raise ValueError("production recording must use ffs_stereo")
        if value["backend"] != "tensorrt_plugin":
            raise ValueError("production recording must use tensorrt_plugin")
        if value["precision"] not in {"fp16", "fp32"}:
            raise ValueError("production FFS precision is invalid")
        if not isinstance(value["artifact_id"], str) or not value["artifact_id"]:
            raise ValueError("production FFS artifact_id must be non-empty")
        digest = value["pipeline_config_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("production pipeline config receipt is invalid")
        checked[name] = dict(value)
    return checked
