"""Atomic, checksummed rig-depth recording artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from pointcloud_builder.camera_model import CameraIntrinsics
from pointcloud_builder.mapping.provenance import validate_production_ffs_provenance
from pointcloud_builder.mapping.types import RigDepthFrameSet, RigDepthObservation
from pointcloud_builder.mapping.validation import (
    artifact_member,
    load_json,
    validate_checksums,
    write_checksums,
)

_SCHEMA_V1 = "pointcloud-builder.rig-depth-recording.v1"
_SCHEMA_V2 = "pointcloud-builder.rig-depth-recording.v2"
_SCHEMA = "pointcloud-builder.rig-depth-recording.v3"


class RigDepthRecordingWriter:
    """Write into a sibling temporary directory and publish only after validation."""

    def __init__(
        self,
        output: str | Path,
        *,
        depth_source: str,
        backend_provenance: dict[str, dict[str, object]] | None = None,
        calibration_purpose: str = "production",
    ) -> None:
        if depth_source not in {"native", "ffs_stereo"}:
            raise ValueError("unsupported rig-depth recording source")
        self.output = Path(output)
        if self.output.exists():
            raise FileExistsError(f"recording output already exists: {self.output}")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._temporary = Path(
            tempfile.mkdtemp(prefix=f".{self.output.name}.tmp-", dir=self.output.parent)
        )
        (self._temporary / "calibration").mkdir()
        (self._temporary / "frames").mkdir()
        (self._temporary / "reports").mkdir()
        self.depth_source = depth_source
        if calibration_purpose not in {"production", "physical_acceptance_candidate"}:
            raise ValueError("unsupported recording calibration purpose")
        self.calibration_purpose = calibration_purpose
        self.backend_provenance = dict(backend_provenance or {})
        if depth_source == "ffs_stereo" and not self.backend_provenance:
            raise ValueError("FFS recording requires backend provenance")
        if depth_source == "native" and self.backend_provenance:
            raise ValueError("native recording must not claim FFS backend provenance")
        self._sets: list[dict[str, Any]] = []
        self._calibrations: dict[str, dict[str, Any]] = {}
        self._closed = False

    def append(self, frame_set: RigDepthFrameSet) -> None:
        if self._closed:
            raise RuntimeError("recording writer is closed")
        expected_source = {item.depth_source for item in frame_set.observations}
        if expected_source != {self.depth_source}:
            raise ValueError("recording depth source differs from observations")
        set_name = f"set_{frame_set.matched_set_index:06d}"
        set_dir = self._temporary / "frames" / set_name
        if set_dir.exists():
            raise ValueError("duplicate matched_set_index in recording")
        set_dir.mkdir()
        cameras = []
        for observation in frame_set.observations:
            self._write_calibration(observation)
            stem = observation.camera_name
            depth_relative = f"frames/{set_name}/{stem}_depth.npy"
            meta_relative = f"frames/{set_name}/{stem}_meta.json"
            np.save(
                self._temporary / depth_relative, observation.depth, allow_pickle=False
            )
            meta = {
                "schema_version": "pointcloud-builder.rig-depth-observation.v1",
                "camera_name": stem,
                "timestamp_ns": observation.timestamp_ns,
                "depth_source": observation.depth_source,
                "depth_unit": observation.depth_unit,
                "depth_scale_m_per_unit": observation.depth_scale_m_per_unit,
                "calibration": f"calibration/{stem}.json",
                "depth": depth_relative,
            }
            _write_json(self._temporary / meta_relative, meta)
            cameras.append(
                {
                    "camera_name": stem,
                    "depth": depth_relative,
                    "meta": meta_relative,
                }
            )
        self._sets.append(
            {
                "matched_set_index": frame_set.matched_set_index,
                "host_timestamp_ns": frame_set.host_timestamp_ns,
                "maximum_skew_ms": frame_set.maximum_skew_ms,
                "raw_to_depth_frame_set_ms": frame_set.raw_to_depth_frame_set_ms,
                "cameras": cameras,
            }
        )

    def _write_calibration(self, observation: RigDepthObservation) -> None:
        value = {
            "schema_version": "pointcloud-builder.rig-depth-calibration.v1",
            "camera_name": observation.camera_name,
            "intrinsics": asdict(observation.intrinsics),
            "T_workspace_from_camera": observation.T_workspace_from_camera.tolist(),
            "source_frame": observation.source_frame,
            "workspace_frame": observation.workspace_frame,
            "bundle_identity": observation.bundle_identity,
            "provision_sha256": observation.provision_sha256,
            "distortion_model": observation.distortion_model,
            "distortion_coeffs": list(observation.distortion_coeffs),
            "rectified": observation.rectified,
            "calibration_mode": observation.calibration_mode,
            "rig_calibration_schema": observation.rig_calibration_schema,
            "rig_calibration_fingerprint": observation.rig_calibration_fingerprint,
            "solution_fingerprint": observation.solution_fingerprint,
            "camera_bundle_sha256": observation.camera_bundle_sha256,
        }
        previous = self._calibrations.get(observation.camera_name)
        if previous is not None and previous != value:
            raise ValueError("camera calibration changed within one recording")
        if previous is None:
            self._calibrations[observation.camera_name] = value
            _write_json(
                self._temporary / "calibration" / f"{observation.camera_name}.json",
                value,
            )

    def finalize(self, *, report: dict[str, Any] | None = None) -> Path:
        if self._closed:
            raise RuntimeError("recording writer is closed")
        if not self._sets:
            raise ValueError("cannot publish an empty rig-depth recording")
        indices = [item["matched_set_index"] for item in self._sets]
        if indices != sorted(indices) or len(set(indices)) != len(indices):
            raise ValueError("recording matched-set indices must be unique and ordered")
        camera_names = sorted(self._calibrations)
        backend_provenance = None
        if self.depth_source == "ffs_stereo":
            backend_provenance = validate_production_ffs_provenance(
                self.backend_provenance, camera_names
            )
        modes = {value["calibration_mode"] for value in self._calibrations.values()}
        schemas = {
            value["rig_calibration_schema"] for value in self._calibrations.values()
        }
        fingerprints = {
            value["rig_calibration_fingerprint"]
            for value in self._calibrations.values()
        }
        solutions = {
            value["solution_fingerprint"] for value in self._calibrations.values()
        }
        if any(
            len(values) != 1 for values in (modes, schemas, fingerprints, solutions)
        ):
            raise ValueError("recording cameras do not share one rig calibration")
        calibration_provenance = {
            "calibration_mode": next(iter(modes)),
            "rig_calibration_schema": next(iter(schemas)),
            "rig_calibration_fingerprint": next(iter(fingerprints)),
            "solution_fingerprint": next(iter(solutions)),
            "camera_bundle_hashes": {
                name: self._calibrations[name]["camera_bundle_sha256"]
                for name in camera_names
            },
            "production_applied": next(iter(modes)) == "validated_multipose_deployment",
        }
        if (
            self.calibration_purpose == "production"
            and calibration_provenance["production_applied"] is not True
        ):
            raise ValueError(
                "production recording requires a validated multi-pose deployment"
            )
        manifest = {
            "schema_version": _SCHEMA,
            "depth_source": self.depth_source,
            "workspace_frame": next(iter(self._calibrations.values()))[
                "workspace_frame"
            ],
            "camera_names": camera_names,
            "matched_set_count": len(self._sets),
            "matched_sets": self._sets,
            "calibrations": [f"calibration/{name}.json" for name in camera_names],
            "backend_provenance": backend_provenance,
            "rig_calibration": calibration_provenance,
            "calibration_purpose": self.calibration_purpose,
        }
        _write_json(self._temporary / "manifest.json", manifest)
        _write_json(self._temporary / "reports" / "recording.json", report or {})
        relative_files = sorted(
            path.relative_to(self._temporary).as_posix()
            for path in self._temporary.rglob("*")
            if path.is_file()
        )
        write_checksums(self._temporary, relative_files)
        validate_rig_depth_recording(self._temporary)
        os.replace(self._temporary, self.output)
        self._closed = True
        return self.output

    def abort(self) -> None:
        if not self._closed and self._temporary.exists():
            shutil.rmtree(self._temporary)
        self._closed = True

    def __enter__(self) -> "RigDepthRecordingWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is not None:
            self.abort()


def validate_rig_depth_recording(root: str | Path) -> dict[str, Any]:
    artifact = Path(root)
    checksums = validate_checksums(artifact)
    manifest = load_json(artifact / "manifest.json")
    schema = manifest.get("schema_version")
    if schema not in {_SCHEMA_V1, _SCHEMA_V2, _SCHEMA}:
        raise ValueError("unsupported rig-depth recording schema")
    camera_names = manifest.get("camera_names")
    matched_sets = manifest.get("matched_sets")
    if (
        manifest.get("depth_source") not in {"native", "ffs_stereo"}
        or not isinstance(manifest.get("workspace_frame"), str)
        or not manifest["workspace_frame"].strip()
    ):
        raise ValueError("recording source/workspace identity is invalid")
    if (
        not isinstance(camera_names, list)
        or any(not isinstance(name, str) for name in camera_names)
        or camera_names != sorted(set(camera_names))
    ):
        raise ValueError("recording camera_names must be canonical and unique")
    if not isinstance(matched_sets, list) or manifest.get("matched_set_count") != len(
        matched_sets
    ):
        raise ValueError("recording matched_set_count mismatch")
    expected_calibrations = [f"calibration/{name}.json" for name in camera_names]
    if manifest.get("calibrations") != expected_calibrations:
        raise ValueError("recording calibration file set mismatch")
    if schema in {_SCHEMA_V2, _SCHEMA}:
        if manifest["depth_source"] == "ffs_stereo":
            validate_production_ffs_provenance(
                manifest.get("backend_provenance"), camera_names
            )
        elif manifest.get("backend_provenance") is not None:
            raise ValueError("native recording cannot contain FFS backend provenance")
    if schema == _SCHEMA:
        _validate_rig_calibration_provenance(
            manifest.get("rig_calibration"),
            camera_names,
            calibration_purpose=manifest.get("calibration_purpose"),
        )
    indices = [
        item.get("matched_set_index") for item in matched_sets if isinstance(item, dict)
    ]
    if (
        len(indices) != len(matched_sets)
        or any(
            isinstance(index, bool) or not isinstance(index, int) for index in indices
        )
        or indices != sorted(set(indices))
    ):
        raise ValueError("recording matched-set indices must be canonical and unique")
    for item in matched_sets:
        cameras = item.get("cameras")
        if (
            not isinstance(cameras, list)
            or [x.get("camera_name") for x in cameras] != camera_names
        ):
            raise ValueError("recording frame camera list mismatch")
    loaded = list(
        iter_rig_depth_recording(
            artifact,
            _validated_manifest=manifest,
            _validated_checksums=checksums,
        )
    )
    if len(loaded) != len(matched_sets):
        raise ValueError("recording failed to load every matched set")
    if any(
        [item.camera_name for item in frame.observations] != camera_names
        for frame in loaded
    ):
        raise ValueError("recording camera set changed between frames")
    if any(
        observation.workspace_frame != manifest["workspace_frame"]
        or observation.depth_source != manifest["depth_source"]
        for frame in loaded
        for observation in frame.observations
    ):
        raise ValueError("recording observation identity differs from manifest")
    for relative in checksums:
        if relative.endswith(".json"):
            load_json(artifact_member(artifact, relative))
    return manifest


def iter_rig_depth_recording(
    root: str | Path,
    *,
    _validated_manifest: dict[str, Any] | None = None,
    _validated_checksums: dict[str, str] | None = None,
) -> Iterator[RigDepthFrameSet]:
    artifact = Path(root)
    if _validated_manifest is None or _validated_checksums is None:
        manifest = validate_rig_depth_recording(artifact)
        checksums = validate_checksums(artifact)
    else:
        manifest = _validated_manifest
        checksums = _validated_checksums
    calibrations = {
        name: load_json(artifact_member(artifact, f"calibration/{name}.json"))
        for name in manifest["camera_names"]
    }
    for name, calibration in calibrations.items():
        if (
            calibration.get("schema_version")
            != "pointcloud-builder.rig-depth-calibration.v1"
            or calibration.get("camera_name") != name
        ):
            raise ValueError("recording calibration identity mismatch")
    for item in manifest["matched_sets"]:
        observations = []
        for camera in item["cameras"]:
            name = camera["camera_name"]
            meta_path = artifact_member(artifact, camera["meta"])
            depth_path = artifact_member(artifact, camera["depth"])
            if camera["meta"] not in checksums or camera["depth"] not in checksums:
                raise ValueError("recording member is absent from checksums")
            meta = load_json(meta_path)
            calibration = calibrations[name]
            expected_calibration = f"calibration/{name}.json"
            if (
                meta.get("schema_version")
                != "pointcloud-builder.rig-depth-observation.v1"
                or meta.get("camera_name") != name
                or meta.get("depth") != camera["depth"]
                or meta.get("calibration") != expected_calibration
                or meta.get("depth_source") != manifest["depth_source"]
            ):
                raise ValueError("recording observation cross-file identity mismatch")
            depth = np.load(depth_path, allow_pickle=False)
            intrinsics = CameraIntrinsics(**calibration["intrinsics"])
            observations.append(
                RigDepthObservation(
                    camera_name=name,
                    depth=depth,
                    depth_unit=meta["depth_unit"],
                    depth_scale_m_per_unit=meta["depth_scale_m_per_unit"],
                    valid_mask=np.isfinite(depth) & (depth > 0),
                    intrinsics=intrinsics,
                    T_workspace_from_camera=np.asarray(
                        calibration["T_workspace_from_camera"]
                    ),
                    timestamp_ns=meta["timestamp_ns"],
                    depth_source=meta["depth_source"],
                    source_frame=calibration["source_frame"],
                    workspace_frame=calibration["workspace_frame"],
                    bundle_identity=calibration["bundle_identity"],
                    provision_sha256=calibration["provision_sha256"],
                    distortion_model=calibration["distortion_model"],
                    distortion_coeffs=tuple(calibration["distortion_coeffs"]),
                    rectified=calibration["rectified"],
                    calibration_mode=calibration.get(
                        "calibration_mode", "fixed_provision"
                    ),
                    rig_calibration_schema=calibration.get("rig_calibration_schema"),
                    rig_calibration_fingerprint=calibration.get(
                        "rig_calibration_fingerprint"
                    ),
                    solution_fingerprint=calibration.get("solution_fingerprint"),
                    camera_bundle_sha256=calibration.get("camera_bundle_sha256"),
                )
            )
        yield RigDepthFrameSet(
            matched_set_index=item["matched_set_index"],
            host_timestamp_ns=item["host_timestamp_ns"],
            maximum_skew_ms=item["maximum_skew_ms"],
            raw_to_depth_frame_set_ms=float(item.get("raw_to_depth_frame_set_ms", 0.0)),
            observations=tuple(observations),
        )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_rig_calibration_provenance(
    value: object,
    camera_names: list[str],
    *,
    calibration_purpose: object,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("recording rig calibration provenance must be a mapping")
    expected = {
        "calibration_mode",
        "rig_calibration_schema",
        "rig_calibration_fingerprint",
        "solution_fingerprint",
        "camera_bundle_hashes",
        "production_applied",
    }
    if set(value) != expected:
        raise ValueError("recording rig calibration provenance field mismatch")
    if calibration_purpose not in {"production", "physical_acceptance_candidate"}:
        raise ValueError("recording calibration purpose is invalid")
    if (
        calibration_purpose == "production"
        and value.get("production_applied") is not True
    ):
        raise ValueError("production recording requires deployed calibration")
    if (
        calibration_purpose == "physical_acceptance_candidate"
        and value.get("production_applied") is not False
    ):
        raise ValueError(
            "candidate physical recording must not claim production calibration"
        )
    hashes = value.get("camera_bundle_hashes")
    if not isinstance(hashes, dict) or sorted(hashes) != camera_names:
        raise ValueError("recording CameraBundle hash set mismatch")
    mode = value.get("calibration_mode")
    if mode == "fixed_provision":
        if value.get("production_applied") is not False:
            raise ValueError("fixed provision is bootstrap-only, not production")
        if any(
            value.get(name) is not None
            for name in (
                "rig_calibration_schema",
                "rig_calibration_fingerprint",
                "solution_fingerprint",
            )
        ):
            raise ValueError("fixed recording cannot claim deployed calibration")
    elif mode == "validated_multipose_deployment":
        if value.get("production_applied") is not True:
            raise ValueError("deployed recording must claim production application")
        if value.get("rig_calibration_schema") != (
            "pointcloud-builder.rig-calibration-deployment.v1"
        ):
            raise ValueError("recording deployment schema mismatch")
        for name in ("rig_calibration_fingerprint", "solution_fingerprint"):
            digest = value.get(name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"recording {name} is invalid")
        if any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in hashes.values()
        ):
            raise ValueError("recording deployed CameraBundle hash is invalid")
    else:
        raise ValueError("recording calibration_mode is unsupported")
    return dict(value)
