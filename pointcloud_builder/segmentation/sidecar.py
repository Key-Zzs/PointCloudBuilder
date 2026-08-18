"""Compact RLE/Zarr sidecar persistence and strict frame joins."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .types import InstanceMask, SegmentationProvenance


def encode_rle(mask: np.ndarray) -> np.ndarray:
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    if flat.size == 0:
        return np.asarray([], dtype=np.uint32)
    transitions = np.flatnonzero(np.diff(flat, prepend=flat[0] ^ 1))
    # Store alternating run lengths, beginning with the value in metadata.
    ends = np.r_[transitions[1:], flat.size]
    starts = transitions
    return (ends - starts).astype(np.uint32)


def decode_rle(runs: np.ndarray, *, shape: tuple[int, int], starts_with_one: bool) -> np.ndarray:
    flat = np.zeros(int(shape[0]) * int(shape[1]), dtype=bool)
    cursor = 0
    value = bool(starts_with_one)
    for run in np.asarray(runs, dtype=np.uint32):
        end = cursor + int(run)
        if end > flat.size:
            raise ValueError("RLE runs exceed declared mask shape")
        if value:
            flat[cursor:end] = True
        cursor = end
        value = not value
    if cursor != flat.size:
        raise ValueError("RLE runs do not cover declared mask shape")
    return flat.reshape(shape)


class SegmentationSidecar:
    """Zarr-backed video-mask sidecar with an exact source frame map."""

    SCHEMA_VERSION = "paper_a_sam3_sidecar_v1"

    @staticmethod
    def open_stream_writer(
        path: str | Path,
        *,
        provenance: SegmentationProvenance,
        expected_frame_indices: Iterable[int],
        overwrite: bool = False,
    ) -> "StreamingSegmentationSidecarWriter":
        """Open an append-only RLE writer for one atomic episode artifact."""

        return StreamingSegmentationSidecarWriter(
            path,
            provenance=provenance,
            expected_frame_indices=expected_frame_indices,
            overwrite=overwrite,
        )

    @staticmethod
    def write(
        path: str | Path,
        *,
        masks: Iterable[InstanceMask],
        provenance: SegmentationProvenance,
        expected_frame_indices: Iterable[int],
        overwrite: bool = False,
    ) -> None:
        try:
            import zarr
        except ImportError as exc:  # optional dependency by design
            raise RuntimeError("zarr is required only for segmentation sidecars; install pointcloud-builder[stage2]") from exc
        destination = Path(path)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite SAM sidecar: {destination}")
        listed = sorted(masks, key=lambda item: (item.frame_index, item.concept_id, item.track_id))
        expected = tuple(int(item) for item in expected_frame_indices)
        observed = {item.frame_index for item in listed}
        missing = sorted(set(expected) - observed)
        # Missing frame records are permitted only if the backend made them
        # explicit through an invalid record.  This prevents silent joins.
        if missing:
            raise ValueError(f"sidecar has no record for source frames: {missing[:8]}")
        root = zarr.open_group(str(destination), mode="w")
        root.attrs.update({"schema_version": SegmentationSidecar.SCHEMA_VERSION, "provenance": provenance.__dict__})
        entries: list[dict[str, object]] = []
        rle_data: list[np.ndarray] = []
        offset = 0
        for item in listed:
            encoded = encode_rle(item.binary_mask)
            starts_with_one = bool(item.binary_mask.reshape(-1)[0]) if item.binary_mask.size else False
            rle_data.append(encoded)
            entries.append({
                "frame_index": item.frame_index, "episode_index": item.episode_index, "track_id": item.track_id,
                "concept_id": item.concept_id, "bbox": list(item.bbox), "score": item.score,
                "prompt_type": item.prompt_type, "prompt_value": item.prompt_value, "valid": item.valid,
                "shape": list(item.binary_mask.shape), "starts_with_one": starts_with_one,
                "rle_offset": offset, "rle_length": int(encoded.size),
            })
            offset += int(encoded.size)
        values = np.concatenate(rle_data) if rle_data else np.asarray([], dtype=np.uint32)
        root.create_dataset("rle_runs", data=values, chunks=(max(1, min(len(values), 65536)),), compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=1))
        root.attrs["entries_json"] = json.dumps(entries, separators=(",", ":"), ensure_ascii=False)
        root.attrs["expected_frame_indices"] = list(expected)

    @staticmethod
    def read(path: str | Path) -> list[InstanceMask]:
        try:
            import zarr
        except ImportError as exc:
            raise RuntimeError("zarr is required only for segmentation sidecars; install pointcloud-builder[stage2]") from exc
        root = zarr.open_group(str(path), mode="r")
        if root.attrs.get("schema_version") != SegmentationSidecar.SCHEMA_VERSION:
            raise ValueError("unsupported segmentation sidecar schema")
        entries = json.loads(root.attrs["entries_json"])
        runs = np.asarray(root["rle_runs"])
        result: list[InstanceMask] = []
        for entry in entries:
            offset, length = int(entry["rle_offset"]), int(entry["rle_length"])
            mask = decode_rle(runs[offset : offset + length], shape=tuple(entry["shape"]), starts_with_one=bool(entry["starts_with_one"]))
            result.append(InstanceMask(
                frame_index=int(entry["frame_index"]), episode_index=int(entry["episode_index"]), track_id=str(entry["track_id"]),
                concept_id=str(entry["concept_id"]), binary_mask=mask, bbox=tuple(int(item) for item in entry["bbox"]),
                score=float(entry["score"]), prompt_type=str(entry["prompt_type"]), prompt_value=str(entry["prompt_value"]), valid=bool(entry["valid"]),  # type: ignore[arg-type]
            ))
        return result

    @staticmethod
    def index_by_frame(path: str | Path, *, expected_frame_indices: Iterable[int] | None = None) -> dict[int, list[InstanceMask]]:
        indexed: dict[int, list[InstanceMask]] = defaultdict(list)
        for item in SegmentationSidecar.read(path):
            indexed[item.frame_index].append(item)
        if expected_frame_indices is not None:
            missing = sorted(set(int(item) for item in expected_frame_indices) - set(indexed))
            if missing:
                raise ValueError(f"source frame mismatch: no sidecar records for {missing[:8]}")
        return dict(indexed)


class StreamingSegmentationSidecarWriter:
    """Incremental RLE writer that avoids retaining a video of masks in RAM."""

    def __init__(
        self,
        path: str | Path,
        *,
        provenance: SegmentationProvenance,
        expected_frame_indices: Iterable[int],
        overwrite: bool,
    ) -> None:
        try:
            import zarr
        except ImportError as exc:
            raise RuntimeError("zarr is required only for segmentation sidecars; install pointcloud-builder[stage2]") from exc
        self._zarr = zarr
        self.path = Path(path)
        self.expected = tuple(int(item) for item in expected_frame_indices)
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite SAM sidecar: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.root = zarr.open_group(str(self.path), mode="w")
        self.root.attrs.update({"schema_version": SegmentationSidecar.SCHEMA_VERSION, "provenance": provenance.__dict__})
        self.runs = self.root.create_dataset(
            "rle_runs",
            shape=(0,),
            chunks=(65536,),
            dtype=np.uint32,
            compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=1),
        )
        self.entries: list[dict[str, object]] = []
        self.seen_frames: set[int] = set()
        self._closed = False

    def add(self, item: InstanceMask) -> None:
        if self._closed:
            raise RuntimeError("cannot add a mask after sidecar finalization")
        encoded = encode_rle(item.binary_mask)
        offset = int(self.runs.shape[0])
        if encoded.size:
            self.runs.resize(offset + int(encoded.size))
            self.runs[offset : offset + int(encoded.size)] = encoded
        starts_with_one = bool(item.binary_mask.reshape(-1)[0]) if item.binary_mask.size else False
        self.entries.append({
            "frame_index": item.frame_index, "episode_index": item.episode_index, "track_id": item.track_id,
            "concept_id": item.concept_id, "bbox": list(item.bbox), "score": item.score,
            "prompt_type": item.prompt_type, "prompt_value": item.prompt_value, "valid": item.valid,
            "shape": list(item.binary_mask.shape), "starts_with_one": starts_with_one,
            "rle_offset": offset, "rle_length": int(encoded.size),
        })
        self.seen_frames.add(item.frame_index)

    def close(self) -> None:
        if self._closed:
            return
        missing = sorted(set(self.expected) - self.seen_frames)
        if missing:
            raise ValueError(f"sidecar has no record for source frames: {missing[:8]}")
        self.root.attrs["entries_json"] = json.dumps(self.entries, separators=(",", ":"), ensure_ascii=False)
        self.root.attrs["expected_frame_indices"] = list(self.expected)
        self._closed = True

    def __enter__(self) -> "StreamingSegmentationSidecarWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.close()
