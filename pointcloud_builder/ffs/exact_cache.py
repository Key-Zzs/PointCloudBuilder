"""Lossless, provenance-bound normalized-disparity cache for FFS."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _canonical_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class ExactDisparityCache:
    """One lossless Zarr group per source frame with immutable provenance."""

    schema_version = "ffs_exact_disparity_cache_v1"

    def __init__(self, root: str | Path, *, provenance: dict[str, Any]) -> None:
        self.provenance = {"schema_version": self.schema_version, **provenance}
        self.key = _canonical_hash(self.provenance)
        self.root = Path(root).expanduser().resolve() / self.key
        self.root.mkdir(parents=True, exist_ok=True)
        provenance_path = self.root / "provenance.json"
        if provenance_path.exists():
            existing = json.loads(provenance_path.read_text(encoding="utf-8"))
            if existing != self.provenance:
                raise ValueError("FFS exact cache provenance collision")
        else:
            temporary = provenance_path.with_name(f".{provenance_path.name}.tmp-{os.getpid()}")
            temporary.write_text(json.dumps(self.provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temporary, provenance_path)
        self.hits = 0
        self.misses = 0

    def load(self, frame_index: int, *, device: torch.device) -> torch.Tensor | None:
        path = self._frame_path(frame_index)
        if not path.is_dir():
            self.misses += 1
            return None
        import zarr

        root = zarr.open_group(str(path), mode="r")
        if root.attrs.get("cache_key") != self.key or root.attrs.get("frame_index") != int(frame_index):
            raise ValueError(f"FFS exact cache entry provenance mismatch: {path}")
        array = np.asarray(root["disparity"][:])
        if array.ndim != 2 or array.dtype != np.float32:
            raise ValueError(f"FFS exact cache entry is not normalized float32 HxW: {path}")
        self.hits += 1
        return torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=torch.float32)

    def store(self, frame_index: int, disparity: torch.Tensor) -> None:
        if disparity.ndim != 2 or disparity.dtype != torch.float32:
            raise ValueError("exact FFS cache stores normalized float32 HxW disparity only")
        array = disparity.detach().contiguous().cpu().numpy()
        target = self._frame_path(frame_index)
        if target.exists():
            existing = self.load(frame_index, device=torch.device("cpu"))
            if existing is None or not torch.equal(existing, torch.from_numpy(array)):
                raise ValueError(f"refusing to overwrite non-identical FFS exact cache entry: {target}")
            return
        temporary = target.with_name(f".{target.name}.incomplete-{os.getpid()}")
        import zarr

        root = zarr.group(str(temporary))
        root.create_dataset(
            "disparity",
            data=array,
            chunks=array.shape,
            compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=1),
        )
        root.attrs.update({"cache_key": self.key, "frame_index": int(frame_index), "dtype": str(array.dtype), "shape": list(array.shape)})
        os.replace(temporary, target)

    def metadata(self) -> dict[str, Any]:
        return {"enabled": True, "cache_key": self.key, "root": str(self.root), "hits": self.hits, "misses": self.misses}

    def _frame_path(self, frame_index: int) -> Path:
        if int(frame_index) < 0:
            raise ValueError("FFS exact cache frame_index must be non-negative")
        return self.root / "frames" / f"{int(frame_index):08d}.zarr"
