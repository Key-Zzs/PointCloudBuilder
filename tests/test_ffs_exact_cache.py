from __future__ import annotations

import json

import pytest
import torch

from pointcloud_builder.ffs.exact_cache import ExactDisparityCache


def test_exact_disparity_cache_is_lossless_and_provenance_bound(tmp_path) -> None:
    provenance = {
        "source_dataset_hash": "source", "frame_join_hash": "join", "ffs_engine_sha": "engine",
        "plugin_sha": "plugin", "calibration_hash": "calibration", "ffs_config_hash": "config",
    }
    cache = ExactDisparityCache(tmp_path, provenance=provenance)
    value = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    cache.store(7, value)
    restored = cache.load(7, device=torch.device("cpu"))
    assert restored is not None and torch.equal(restored, value)
    assert cache.metadata()["hits"] == 1
    assert cache.metadata()["misses"] == 0
    entry = cache.root / "frames" / "00000007.zarr"
    assert json.loads((cache.root / "provenance.json").read_text())["source_dataset_hash"] == "source"
    assert entry.is_dir()


def test_exact_disparity_cache_rejects_non_normalized_storage_and_negative_indices(tmp_path) -> None:
    cache = ExactDisparityCache(tmp_path, provenance={"source_dataset_hash": "source"})
    with pytest.raises(ValueError, match="float32"):
        cache.store(0, torch.ones((2, 2), dtype=torch.float16))
    with pytest.raises(ValueError, match="non-negative"):
        cache.load(-1, device=torch.device("cpu"))
