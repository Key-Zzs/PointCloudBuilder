from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).parents[1] / "tools/mapping/check_memory_plateau.py"
SPEC = importlib.util.spec_from_file_location("check_memory_plateau", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _samples(rss_step: int = 0, cuda_step: int = 0):
    return [
        {
            "frame_index": frame,
            "rss_bytes": 1_000_000_000 + index * rss_step,
            "vmhwm_bytes": 1_100_000_000 + index * max(rss_step, 0),
            "cuda_allocated_bytes": 500_000_000 + index * cuda_step,
            "cuda_reserved_bytes": 700_000_000 + index * cuda_step,
            "cuda_max_allocated_bytes": 550_000_000 + index * max(cuda_step, 0),
            "cuda_max_reserved_bytes": 750_000_000 + index * max(cuda_step, 0),
        }
        for index, frame in enumerate(range(0, 3100, 100))
    ]


def test_flat_memory_passes_plateau() -> None:
    report = MODULE.analyze_memory_plateau(_samples())
    assert report["status"] == "PASS"
    assert report["steady_sample_count"] == 27


def test_linear_rss_growth_fails_slope_gate() -> None:
    report = MODULE.analyze_memory_plateau(_samples(rss_step=6 * MODULE.MIB))
    assert report["status"] == "FAIL"
    assert not report["metrics"]["rss_bytes"]["slope_passed"]


def test_linear_cuda_growth_fails_allocated_and_reserved() -> None:
    report = MODULE.analyze_memory_plateau(_samples(cuda_step=9 * MODULE.MIB))
    assert report["status"] == "FAIL"
    assert not report["metrics"]["cuda_allocated_bytes"]["passed"]
    assert not report["metrics"]["cuda_reserved_bytes"]["passed"]


def test_duplicate_indices_and_too_few_steady_samples_fail_closed() -> None:
    samples = _samples()
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.analyze_memory_plateau(samples + [dict(samples[-1])])
    with pytest.raises(ValueError, match="eight"):
        MODULE.analyze_memory_plateau(samples[:8])
