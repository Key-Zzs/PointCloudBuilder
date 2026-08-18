"""Pure contracts for Stage-2 benchmark reporting."""

from __future__ import annotations

import pytest

from tools.stage2 import benchmark_profiles
from tools.stage2.benchmark_profiles import _online_gate, _summary


def test_profile_benchmark_summary_and_online_gate_are_explicit() -> None:
    summary = _summary([1.0, 2.0, 3.0])
    assert summary["count"] == 3
    assert summary["p95_ms"] > summary["median_ms"]
    passing = _online_gate([10.0, 10.0], [11.0, 11.0])
    assert passing["status"] == "ONLINE_ELIGIBLE"
    blocked = _online_gate([10.0, 10.0], [14.0, 14.0])
    assert blocked["status"] == "OFFLINE_ONLY"


def test_missing_outer_source_tools_fail_only_when_benchmark_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_module(name: str) -> None:
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(benchmark_profiles, "import_module", missing_module)
    with pytest.raises(RuntimeError, match="requires the 3D-Diffusion-Policy source-reader tools"):
        benchmark_profiles._load_dp3_source_tools()
