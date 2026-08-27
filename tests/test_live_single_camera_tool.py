from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from pointcloud_builder.config import SamplingConfig


PATH = Path(__file__).parents[1] / "tools/mapping/run_live_single_camera.py"
sys.path.insert(0, str(PATH.parent))
SPEC = importlib.util.spec_from_file_location("run_live_single_camera", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_ffs_context_honors_rgb_config(monkeypatch) -> None:
    configured = SimpleNamespace(
        depth_source=SimpleNamespace(ffs=object()),
        pointcloud=SimpleNamespace(use_rgb=True),
        device="cuda",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(MODULE, "load_config", lambda _path: configured)

    def fake_create(bundle, **kwargs):
        captured["bundle"] = bundle
        captured.update(kwargs)
        return "context"

    monkeypatch.setattr(MODULE, "create_ffs_builder", fake_create)
    sampling = SamplingConfig(mode="random", num_points=1024, enabled=False)
    bundle = object()

    result = MODULE._create_ffs_context(bundle, "ffs.yaml", sampling)

    assert result == "context"
    assert captured["bundle"] is bundle
    assert captured["sampling"] is sampling
    assert captured["use_rgb"] is True
