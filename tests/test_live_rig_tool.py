from __future__ import annotations

import importlib.util
import sys
from types import ModuleType
from pathlib import Path

PATH = Path(__file__).parents[1] / "tools/mapping/run_live_rig.py"
sys.path.insert(0, str(PATH.parent))
SPEC = importlib.util.spec_from_file_location("run_live_rig", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
PSUTIL_MISSING = importlib.util.find_spec("psutil") is None
if PSUTIL_MISSING:
    sys.modules["psutil"] = ModuleType("psutil")
try:
    SPEC.loader.exec_module(MODULE)
finally:
    if PSUTIL_MISSING:
        sys.modules.pop("psutil", None)


def _matcher() -> dict:
    return {
        "matched_sets": 1000,
        "match_ratio": 0.58,
        "absolute_skew_ms": {
            "camera_a": {"p95": 0.0},
            "camera_b": {"p95": 10.0},
        },
        "maximum_absolute_skew_ms": 10.5,
        "frame_reuse_violations": 0,
        "wait_timeouts": 0,
    }


def test_capture_matching_scope_accepts_latest_biased_delivery() -> None:
    matcher = _matcher()

    assert MODULE._matcher_passed(
        matcher,
        received_frames=1000,
        requested_frames=1000,
        reference_camera="camera_a",
        require_match_ratio=False,
    )
    assert not MODULE._matcher_passed(
        matcher,
        received_frames=1000,
        requested_frames=1000,
        reference_camera="camera_a",
        require_match_ratio=True,
    )


def test_capture_matching_scope_rejects_reuse_or_wait_timeout() -> None:
    matcher = _matcher()
    matcher["frame_reuse_violations"] = 1
    assert not MODULE._matcher_passed(
        matcher,
        received_frames=1000,
        requested_frames=1000,
        reference_camera="camera_a",
        require_match_ratio=False,
    )

    matcher["frame_reuse_violations"] = 0
    matcher["wait_timeouts"] = 1
    assert not MODULE._matcher_passed(
        matcher,
        received_frames=1000,
        requested_frames=1000,
        reference_camera="camera_a",
        require_match_ratio=False,
    )
