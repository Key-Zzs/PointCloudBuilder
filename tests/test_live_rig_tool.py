from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

PATH = Path(__file__).parents[1] / "tools/mapping/run_live_rig.py"
sys.path.insert(0, str(PATH.parent))
SPEC = importlib.util.spec_from_file_location("run_live_rig", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_help_does_not_import_optional_psutil(tmp_path: Path) -> None:
    (tmp_path / "psutil.py").write_text(
        "raise RuntimeError('psutil imported during --help')\n", encoding="utf-8"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path), str(PATH.parents[2]), str(PATH.parents[2] / "third_party/CameraRig/src"))
    )
    result = subprocess.run(
        [sys.executable, str(PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=PATH.parents[2],
    )

    assert result.returncode == 0, result.stderr
    assert "--acceptance-scope" in result.stdout


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
