from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import signal

import pytest


PATH = Path(__file__).parents[1] / "tools/mapping/run_live_tsdf_mapping.py"
SPEC = importlib.util.spec_from_file_location("run_live_tsdf_mapping", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _parse(*extra: str):
    return MODULE.resolve_args(
        MODULE.build_parser().parse_args(
            ["--rig-config", "rig.yaml", "--tsdf-config", "tsdf.yaml", *extra]
        )
    )


def test_finite_defaults_remain_backward_compatible_with_optional_report() -> None:
    args = _parse()
    assert args.interactive is False
    assert args.matched_sets == 300
    assert args.viewer == "none"
    assert args.viewer_point_budget == 30_000
    assert args.report is None


def test_interactive_defaults_to_unbounded_rerun_and_larger_view_budget() -> None:
    args = _parse("--interactive")
    assert args.interactive is True
    assert args.matched_sets is None
    assert args.viewer == "rerun"
    assert args.viewer_point_budget == 100_000
    assert args.report is None


def test_explicit_interactive_viewer_overrides_are_preserved() -> None:
    args = _parse(
        "--interactive",
        "--viewer",
        "none",
        "--viewer-point-budget",
        "200000",
    )
    assert args.viewer == "none"
    assert args.viewer_point_budget == 200_000


def test_interactive_and_finite_count_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        _parse("--interactive", "--matched-sets", "12")


def test_interactive_history_is_bounded_and_finite_history_is_complete() -> None:
    interactive = _parse("--interactive", "--interactive-stats-window", "3")
    interactive_history = MODULE._history(interactive)
    for value in range(10):
        interactive_history.append(float(value))
    assert list(interactive_history) == [7.0, 8.0, 9.0]

    finite_history = MODULE._history(_parse("--matched-sets", "10"))
    for value in range(10):
        finite_history.append(float(value))
    assert list(finite_history) == [float(value) for value in range(10)]


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_operator_signals_request_normal_cooperative_shutdown(signum: int) -> None:
    request = MODULE._StopRequest()
    originals = MODULE._install_signal_handlers(request)
    try:
        signal.raise_signal(signum)
        assert request.requested
        assert request.signal_name == signal.Signals(signum).name
    finally:
        MODULE._restore_signal_handlers(originals)


def test_report_writer_is_explicit_and_stdout_summary_is_small(tmp_path: Path) -> None:
    destination = tmp_path / "report.json"
    report = {
        "mode": "interactive",
        "interrupted_by": "SIGINT",
        "matched_sets": 42,
        "duration_s": 2.0,
        "snapshot_fps": 21.0,
        "viewer": {"telemetry": {"dropped_packets": 3}},
        "acquisition": {"worker_errors": []},
        "passed": True,
    }
    assert not destination.exists()
    summary = MODULE._stdout_summary(report)
    assert summary == {
        "mode": "interactive",
        "interrupted_by": "SIGINT",
        "matched_sets": 42,
        "duration_s": 2.0,
        "snapshot_fps": 21.0,
        "viewer_dropped_packets": 3,
        "worker_errors": [],
        "passed": True,
    }
    MODULE._write_report(destination, report)
    assert json.loads(destination.read_text(encoding="utf-8")) == report
