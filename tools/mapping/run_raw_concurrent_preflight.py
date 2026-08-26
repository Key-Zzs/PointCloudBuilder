#!/usr/bin/env python3
"""Run the M7 bounded dual-camera USB bandwidth preflight.

The tool deliberately does not alter device options.  Each CameraRig session is
constructed, opened, captured, and closed by the same non-daemon worker thread.
Serial numbers stay in the private CameraRig YAML files and are never copied to
the report.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
import time
import traceback
from typing import Any

from camera_rig.api import CameraSession, load_camera_config


REQUIRED_STREAMS = ("color", "depth", "ir_left", "ir_right")


@dataclass
class WorkerResult:
    camera_name: str
    requested_frames: int
    captured: int = 0
    missing_required_streams: Counter[str] = field(default_factory=Counter)
    host_timestamps_ns: list[int] = field(default_factory=list)
    per_stream_frame_numbers: dict[str, list[int]] = field(default_factory=dict)
    fatal_error: str | None = None
    traceback_path: str | None = None
    session_closed: bool = False

    def report(self) -> dict[str, Any]:
        elapsed_ns = (
            self.host_timestamps_ns[-1] - self.host_timestamps_ns[0]
            if len(self.host_timestamps_ns) > 1
            else 0
        )
        fps = (
            (len(self.host_timestamps_ns) - 1) * 1_000_000_000.0 / elapsed_ns
            if elapsed_ns > 0
            else 0.0
        )
        monotonic = all(
            current > previous
            for previous, current in zip(
                self.host_timestamps_ns, self.host_timestamps_ns[1:], strict=False
            )
        )
        jumps: dict[str, dict[str, int]] = {}
        for name, values in self.per_stream_frame_numbers.items():
            histogram = Counter(
                str(current - previous)
                for previous, current in zip(values, values[1:], strict=False)
                if current != previous + 1
            )
            jumps[name] = dict(sorted(histogram.items(), key=lambda item: int(item[0])))
        passed = (
            self.captured >= self.requested_frames
            and fps >= 27.0
            and not self.missing_required_streams
            and monotonic
            and self.fatal_error is None
            and self.session_closed
        )
        return {
            "camera_name": self.camera_name,
            "requested_frames": self.requested_frames,
            "captured_frames": self.captured,
            "capture_fps": fps,
            "required_stream_missing": dict(self.missing_required_streams),
            "host_timestamp_monotonic": monotonic,
            "frame_number_jump_histogram": jumps,
            "fatal_error": self.fatal_error,
            "traceback_path": self.traceback_path,
            "session_closed": self.session_closed,
            "status": "PASS" if passed else "FAIL",
        }


def _worker(
    camera_name: str,
    config_path: Path,
    requested_frames: int,
    barrier: threading.Barrier,
    stop_event: threading.Event,
    traceback_dir: Path,
    result: WorkerResult,
) -> None:
    session: CameraSession | None = None
    try:
        config = load_camera_config(config_path)
        if config.camera.name != camera_name:
            raise ValueError(
                f"camera name mismatch: argument={camera_name!r}, config={config.camera.name!r}"
            )
        session = CameraSession.from_config(config)
        session.open()
        barrier.wait(timeout=30.0)
        for _ in range(requested_frames):
            if stop_event.is_set():
                break
            frame = session.capture()
            result.captured += 1
            result.host_timestamps_ns.append(int(frame.host_receive_timestamp_ns))
            for stream_name in REQUIRED_STREAMS:
                stream = frame.streams.get(stream_name)
                if stream is None:
                    result.missing_required_streams[stream_name] += 1
                else:
                    result.per_stream_frame_numbers.setdefault(stream_name, []).append(
                        int(stream.frame_number)
                    )
    except Exception as exc:  # hardware boundary: preserve the complete traceback privately
        result.fatal_error = f"{type(exc).__name__}: {exc}"
        traceback_dir.mkdir(parents=True, exist_ok=True)
        path = traceback_dir / f"m7-raw-preflight-{camera_name}.traceback.txt"
        path.write_text(traceback.format_exc(), encoding="utf-8")
        result.traceback_path = str(path)
        stop_event.set()
        try:
            barrier.abort()
        except Exception:
            pass
    finally:
        if session is not None:
            try:
                session.close()
                result.session_closed = True
            except Exception as exc:
                result.fatal_error = result.fatal_error or f"close {type(exc).__name__}: {exc}"
                stop_event.set()


def run_preflight(
    cameras: dict[str, Path],
    *,
    requested_frames: int,
    report_path: Path,
    traceback_dir: Path,
) -> dict[str, Any]:
    if len(cameras) < 2:
        raise ValueError("raw concurrent preflight requires at least two cameras")
    if requested_frames < 1:
        raise ValueError("requested_frames must be positive")
    barrier = threading.Barrier(len(cameras))
    stop_event = threading.Event()
    results = {
        name: WorkerResult(camera_name=name, requested_frames=requested_frames)
        for name in sorted(cameras)
    }
    threads = [
        threading.Thread(
            name=f"raw-preflight-{name}",
            target=_worker,
            args=(
                name,
                cameras[name],
                requested_frames,
                barrier,
                stop_event,
                traceback_dir,
                results[name],
            ),
            daemon=False,
        )
        for name in sorted(cameras)
    ]
    started_ns = time.monotonic_ns()
    for thread in threads:
        thread.start()
    join_timeout_s = max(30.0, requested_frames / 20.0 + 30.0)
    for thread in threads:
        thread.join(timeout=join_timeout_s)
    alive = [thread.name for thread in threads if thread.is_alive()]
    if alive:
        stop_event.set()
        try:
            barrier.abort()
        except Exception:
            pass
        for thread in threads:
            thread.join(timeout=5.0)
        alive = [thread.name for thread in threads if thread.is_alive()]
    camera_reports = {name: result.report() for name, result in results.items()}
    passed = not alive and all(
        item["status"] == "PASS" for item in camera_reports.values()
    )
    report = {
        "schema_version": "pointcloud-builder.m7-raw-concurrent-preflight.v1",
        "requested_frames_per_camera": requested_frames,
        "elapsed_s": (time.monotonic_ns() - started_ns) / 1_000_000_000.0,
        "stream_profile": {
            "color": "640x480 rgb8 30Hz",
            "depth": "640x480 z16 30Hz",
            "ir_left": "640x480 y8 30Hz",
            "ir_right": "640x480 y8 30Hz",
        },
        "cameras": camera_reports,
        "threads_alive_after_join": alive,
        "status": "PASS" if passed else "FAIL",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _parse_camera(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("camera must be NAME=CONFIG_PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("camera must be NAME=CONFIG_PATH")
    return name, Path(path).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", action="append", required=True, type=_parse_camera)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--traceback-dir", type=Path, required=True)
    args = parser.parse_args()
    cameras = dict(args.camera)
    if len(cameras) != len(args.camera):
        raise SystemExit("duplicate camera names are not allowed")
    report = run_preflight(
        cameras,
        requested_frames=args.frames,
        report_path=args.report,
        traceback_dir=args.traceback_dir,
    )
    safe_summary = {
        "status": report["status"],
        "elapsed_s": report["elapsed_s"],
        "cameras": {
            name: {
                "captured_frames": item["captured_frames"],
                "capture_fps": item["capture_fps"],
                "host_timestamp_monotonic": item["host_timestamp_monotonic"],
                "missing_required_streams": item["required_stream_missing"],
                "fatal_error": item["fatal_error"],
                "session_closed": item["session_closed"],
            }
            for name, item in report["cameras"].items()
        },
    }
    print(json.dumps(safe_summary, indent=2))
    if report["status"] != "PASS":
        raise SystemExit("raw concurrent preflight failed")


if __name__ == "__main__":
    main()
