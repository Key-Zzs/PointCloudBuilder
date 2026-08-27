#!/usr/bin/env python3
"""Fail-closed reconstruction environment and hardware doctor."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Callable


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: dict[str, Any]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-hardware", action="store_true")
    parser.add_argument("--expected-env", default="pcb-reconstruction")
    parser.add_argument("--asset-root", default=".local/ffs")
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    checks: list[Check] = []
    checks.append(
        _check(
            "python",
            platform.python_version(),
            (sys.version_info.major, sys.version_info.minor) == (3, 10),
        )
    )
    active_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    checks.append(
        _check(
            "conda_environment",
            active_env or "not-active",
            active_env == args.expected_env,
            warning=args.no_hardware,
        )
    )
    checks.extend(_core_import_checks(no_hardware=args.no_hardware))
    checks.append(_opencv_check(no_hardware=args.no_hardware))
    checks.extend(_torch_checks(no_hardware=args.no_hardware))
    checks.extend(_ffs_asset_checks(Path(args.asset_root), warning=args.no_hardware))
    if args.no_hardware:
        checks.append(Check("d435i_devices", "WARNING", {"reason": "skipped"}))
        checks.append(Check("usb3_links", "WARNING", {"reason": "skipped"}))
    else:
        checks.extend(_device_checks())
    report = {
        "schema_version": "pointcloud-builder.reconstruction-doctor.v1",
        "mode": "no_hardware" if args.no_hardware else "full",
        "checks": [asdict(item) for item in checks],
        "summary": {
            status: sum(item.status == status for item in checks)
            for status in ("PASS", "WARNING", "FAIL")
        },
        "passed": all(item.status != "FAIL" for item in checks),
    }
    if args.report:
        destination = _private_output(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def _core_import_checks(*, no_hardware: bool) -> list[Check]:
    specifications: tuple[tuple[str, str, str, bool], ...] = (
        ("numpy", "numpy", "numpy", False),
        ("yaml", "PyYAML", "yaml", False),
        ("camera_rig", "camera-rig", "camera_rig", False),
        ("tensorrt", "tensorrt-cu13", "tensorrt", True),
        ("pyrealsense2", "pyrealsense2", "pyrealsense2", True),
        ("open3d", "open3d", "open3d", True),
        ("rerun", "rerun-sdk", "rerun", True),
    )
    checks = []
    for name, distribution, module, heavy in specifications:
        try:
            imported = __import__(module)
            version = _version(distribution, imported)
        except Exception as error:
            checks.append(
                Check(
                    name,
                    "WARNING" if no_hardware and heavy else "FAIL",
                    {"error": f"{type(error).__name__}: {str(error)[:300]}"},
                )
            )
        else:
            checks.append(Check(name, "PASS", {"version": version}))
    return checks


def _opencv_check(*, no_hardware: bool) -> Check:
    try:
        import cv2

        contrib = metadata.version("opencv-contrib-python-headless")
        conflicts = []
        for name in ("opencv-python", "opencv-python-headless"):
            try:
                metadata.version(name)
            except metadata.PackageNotFoundError:
                pass
            else:
                conflicts.append(name)
        passed = hasattr(cv2, "aruco") and not conflicts
        return Check(
            "opencv",
            "PASS" if passed else "FAIL",
            {
                "cv2_version": cv2.__version__,
                "distribution": "opencv-contrib-python-headless",
                "distribution_version": contrib,
                "aruco": hasattr(cv2, "aruco"),
                "conflicting_wheels": conflicts,
            },
        )
    except Exception as error:
        return Check(
            "opencv",
            "WARNING" if no_hardware else "FAIL",
            {"error": f"{type(error).__name__}: {str(error)[:300]}"},
        )


def _torch_checks(*, no_hardware: bool) -> list[Check]:
    try:
        import torch
    except Exception as error:
        return [
            Check(
                "torch",
                "WARNING" if no_hardware else "FAIL",
                {"error": f"{type(error).__name__}: {str(error)[:300]}"},
            )
        ]
    checks = [
        Check(
            "torch",
            "PASS",
            {"version": torch.__version__, "cuda_version": torch.version.cuda},
        )
    ]
    available = bool(torch.cuda.is_available())
    checks.append(
        Check(
            "cuda",
            "PASS" if available else ("WARNING" if no_hardware else "FAIL"),
            {
                "available": available,
                "device_count": torch.cuda.device_count() if available else 0,
                "gpu": torch.cuda.get_device_name(0) if available else None,
            },
        )
    )
    return checks


def _ffs_asset_checks(root: Path, *, warning: bool) -> list[Check]:
    artifacts = root / "artifacts"
    paths = {
        "ffs_checkpoint": artifacts / "model_best_bp2_serialize.pth",
        "ffs_model_config": artifacts / "cfg.yaml",
        "ffs_plugin_engine": artifacts / "tensorrt_plugin_fp16_o3.engine",
        "ffs_plugin_manifest": artifacts / "tensorrt_plugin_fp16_o3.manifest.json",
        "ffs_plugin_library": root / "build/libffs_gwc_plugin.so",
    }
    checks = []
    for name, path in paths.items():
        exists = path.is_file() and path.stat().st_size > 0
        checks.append(
            Check(
                name,
                "PASS" if exists else ("WARNING" if warning else "FAIL"),
                {"present": exists, "expected_name": path.name},
            )
        )
    if paths["ffs_checkpoint"].is_file():
        from pointcloud_builder.ffs.manifest import sha256_file

        expected = "98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692"
        actual = sha256_file(paths["ffs_checkpoint"])
        checks.append(
            Check(
                "ffs_checkpoint_sha256",
                "PASS" if actual == expected else "FAIL",
                {"matches_expected": actual == expected},
            )
        )
    return checks


def _device_checks() -> list[Check]:
    try:
        import pyrealsense2 as rs

        devices = list(rs.context().query_devices())
        d435i = [
            item
            for item in devices
            if "D435" in item.get_info(rs.camera_info.name).upper()
        ]
        usb = []
        for item in d435i:
            try:
                usb.append(item.get_info(rs.camera_info.usb_type_descriptor))
            except Exception:
                usb.append("unknown")
        return [
            Check(
                "d435i_devices",
                "PASS" if len(d435i) >= 2 else "FAIL",
                {"count": len(d435i)},
            ),
            Check(
                "usb3_links",
                "PASS" if len(usb) >= 2 and all(v.startswith("3") for v in usb) else "FAIL",
                {"descriptors": usb},
            ),
        ]
    except Exception as error:
        detail = {"error": f"{type(error).__name__}: {str(error)[:300]}"}
        return [Check("d435i_devices", "FAIL", detail), Check("usb3_links", "FAIL", detail)]


def _check(name: str, value: str, passed: bool, *, warning: bool = False) -> Check:
    return Check(name, "PASS" if passed else ("WARNING" if warning else "FAIL"), {"value": value})


def _version(distribution: str, imported: Any) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return str(getattr(imported, "__version__", "unknown"))


def _private_output(value: str) -> Path:
    output = Path(value).resolve()
    if not output.is_relative_to((Path.cwd() / ".local").resolve()):
        raise ValueError("doctor reports must be written under .local/")
    return output


if __name__ == "__main__":
    raise SystemExit(main())
