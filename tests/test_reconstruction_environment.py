from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
DOCTOR_PATH = ROOT / "scripts/doctor_reconstruction_env.py"
SPEC = importlib.util.spec_from_file_location("doctor_reconstruction_env", DOCTOR_PATH)
assert SPEC and SPEC.loader
DOCTOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DOCTOR
SPEC.loader.exec_module(DOCTOR)


def test_reconstruction_environment_pins_one_opencv_provider_and_critical_abi() -> None:
    raw = yaml.safe_load(
        (ROOT / "environment.reconstruction.yml").read_text(encoding="utf-8")
    )
    pip = next(item["pip"] for item in raw["dependencies"] if isinstance(item, dict))
    assert "opencv-contrib-python-headless==4.14.0.94" in pip
    assert not any(
        item.startswith("opencv-python-headless") or item.startswith("opencv-python==")
        for item in pip
    )
    assert "torch==2.11.0" in pip
    assert "tensorrt-cu13==10.16.1.11" in pip
    assert "pyrealsense2==2.58.3.10794" in pip
    assert "open3d==0.19.0" in pip
    assert "rerun-sdk==0.36.3" in pip
    assert "omegaconf==2.3.0" in pip


def test_doctor_never_reads_or_reports_realsense_serial_numbers() -> None:
    text = DOCTOR_PATH.read_text(encoding="utf-8")
    assert "serial_number" not in text
    assert "camera_info.serial" not in text


def test_no_hardware_doctor_treats_absent_private_assets_as_warnings(tmp_path: Path) -> None:
    checks = DOCTOR._ffs_asset_checks(tmp_path, warning=True)
    assert checks
    assert all(item.status == "WARNING" for item in checks)


def test_doctor_expected_camera_count_is_configurable_and_private(capsys) -> None:
    status = DOCTOR.main(
        [
            "--no-hardware",
            "--expected-env",
            "not-active",
            "--expected-d435i-count",
            "3",
            "--asset-root",
            "/missing/private/assets",
        ]
    )
    report = __import__("json").loads(capsys.readouterr().out)
    assert status == 0
    devices = next(
        item for item in report["checks"] if item["name"] == "d435i_devices"
    )
    assert devices["detail"]["expected_count"] == 3


def test_full_doctor_fails_closed_for_absent_private_assets(tmp_path: Path) -> None:
    checks = DOCTOR._ffs_asset_checks(tmp_path, warning=False)
    assert checks
    assert all(item.status == "FAIL" for item in checks)


def test_bootstrap_is_scoped_to_named_env_and_never_installs_into_base() -> None:
    text = (ROOT / "scripts/bootstrap_reconstruction_env.sh").read_text(
        encoding="utf-8"
    )
    assert "PCB_ENV_NAME" in text
    assert 'run --name "${ENV_NAME}" python -m pip install' in text
    assert "pip install" not in "\n".join(
        line for line in text.splitlines() if "ENV_NAME" not in line and "REPO_ROOT" not in line
    )
