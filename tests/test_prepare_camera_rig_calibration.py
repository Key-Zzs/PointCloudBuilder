from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
SCRIPT_PATH = ROOT / "scripts/prepare_camera_rig_calibration.py"
CAMERA_RIG_SRC = ROOT / "third_party/CameraRig/src"
sys.path.insert(0, str(CAMERA_RIG_SRC))
from camera_rig.provision.config import load_provision_config  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "prepare_camera_rig_calibration", SCRIPT_PATH
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_identity(path: Path, serials: tuple[str, ...] = ("A", "B", "C")) -> None:
    value = {
        "schema_version": "pointcloud-builder.camera-identity-map.v1",
        **{
            f"camera_{chr(97 + index)}": {
                "serial": serial,
                "physical_port": f"port-{index}",
            }
            for index, serial in enumerate(serials)
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_target(root: Path) -> Path:
    root.mkdir(parents=True)
    target = root / "target_spec.json"
    target.write_text(
        json.dumps(
            {
                "schema_version": "camera-rig.target.charuco-resolved.v1",
                "target_name": "charuco_a4_v1",
                "target_frame": "charuco_target",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    for name in (
        "charuco_a4_v1_board.png",
        "charuco_a4_v1_print.pdf",
        "charuco_a4_v1_preview.png",
    ):
        (root / name).write_bytes(name.encode("utf-8"))
    report = root / "generation_report.json"
    report.write_text(
        json.dumps({"status": "PASS", "target_spec_sha256": _sha256(target)}),
        encoding="utf-8",
    )
    payloads = sorted(path for path in root.iterdir() if path.is_file())
    (root / "checksums.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in payloads),
        encoding="utf-8",
    )
    return target


def _arguments(tmp_path: Path, *, extra: tuple[str, ...] = ()) -> list[str]:
    identity = tmp_path / "camera_rig/camera_identity_map.json"
    target = tmp_path / "camera_rig/shared_target/charuco_a4_v1/target_spec.json"
    return [
        "--identity-map",
        str(identity),
        "--target",
        str(target),
        "--asset-root",
        str(tmp_path / "camera_rig"),
        "--expected-camera-count",
        "3",
        "--workspace-equals-target",
        "--report",
        str(tmp_path / "reports/preparation.json"),
        *extra,
    ]


def test_prepare_creates_private_configs_without_reporting_serials(
    tmp_path: Path, capsys
) -> None:
    identity = tmp_path / "camera_rig/camera_identity_map.json"
    _write_identity(identity)
    target = _write_target(tmp_path / "camera_rig/shared_target/charuco_a4_v1")

    assert MODULE.main(_arguments(tmp_path)) == 0

    runtime_path = tmp_path / "camera_rig/camera_a/configs/runtime.yaml"
    provision_path = tmp_path / "camera_rig/camera_a/configs/fixed_provision.yaml"
    runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    provision = yaml.safe_load(provision_path.read_text(encoding="utf-8"))
    assert runtime["camera"]["name"] == "camera_a"
    assert runtime["camera"]["serial"] == "A"
    assert provision["camera"]["name"] == "camera_a"
    assert provision["camera"]["serial"] == "A"
    assert provision["target"] == {
        "artifact": "../../shared_target/charuco_a4_v1/target_spec.json",
        "expected_sha256": _sha256(target),
        "detection_stream": "color",
        "detection_policy": "pose_validated",
    }
    assert provision["workspace"]["T_workspace_from_target"]["matrix"] == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    assert runtime_path.stat().st_mode & 0o777 == 0o600
    assert provision_path.stat().st_mode & 0o777 == 0o600
    strict_config = load_provision_config(provision_path)
    assert strict_config.camera_config.camera.name == "camera_a"
    assert strict_config.target.detection_policy == "pose_validated"
    assert strict_config.target.artifact_path == target.resolve()
    public_output = capsys.readouterr().out
    report_text = (tmp_path / "reports/preparation.json").read_text(encoding="utf-8")
    assert '"A"' not in public_output
    assert '"A"' not in report_text
    assert "camera_a" in public_output


def test_prepare_is_idempotent_and_check_mode_is_read_only(tmp_path: Path) -> None:
    _write_identity(tmp_path / "camera_rig/camera_identity_map.json")
    _write_target(tmp_path / "camera_rig/shared_target/charuco_a4_v1")
    assert MODULE.main(_arguments(tmp_path)) == 0
    runtime = tmp_path / "camera_rig/camera_a/configs/runtime.yaml"
    before = runtime.read_bytes()

    assert MODULE.main(_arguments(tmp_path, extra=("--check",))) == 0

    assert runtime.read_bytes() == before
    report = json.loads(
        (tmp_path / "reports/preparation.json").read_text(encoding="utf-8")
    )
    assert report["mode"] == "full_check"
    assert all(
        config["status"] == "UNCHANGED"
        for camera in report["prepared"].values()
        for config in camera.values()
    )


def test_prepare_rejects_conflict_then_backs_up_explicit_update(tmp_path: Path) -> None:
    _write_identity(tmp_path / "camera_rig/camera_identity_map.json")
    _write_target(tmp_path / "camera_rig/shared_target/charuco_a4_v1")
    assert MODULE.main(_arguments(tmp_path)) == 0
    runtime_path = tmp_path / "camera_rig/camera_a/configs/runtime.yaml"
    runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    runtime["camera"]["name"] = "wrong_name"
    runtime_path.write_text(yaml.safe_dump(runtime), encoding="utf-8")

    assert MODULE.main(_arguments(tmp_path)) == 2
    assert MODULE.main(_arguments(tmp_path, extra=("--update-existing",))) == 0

    updated = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    assert updated["camera"]["name"] == "camera_a"
    assert len(list(runtime_path.parent.glob("runtime.yaml.bak-*"))) == 1


def test_prepare_plans_all_cameras_before_writing(tmp_path: Path) -> None:
    _write_identity(tmp_path / "camera_rig/camera_identity_map.json")
    _write_target(tmp_path / "camera_rig/shared_target/charuco_a4_v1")
    conflicting = tmp_path / "camera_rig/camera_b/configs/runtime.yaml"
    conflicting.parent.mkdir(parents=True)
    conflicting.write_text("camera: {}\n", encoding="utf-8")

    assert MODULE.main(_arguments(tmp_path)) == 2

    assert not (tmp_path / "camera_rig/camera_a/configs/runtime.yaml").exists()
    assert conflicting.read_text(encoding="utf-8") == "camera: {}\n"


def test_prepare_fails_closed_for_wrong_count_and_bad_target(tmp_path: Path) -> None:
    identity = tmp_path / "camera_rig/camera_identity_map.json"
    _write_identity(identity, ("A", "B"))
    target = _write_target(tmp_path / "camera_rig/shared_target/charuco_a4_v1")
    assert MODULE.main(_arguments(tmp_path)) == 2

    _write_identity(identity)
    target.write_text("{}", encoding="utf-8")
    assert MODULE.main(_arguments(tmp_path)) == 2
    assert not (tmp_path / "camera_rig/camera_a/configs/runtime.yaml").exists()


def test_prepare_requires_explicit_workspace_equals_target(tmp_path: Path) -> None:
    _write_identity(tmp_path / "camera_rig/camera_identity_map.json")
    _write_target(tmp_path / "camera_rig/shared_target/charuco_a4_v1")
    args = _arguments(tmp_path)
    args.remove("--workspace-equals-target")
    assert MODULE.main(args) == 2
    assert not (tmp_path / "camera_rig/camera_a/configs/runtime.yaml").exists()


def test_runtime_only_breaks_existing_target_capture_dependency(
    tmp_path: Path, capsys
) -> None:
    identity = tmp_path / "camera_rig/camera_identity_map.json"
    _write_identity(identity)
    args = [
        "--identity-map",
        str(identity),
        "--asset-root",
        str(tmp_path / "camera_rig"),
        "--expected-camera-count",
        "3",
        "--runtime-only",
        "--report",
        str(tmp_path / "reports/runtime-only.json"),
    ]

    assert MODULE.main(args) == 0

    assert (tmp_path / "camera_rig/camera_a/configs/runtime.yaml").is_file()
    assert not (tmp_path / "camera_rig/camera_a/configs/fixed_provision.yaml").exists()
    report = json.loads(
        (tmp_path / "reports/runtime-only.json").read_text(encoding="utf-8")
    )
    assert report["mode"] == "runtime_only_prepare"
    assert report["target_spec_sha256"] is None
    assert "target_spec_sha256" not in capsys.readouterr().out
