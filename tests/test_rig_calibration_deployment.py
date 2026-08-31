from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pointcloud_builder.integrations.camera_rig.types import FrameExplicitTransform
from pointcloud_builder.rig.config import parse_rig_config
from pointcloud_builder.rig_calibration.artifact import (
    solution_fingerprint,
    write_solution,
)
from pointcloud_builder.rig_calibration.deployment import (
    PHYSICAL_ACCEPTANCE_SCHEMA_VERSION,
    ResolvedRigCalibrationDeployment,
    apply_deployment_to_context,
    load_rig_calibration_deployment,
    promote_rig_calibration,
)
from pointcloud_builder.rig_calibration.solver import solve_rig_calibration
from pointcloud_builder.rig_calibration.validation import (
    validate_rig_calibration_solution,
)
from tests.rig_calibration_synthetic import make_scene


def _inputs(tmp_path: Path):
    data, _truth, _poses = make_scene(
        noise_px=0.1, holdout_pose_ids={"pose_9", "pose_11"}
    )
    hashes = {
        camera_id: (str(index + 1) * 64)
        for index, camera_id in enumerate(data.camera_ids)
    }
    data = replace(data, camera_bundle_hashes=hashes)
    solution = solve_rig_calibration(data)
    solution_path = tmp_path / "solution.json"
    validation_path = tmp_path / "validation.json"
    physical_path = tmp_path / "physical.json"
    write_solution(solution, solution_path)
    validation = validate_rig_calibration_solution(solution, data)
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    physical = {
        "schema_version": PHYSICAL_ACCEPTANCE_SCHEMA_VERSION,
        "status": "PASS",
        "passed": True,
        "solution_fingerprint": solution_fingerprint(solution),
        "workspace_frame": solution.workspace_frame,
        "target_identity": solution.target_identity,
        "camera_set": sorted(solution.T_workspace_from_camera),
        "camera_bundle_hashes": solution.camera_bundle_hashes,
        "camera_identities": solution.camera_identities,
        "per_pair": {
            "camera_a__camera_b": {
                "status": "PASS",
                "overlap_point_count": 1000,
                "symmetric_nn": {"median_mm": 0.5, "p95_mm": 1.0},
                "diagnostic_residual_se3": {
                    "diagnostic_only": True,
                    "written_back": False,
                },
                "gates": {"pairwise_geometry": True},
            }
        },
        "thresholds": {"maximum_symmetric_median_mm": 1.0},
        "gates": {"pairwise_geometry": True, "connected_overlap_graph": True},
        "diagnostic_residual_writeback": False,
    }
    physical_path.write_text(json.dumps(physical), encoding="utf-8")
    return solution, validation, physical, solution_path, validation_path, physical_path


def test_valid_promotion_round_trip(tmp_path: Path) -> None:
    solution, _validation, _physical, solution_path, validation_path, physical_path = (
        _inputs(tmp_path)
    )
    output = tmp_path / "deployment.json"
    promoted = promote_rig_calibration(
        solution_path, validation_path, physical_path, output
    )
    deployment = load_rig_calibration_deployment(output)

    assert promoted["status"] == "deployed"
    assert deployment.solution_fingerprint == solution_fingerprint(solution)
    assert deployment.camera_ids == tuple(sorted(solution.T_workspace_from_camera))
    assert deployment.provenance["calibration_mode"] == (
        "validated_multipose_deployment"
    )
    assert deployment.provenance["production_applied"] is True


@pytest.mark.parametrize(
    ("artifact", "mutate", "message"),
    [
        ("validation", lambda value: value.update(solution_fingerprint="0" * 64), "exact solution"),
        ("validation", lambda value: value.update(status="FAIL", passed=False), "must PASS"),
        ("validation", lambda value: value["holdout"].update(status="NOT_RUN"), "passed holdout"),
        ("validation", lambda value: value.update(workspace_frame="wrong"), "workspace_frame"),
        ("validation", lambda value: value.update(target_identity={"wrong": True}), "target_identity"),
        ("validation", lambda value: value["camera_bundle_hashes"].update(camera_a="f" * 64), "camera_bundle_hashes"),
        ("physical", lambda value: value.update(solution_fingerprint="0" * 64), "exact solution"),
        ("physical", lambda value: value.update(status="FAIL", passed=False), "must PASS"),
        ("physical", lambda value: value.update(workspace_frame="wrong"), "workspace_frame"),
        ("physical", lambda value: value.update(target_identity={"wrong": True}), "target_identity"),
        ("physical", lambda value: value.update(camera_set=["camera_a"]), "camera_set"),
        ("physical", lambda value: value["camera_bundle_hashes"].update(camera_b="f" * 64), "camera_bundle_hashes"),
        ("physical", lambda value: value.update(per_pair={}), "per_pair"),
        ("physical", lambda value: value["gates"].update(pairwise_geometry=False), "all_gates"),
    ],
)
def test_promotion_fails_closed_for_mismatches(
    tmp_path: Path, artifact: str, mutate, message: str
) -> None:
    _solution, validation, physical, solution_path, validation_path, physical_path = (
        _inputs(tmp_path)
    )
    selected = validation if artifact == "validation" else physical
    mutate(selected)
    (validation_path if artifact == "validation" else physical_path).write_text(
        json.dumps(selected), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=message):
        promote_rig_calibration(
            solution_path, validation_path, physical_path, tmp_path / "output.json"
        )


def test_promotion_requires_physical_acceptance(tmp_path: Path) -> None:
    _solution, _validation, _physical, solution_path, validation_path, _physical_path = (
        _inputs(tmp_path)
    )
    with pytest.raises(FileNotFoundError):
        promote_rig_calibration(
            solution_path,
            validation_path,
            tmp_path / "missing.json",
            tmp_path / "output.json",
        )


def test_deployment_fingerprint_is_tamper_evident(tmp_path: Path) -> None:
    _solution, _validation, _physical, solution_path, validation_path, physical_path = (
        _inputs(tmp_path)
    )
    output = tmp_path / "deployment.json"
    promote_rig_calibration(solution_path, validation_path, physical_path, output)
    raw = json.loads(output.read_text())
    raw["workspace_frame"] = "tampered"
    output.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_rig_calibration_deployment(output)


def test_color_to_ir_composition_uses_frame_explicit_camera_rig_edge() -> None:
    T_workspace_from_color = np.eye(4)
    T_workspace_from_color[:3, 3] = (0.4, -0.2, 0.1)
    T_color_from_ir = np.eye(4)
    T_color_from_ir[:3, 3] = (0.025, 0.001, -0.002)
    deployment = ResolvedRigCalibrationDeployment(
        artifact_path=Path("deployment.json"),
        artifact_fingerprint="a" * 64,
        workspace_frame="workspace",
        solution_fingerprint="b" * 64,
        target_identity={"kind": "synthetic"},
        per_camera={
            "left_cam": {
                "camera_bundle_sha256": "c" * 64,
                "projection_frame": "left_cam/color_optical",
                "T_workspace_from_camera": T_workspace_from_color,
            }
        },
    )

    class Calibration:
        bundle = SimpleNamespace(to_dict=lambda: {"bundle": "synthetic"})

        @staticmethod
        def transform(source: str, target: str) -> FrameExplicitTransform:
            assert source == "left_cam/ir_left_optical"
            assert target == "left_cam/color_optical"
            return FrameExplicitTransform(source, target, T_color_from_ir)

    @dataclass(frozen=True)
    class Context:
        calibration: object
        source_frame: str
        workspace_frame: str
        T_workspace_from_source: object

    context = Context(
        calibration=Calibration(),
        source_frame="left_cam/ir_left_optical",
        workspace_frame="old_workspace",
        T_workspace_from_source=None,
    )
    updated, provenance = apply_deployment_to_context(
        context, "left_cam", deployment
    )
    np.testing.assert_allclose(
        updated.T_workspace_from_source.matrix,
        T_workspace_from_color @ T_color_from_ir,
    )
    assert updated.T_workspace_from_source.source_frame == (
        "left_cam/ir_left_optical"
    )
    assert updated.T_workspace_from_source.target_frame == "workspace"
    assert provenance["production_applied"] is True


def test_rig_config_deployment_is_optional_and_path_resolved(tmp_path: Path) -> None:
    raw = {
        "schema_version": "pointcloud-builder.rig.v1",
        "output_frame": "workspace",
        "cameras": [
            {
                "name": name,
                "enabled": True,
                "source": {
                    "type": "synthetic",
                    "capture_artifact": f"synthetic://{name}",
                    "provision_artifact": f"synthetic://{name}/bundle",
                },
                "depth": {"mode": "native"},
            }
            for name in ("left_cam", "right_cam", "top_cam")
        ],
        "timing": {"mode": "exact_index", "maximum_skew_ms": 0.0},
        "workspace_crop": {"enabled": False},
        "fusion": {"enabled": False, "voxel_size_m": 0.01},
        "sampling": {"enabled": False, "mode": "stride", "num_points": 1},
        "rig_calibration": {
            "type": "validated_multipose",
            "artifact": "private/deployment.json",
        },
    }
    configured = parse_rig_config(raw, base_dir=tmp_path)
    assert configured.rig_calibration is not None
    assert configured.rig_calibration.artifact == str(
        (tmp_path / "private/deployment.json").resolve()
    )
    raw.pop("rig_calibration")
    assert parse_rig_config(raw, base_dir=tmp_path).rig_calibration is None
