from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest

from pointcloud_builder.camera_model import CameraIntrinsics
from pointcloud_builder.integrations.camera_rig.types import FrameExplicitTransform
from pointcloud_builder.mapping.recording import RigDepthRecordingWriter
from pointcloud_builder.mapping.types import RigDepthFrameSet, RigDepthObservation
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
    sha256_file,
)
from pointcloud_builder.rig_calibration.intrinsic_health import (
    evaluate_rig_intrinsic_health,
)
from pointcloud_builder.rig_calibration.physical_acceptance import (
    REAL_DUAL_MULTIPOSE_V1_THRESHOLDS,
)
from pointcloud_builder.rig_calibration.solver import solve_rig_calibration
from pointcloud_builder.rig_calibration.validation import (
    validate_rig_calibration_solution,
)
from tests.rig_calibration_synthetic import diverse_target_poses, make_scene

_RECORDING_TEMP = TemporaryDirectory(prefix="pcb-physical-recording-test-")


@lru_cache(maxsize=1)
def _physical_recording_manifest(
    camera_hash_items: tuple[tuple[str, str], ...], workspace_frame: str
) -> Path:
    camera_hashes = dict(camera_hash_items)
    camera_ids = tuple(sorted(camera_hashes))
    backend = {
        camera_id: {
            "depth_source": "ffs_stereo",
            "backend": "tensorrt_plugin",
            "precision": "fp16",
            "artifact_id": f"synthetic-{camera_id}",
            "pipeline_config_sha256": "9" * 64,
        }
        for camera_id in camera_ids
    }
    depth = np.asarray([[0.75]], dtype=np.float32)
    observations = tuple(
        RigDepthObservation(
            camera_name=camera_id,
            depth=depth,
            depth_unit="meters",
            depth_scale_m_per_unit=1.0,
            valid_mask=np.asarray([[True]], dtype=bool),
            intrinsics=CameraIntrinsics(
                width=1,
                height=1,
                fx=1.0,
                fy=1.0,
                cx=0.0,
                cy=0.0,
                frame=f"{camera_id}/ir_left_optical",
            ),
            T_workspace_from_camera=np.eye(4),
            timestamp_ns=0,
            depth_source="ffs_stereo",
            source_frame=f"{camera_id}/ir_left_optical",
            workspace_frame=workspace_frame,
            bundle_identity=f"synthetic-{camera_id}",
            provision_sha256="8" * 64,
            distortion_model="none",
            distortion_coeffs=(),
            rectified=True,
            camera_bundle_sha256=camera_hashes[camera_id],
        )
        for camera_id in camera_ids
    )
    root = Path(_RECORDING_TEMP.name) / "recording"
    writer = RigDepthRecordingWriter(
        root,
        depth_source="ffs_stereo",
        backend_provenance=backend,
        calibration_purpose="physical_acceptance_candidate",
    )
    for index in range(300):
        writer.append(
            RigDepthFrameSet(
                matched_set_index=index,
                host_timestamp_ns=index,
                maximum_skew_ms=0.0,
                observations=observations,
            )
        )
    writer.finalize(report={"synthetic": True})
    return root / "manifest.json"


@lru_cache(maxsize=1)
def _base_inputs():
    base_poses = diverse_target_poses()
    poses = {}
    for index in range(30):
        matrix = base_poses[f"pose_{index % 12}"].copy()
        matrix[2, 3] += 0.001 * (index // 12)
        poses[f"pose_{index}"] = matrix
    data, _truth, _poses = make_scene(
        camera_ids=("camera_a", "camera_b", "camera_c"),
        poses=poses,
        noise_px=0.1,
        holdout_pose_ids={f"pose_{index}" for index in range(24, 30)},
    )
    hashes = {
        camera_id: (str(index + 1) * 64)
        for index, camera_id in enumerate(data.camera_ids)
    }
    bootstrap = {
        camera_id: {
            "schema_version": "camera-rig.calibration-authority.v1",
            "qualification_scope": "bootstrap_only",
            "production_authoritative": False,
            "qualification_state": "BOOTSTRAP_QUALIFIED",
            "qualification_fingerprint": str(index + 3) * 64,
            "target_metrology_sha256": str(index + 5) * 64,
            "metric_depth_receipt_sha256": str(index + 7) * 64,
        }
        for index, camera_id in enumerate(data.camera_ids)
    }
    data = replace(
        data,
        camera_bundle_hashes=hashes,
        bootstrap_qualifications=bootstrap,
        pose_plan_sha256="e" * 64,
        pose_plan_summary={
            "pose_ids": [f"pose_{index}" for index in range(30)],
            "solve_pose_ids": [f"pose_{index}" for index in range(24)],
            "holdout_pose_ids": [f"pose_{index}" for index in range(24, 30)],
            "capture_complete": True,
            "per_pose_camera_ids": {
                f"pose_{index}": ["camera_a", "camera_b", "camera_c"]
                for index in range(30)
            },
        },
    )
    solution = solve_rig_calibration(data, observations_sha256="f" * 64)
    validation = validate_rig_calibration_solution(solution, data)
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
            pair: {
                "status": "PASS",
                "cameras": pair.split("__"),
                "evaluated_matched_set_count": 300,
                "overlap_point_count": {"minimum_per_direction": 100_000},
                "symmetric_nn": {
                    "median_of_matched_set_medians_mm": 0.5,
                    "p95_of_matched_set_p95_mm": 1.0,
                },
                "board_interior": {
                    "status": "AVAILABLE",
                    "median_of_matched_set_medians_mm": 0.5,
                    "p95_of_matched_set_p95_mm": 1.0,
                },
                "board_plane": {
                    "absolute_signed_offset_p95_mm": 0.5,
                    "normal_split_p95_deg": 0.2,
                    "double_layer_thickness_p95_mm": 1.0,
                },
                "diagnostic_residual_se3": {
                    "translation_xyz_mm": [0.1, 0.0, 0.0],
                    "translation_norm_mm": 0.1,
                    "rotation_geodesic_deg": 0.1,
                    "aggregation": "aggregate_matched_set_point_to_plane_icp",
                    "diagnostic_only": True,
                    "written_back": False,
                },
                "gates": {
                    "minimum_overlap_points": True,
                    "symmetric_median": True,
                    "symmetric_p95": True,
                    "board_available": True,
                    "board_median": True,
                    "board_p95": True,
                    "plane_offset": True,
                    "normal_split": True,
                    "double_layer_thickness": True,
                    "diagnostic_translation": True,
                    "diagnostic_rotation": True,
                },
            }
            for pair in (
                "camera_a__camera_b",
                "camera_a__camera_c",
                "camera_b__camera_c",
            )
        },
        "all_rig": {
            "camera_graph_nodes": ["camera_a", "camera_b", "camera_c"],
            "accepted_pairwise_overlap_edges": [
                ["camera_a", "camera_b"],
                ["camera_a", "camera_c"],
                ["camera_b", "camera_c"],
            ],
            "pairwise_overlap_connectivity": {
                "connected": True,
                "accepted_edge_count": 3,
                "required_node_count": 3,
            },
            "per_camera_point_contribution": {
                camera_id: {"point_count": 100_000, "fraction": 1.0 / 3.0}
                for camera_id in ("camera_a", "camera_b", "camera_c")
            },
            "all_camera_input_point_count": 300_000,
            "camera_drop_statistics": {
                "camera_a": 0,
                "camera_b": 0,
                "camera_c": 0,
            },
            "matcher_statistics": {
                "available_complete_sets": 300,
                "evaluated_complete_sets": 300,
            },
        },
        "thresholds": asdict(REAL_DUAL_MULTIPOSE_V1_THRESHOLDS),
        "gates": {
            "all_pair_declarations_valid": True,
            "connected_overlap_graph": True,
        },
        "diagnostic_residual_writeback": False,
        "source_receipts": {},
        "calibration_mode": "validated_candidate",
        "production_applied": False,
        "rig_calibration_schema": None,
        "rig_calibration_fingerprint": None,
        "source_recording_manifest_sha256": "d" * 64,
        "selected_matched_set_indices": list(range(300)),
    }
    intrinsic = evaluate_rig_intrinsic_health(data, observations_sha256="f" * 64)
    return solution, validation, physical, intrinsic


def _inputs(tmp_path: Path):
    solution, validation_template, physical_template, intrinsic_template = (
        _base_inputs()
    )
    validation = deepcopy(validation_template)
    physical = deepcopy(physical_template)
    intrinsic = deepcopy(intrinsic_template)
    solution_path = tmp_path / "solution.json"
    validation_path = tmp_path / "validation.json"
    physical_path = tmp_path / "physical.json"
    intrinsic_path = tmp_path / "intrinsic.json"
    write_solution(solution, solution_path)
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    recording_manifest = _physical_recording_manifest(
        tuple(sorted(solution.camera_bundle_hashes.items())), solution.workspace_frame
    )
    recording_sha256 = sha256_file(recording_manifest)
    physical["source_recording_manifest_sha256"] = recording_sha256
    physical["source_receipts"] = {
        "solution": {"path": str(solution_path), "sha256": sha256_file(solution_path)},
        "validation": {
            "path": str(validation_path),
            "sha256": sha256_file(validation_path),
        },
        "source_recording_manifest": {
            "path": str(recording_manifest),
            "sha256": recording_sha256,
        },
    }
    physical_path.write_text(json.dumps(physical), encoding="utf-8")
    intrinsic_path.write_text(json.dumps(intrinsic), encoding="utf-8")
    return solution, validation, physical, solution_path, validation_path, physical_path


def test_valid_promotion_round_trip(tmp_path: Path) -> None:
    solution, _validation, _physical, solution_path, validation_path, physical_path = (
        _inputs(tmp_path)
    )
    output = tmp_path / "deployment.json"
    promoted = promote_rig_calibration(
        solution_path,
        validation_path,
        physical_path,
        output,
        intrinsic_health_path=physical_path.with_name("intrinsic.json"),
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
        (
            "validation",
            lambda value: value.update(solution_fingerprint="0" * 64),
            "exact solution",
        ),
        (
            "validation",
            lambda value: value.update(status="FAIL", passed=False),
            "must PASS",
        ),
        (
            "validation",
            lambda value: value["holdout"].update(status="NOT_RUN"),
            "passed holdout",
        ),
        (
            "validation",
            lambda value: value.update(workspace_frame="wrong"),
            "workspace_frame",
        ),
        (
            "validation",
            lambda value: value.update(target_identity={"wrong": True}),
            "target_identity",
        ),
        (
            "validation",
            lambda value: value["camera_bundle_hashes"].update(camera_a="f" * 64),
            "camera_bundle_hashes",
        ),
        (
            "physical",
            lambda value: value.update(solution_fingerprint="0" * 64),
            "exact solution",
        ),
        (
            "physical",
            lambda value: value.update(status="FAIL", passed=False),
            "must PASS",
        ),
        (
            "physical",
            lambda value: value.update(workspace_frame="wrong"),
            "workspace_frame",
        ),
        (
            "physical",
            lambda value: value.update(target_identity={"wrong": True}),
            "target_identity",
        ),
        ("physical", lambda value: value.update(camera_set=["camera_a"]), "camera_set"),
        (
            "physical",
            lambda value: value["camera_bundle_hashes"].update(camera_b="f" * 64),
            "camera_bundle_hashes",
        ),
        ("physical", lambda value: value.update(per_pair={}), "per_pair"),
        (
            "physical",
            lambda value: value["gates"].update(pairwise_geometry=False),
            "all_gates",
        ),
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
            solution_path,
            validation_path,
            physical_path,
            tmp_path / "output.json",
            intrinsic_health_path=physical_path.with_name("intrinsic.json"),
        )


def test_promotion_requires_physical_acceptance(tmp_path: Path) -> None:
    (
        _solution,
        _validation,
        _physical,
        solution_path,
        validation_path,
        _physical_path,
    ) = _inputs(tmp_path)
    with pytest.raises(FileNotFoundError):
        promote_rig_calibration(
            solution_path,
            validation_path,
            tmp_path / "missing.json",
            tmp_path / "output.json",
            intrinsic_health_path=tmp_path / "intrinsic.json",
        )


def test_deployment_fingerprint_is_tamper_evident(tmp_path: Path) -> None:
    _solution, _validation, _physical, solution_path, validation_path, physical_path = (
        _inputs(tmp_path)
    )
    output = tmp_path / "deployment.json"
    promote_rig_calibration(
        solution_path,
        validation_path,
        physical_path,
        output,
        intrinsic_health_path=physical_path.with_name("intrinsic.json"),
    )
    raw = json.loads(output.read_text())
    raw["workspace_frame"] = "tampered"
    output.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_rig_calibration_deployment(output)


def test_deployment_rejects_rehashed_contradictory_source_receipt(
    tmp_path: Path,
) -> None:
    from pointcloud_builder.rig_calibration.deployment import deployment_fingerprint

    _solution, _validation, _physical, solution_path, validation_path, physical_path = (
        _inputs(tmp_path)
    )
    output = tmp_path / "deployment.json"
    promote_rig_calibration(
        solution_path,
        validation_path,
        physical_path,
        output,
        intrinsic_health_path=physical_path.with_name("intrinsic.json"),
    )
    raw = json.loads(output.read_text())
    raw["source_receipts"]["validation"]["sha256"] = "0" * 64
    raw["rig_calibration_fingerprint"] = deployment_fingerprint(raw)
    output.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="validation source receipt differs"):
        load_rig_calibration_deployment(output)


def test_promotion_rejects_no_overlap_graph_forgery(tmp_path: Path) -> None:
    _solution, _validation, physical, solution_path, validation_path, physical_path = (
        _inputs(tmp_path)
    )
    for pair in physical["per_pair"].values():
        pair["status"] = "NOT_APPLICABLE_NO_OVERLAP"
    physical_path.write_text(json.dumps(physical), encoding="utf-8")
    with pytest.raises(ValueError, match="accepted_overlap_connectivity"):
        promote_rig_calibration(
            solution_path,
            validation_path,
            physical_path,
            tmp_path / "deployment.json",
            intrinsic_health_path=physical_path.with_name("intrinsic.json"),
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda physical: physical["per_pair"]["camera_a__camera_b"][
            "overlap_point_count"
        ].update(minimum_per_direction=0),
        lambda physical: physical["per_pair"]["camera_a__camera_b"][
            "symmetric_nn"
        ].update(median_of_matched_set_medians_mm=1e9),
        lambda physical: physical["per_pair"]["camera_a__camera_b"].update(
            evaluated_matched_set_count=299
        ),
    ],
)
def test_promotion_recomputes_physical_numeric_gates(tmp_path: Path, mutate) -> None:
    _solution, _validation, physical, solution_path, validation_path, physical_path = (
        _inputs(tmp_path)
    )
    mutate(physical)
    physical_path.write_text(json.dumps(physical), encoding="utf-8")

    with pytest.raises(ValueError, match="per_pair_semantics"):
        promote_rig_calibration(
            solution_path,
            validation_path,
            physical_path,
            tmp_path / "deployment.json",
            intrinsic_health_path=physical_path.with_name("intrinsic.json"),
        )


def test_promotion_requires_the_immutable_physical_recording(tmp_path: Path) -> None:
    _solution, _validation, physical, solution_path, validation_path, physical_path = (
        _inputs(tmp_path)
    )
    physical["source_receipts"]["source_recording_manifest"]["path"] = str(
        tmp_path / "missing-recording/manifest.json"
    )
    physical_path.write_text(json.dumps(physical), encoding="utf-8")

    with pytest.raises(ValueError, match="source_receipts"):
        promote_rig_calibration(
            solution_path,
            validation_path,
            physical_path,
            tmp_path / "deployment.json",
            intrinsic_health_path=physical_path.with_name("intrinsic.json"),
        )


def test_pose_plan_rejects_empty_visibility_matrix() -> None:
    from pointcloud_builder.rig_calibration.deployment import (
        _require_fixed_three_camera_pose_plan,
    )

    solution, _validation, _physical, intrinsic = _base_inputs()
    summary = deepcopy(solution.pose_plan_summary)
    summary["per_pose_camera_ids"] = {pose_id: [] for pose_id in summary["pose_ids"]}
    forged_solution = replace(solution, pose_plan_summary=summary)
    with pytest.raises(ValueError, match="nonempty_valid_visibility"):
        _require_fixed_three_camera_pose_plan(forged_solution, intrinsic)


def test_deployment_rejects_malformed_bootstrap_after_rehash(tmp_path: Path) -> None:
    from pointcloud_builder.rig_calibration.deployment import deployment_fingerprint

    _solution, _validation, _physical, solution_path, validation_path, physical_path = (
        _inputs(tmp_path)
    )
    output = tmp_path / "deployment.json"
    promote_rig_calibration(
        solution_path,
        validation_path,
        physical_path,
        output,
        intrinsic_health_path=physical_path.with_name("intrinsic.json"),
    )
    raw = json.loads(output.read_text())
    raw["bootstrap_qualifications"] = {camera_id: {} for camera_id in raw["cameras"]}
    raw["rig_calibration_fingerprint"] = deployment_fingerprint(raw)
    output.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="bootstrap qualification authority"):
        load_rig_calibration_deployment(output)


def test_promotion_rejects_forged_intrinsic_camera_stub(tmp_path: Path) -> None:
    _solution, _validation, _physical, solution_path, validation_path, physical_path = (
        _inputs(tmp_path)
    )
    intrinsic_path = physical_path.with_name("intrinsic.json")
    raw = json.loads(intrinsic_path.read_text())
    raw["per_camera"]["camera_a"] = {
        "schema_version": "camera-rig.intrinsic-health.v1",
        "status": "PASS",
        "factory_intrinsics_immutable": True,
    }
    from pointcloud_builder.rig_calibration.intrinsic_health import (
        intrinsic_health_fingerprint,
    )

    raw["intrinsic_health_fingerprint"] = intrinsic_health_fingerprint(raw)
    intrinsic_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="intrinsic-health semantics invalid"):
        promote_rig_calibration(
            solution_path,
            validation_path,
            physical_path,
            tmp_path / "output.json",
            intrinsic_health_path=intrinsic_path,
        )


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
    updated, provenance = apply_deployment_to_context(context, "left_cam", deployment)
    np.testing.assert_allclose(
        updated.T_workspace_from_source.matrix,
        T_workspace_from_color @ T_color_from_ir,
    )
    assert updated.T_workspace_from_source.source_frame == ("left_cam/ir_left_optical")
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
