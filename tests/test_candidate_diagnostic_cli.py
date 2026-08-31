from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

import tools.mapping.diagnose_cross_camera_alignment as diagnostic_cli
from pointcloud_builder.camera_model import CameraIntrinsics
from pointcloud_builder.integrations.camera_rig.types import (
    CameraRigBuilderContext,
    FrameExplicitTransform,
)
from pointcloud_builder.rig_calibration.artifact import (
    solution_fingerprint,
    write_solution,
)
from pointcloud_builder.rig_calibration.diagnostics import (
    apply_candidate_to_live_pipeline,
)
from pointcloud_builder.rig_calibration.solver import solve_rig_calibration
from tests.rig_calibration_synthetic import make_scene
from tools.mapping.diagnose_cross_camera_alignment import (
    _aggregate_raw_metrics,
    _candidate_geometry_overrides,
    _geometry,
    _reframe_selected_tensors,
)


def test_selected_tensors_are_reframed_without_mutating_capture() -> None:
    points = np.asarray(((1.0, 2.0, 3.0, 0.1, 0.2, 0.3),))
    selected = {0: {"camera_a": {"points": points}}}
    production = {"camera_a": np.eye(4)}
    candidate = {"camera_a": np.eye(4)}
    candidate["camera_a"][0, 3] = 0.25

    reframed = _reframe_selected_tensors(selected, candidate, production)

    np.testing.assert_allclose(
        reframed[0]["camera_a"]["points"][0, :3], (1.25, 2.0, 3.0)
    )
    np.testing.assert_allclose(points[0, :3], (1.0, 2.0, 3.0))


def test_geometry_uses_candidate_override_only_when_explicit() -> None:
    observation = SimpleNamespace(
        depth_source="ffs_stereo",
        rectified=True,
        distortion_coeffs=(0.0,) * 5,
        metric_depth=np.ones((2, 2), dtype=np.float32),
        intrinsics=CameraIntrinsics(
            width=2,
            height=2,
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
        ),
        T_workspace_from_camera=np.eye(4),
    )
    crop = SimpleNamespace(enabled=False)
    candidate = np.eye(4)
    candidate[2, 3] = 0.1

    production_geometry = _geometry(observation, crop)
    candidate_geometry = _geometry(
        observation, crop, T_workspace_from_camera=candidate
    )

    np.testing.assert_allclose(
        candidate_geometry["xyz"][:, 2] - production_geometry["xyz"][:, 2],
        0.1,
    )


def test_empty_optional_scene_roi_is_explicitly_not_run() -> None:
    summary = {
        "count": 1,
        "median_mm": 1.0,
        "p95_mm": 1.0,
        "maximum_mm": 1.0,
        "rmse_mm": 1.0,
    }
    records = [{"before": {"symmetric": summary}, "after": {"symmetric": summary}}]
    rois = {
        "board": {"before": [], "after": []},
        "full_overlap": {"before": [summary], "after": [summary]},
    }

    result = _aggregate_raw_metrics(records, rois, [np.eye(4)])

    assert result["roi"]["board"]["status"] == "NOT_RUN"
    assert result["roi"]["full_overlap"]["status"] == "AVAILABLE"


def test_candidate_overrides_require_bound_holdout_and_bundle_provenance(
    tmp_path, monkeypatch
) -> None:
    data, _truth, _poses = make_scene(noise_px=0.1)
    solution = solve_rig_calibration(data)
    identities = {
        name: {"camera_name": name, "kind": "recorded"}
        for name in ("camera_a", "camera_b")
    }
    bundle_hashes = {}
    bundles = {}
    rig_cameras = []
    root = tmp_path / "capture"
    calibration_root = root / "depth_recording" / "calibration"
    calibration_root.mkdir(parents=True)
    for name in ("camera_a", "camera_b"):
        provision = tmp_path / name
        provision.mkdir()
        (provision / "camera_bundle.json").write_text(f"bundle:{name}")
        (provision / "manifest.json").write_text(f"manifest:{name}")
        bundle_hashes[name] = hashlib.sha256(
            (provision / "camera_bundle.json").read_bytes()
        ).hexdigest()
        bundles[str(provision)] = SimpleNamespace(
            bundle_id=f"bundle-{name}",
            device=SimpleNamespace(to_dict=lambda name=name: identities[name]),
        )
        rig_cameras.append(
            SimpleNamespace(
                name=name,
                source=SimpleNamespace(provision_artifact=str(provision)),
            )
        )
        (calibration_root / f"{name}.json").write_text(
            json.dumps(
                {
                    "workspace_frame": "workspace",
                    "bundle_identity": f"bundle-{name}",
                    "provision_sha256": hashlib.sha256(
                        (provision / "manifest.json").read_bytes()
                    ).hexdigest(),
                    "source_frame": f"{name}/ir_left_optical",
                    "T_workspace_from_camera": np.eye(4).tolist(),
                }
            )
        )
    solution = replace(
        solution,
        camera_bundle_hashes=bundle_hashes,
        camera_identities=identities,
    )
    solution_path = tmp_path / "solution.json"
    write_solution(solution, solution_path)
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "passed": True,
                "status": "PASS",
                "solution_fingerprint": solution_fingerprint(solution),
                "holdout": {"status": "PASS", "pose_count": 4},
            }
        )
    )
    monkeypatch.setattr(
        diagnostic_cli,
        "load_rig_config",
        lambda _path: SimpleNamespace(enabled_cameras=tuple(rig_cameras)),
    )
    monkeypatch.setattr(
        diagnostic_cli,
        "load_provisioned_camera_bundle",
        lambda path: bundles[str(path)],
    )
    monkeypatch.setattr(
        diagnostic_cli,
        "resolve_bundle_transform",
        lambda _bundle, source, target: SimpleNamespace(
            source_frame=source,
            target_frame=target,
            matrix=np.eye(4),
        ),
    )

    overrides, contract = _candidate_geometry_overrides(
        root,
        {"rig_config": "ignored.yaml"},
        {"camera_names": ["camera_a", "camera_b"], "workspace_frame": "workspace"},
        solution_path,
        validation_path,
    )

    assert set(overrides) == {"camera_a", "camera_b"}
    assert contract["production_applied"] is False
    assert contract["holdout"]["status"] == "PASS"


def test_live_candidate_override_is_in_memory_and_validation_bound(tmp_path) -> None:
    data, _truth, _poses = make_scene(noise_px=0.1)
    solution = solve_rig_calibration(data)
    identities = {
        name: {"camera_name": name, "kind": "live"}
        for name in ("camera_a", "camera_b")
    }
    hashes = {}
    cameras = []
    runtimes = {}
    bundle_files = {}
    for name in ("camera_a", "camera_b"):
        provision = tmp_path / name
        provision.mkdir()
        bundle_file = provision / "camera_bundle.json"
        bundle_file.write_text(f"live-bundle:{name}")
        bundle_files[name] = bundle_file.read_bytes()
        hashes[name] = hashlib.sha256(bundle_files[name]).hexdigest()
        bundle = SimpleNamespace(
            device=SimpleNamespace(to_dict=lambda name=name: identities[name])
        )
        context = CameraRigBuilderContext(
            builder=SimpleNamespace(),
            source_frame=f"{name}/ir_left_optical",
            workspace_frame="workspace",
            calibration=SimpleNamespace(
                bundle=bundle,
                transform=lambda source, target: FrameExplicitTransform(
                    source_frame=source,
                    target_frame=target,
                    matrix=np.eye(4),
                ),
            ),
            T_workspace_from_source=FrameExplicitTransform(
                source_frame=f"{name}/ir_left_optical",
                target_frame="workspace",
                matrix=np.eye(4),
            ),
            depth_mode="ffs_stereo",
            frame_adapter=SimpleNamespace(),
        )
        cameras.append(
            SimpleNamespace(
                name=name,
                source=SimpleNamespace(provision_artifact=str(provision)),
            )
        )
        runtimes[name] = SimpleNamespace(
            pipeline=SimpleNamespace(context=context), provenance={}
        )
    solution = replace(
        solution,
        camera_bundle_hashes=hashes,
        camera_identities=identities,
    )
    validation = {
        "passed": True,
        "status": "PASS",
        "solution_fingerprint": solution_fingerprint(solution),
        "holdout": {"status": "PASS", "pose_count": 4},
    }
    pipeline = SimpleNamespace(processor=SimpleNamespace(runtimes=runtimes))
    rig = SimpleNamespace(enabled_cameras=tuple(cameras), output_frame="workspace")

    contract = apply_candidate_to_live_pipeline(pipeline, rig, solution, validation)

    for name in ("camera_a", "camera_b"):
        np.testing.assert_allclose(
            runtimes[name].pipeline.context.T_workspace_from_source.matrix,
            solution.T_workspace_from_camera[name],
        )
        assert runtimes[name].provenance["calibration_mode"] == (
            "validated_candidate_only"
        )
        assert (tmp_path / name / "camera_bundle.json").read_bytes() == bundle_files[name]
    assert contract["live_view_only"] is True
    assert contract["production_applied"] is False
