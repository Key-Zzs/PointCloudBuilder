"""Generic N-camera pairwise geometry acceptance and legacy evidence binding."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from pointcloud_builder.diagnostics.cross_camera_alignment import (
    robust_point_to_point_icp,
)
from pointcloud_builder.rig_calibration.artifact import (
    load_solution,
    solution_fingerprint,
)
from pointcloud_builder.rig_calibration.deployment import (
    PHYSICAL_ACCEPTANCE_SCHEMA_VERSION,
    sha256_file,
)
from pointcloud_builder.rig_calibration.diagnostics import require_validated_candidate


@dataclass(frozen=True)
class NCameraAcceptanceThresholds:
    maximum_overlap_distance_mm: float = 30.0
    minimum_overlap_points: int = 500
    maximum_symmetric_median_mm: float = 3.0
    maximum_symmetric_p95_mm: float = 10.0
    maximum_board_median_mm: float = 2.0
    maximum_board_p95_mm: float = 5.0
    maximum_plane_offset_mm: float = 2.0
    maximum_normal_split_deg: float = 1.0
    maximum_double_layer_thickness_mm: float = 4.0
    maximum_diagnostic_translation_mm: float = 3.0
    maximum_diagnostic_rotation_deg: float = 1.0
    voxel_size_mm: float = 2.5

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
            for value in values.values()
        ):
            raise ValueError("N-camera acceptance thresholds must be positive")


REAL_DUAL_MULTIPOSE_V1_THRESHOLDS = NCameraAcceptanceThresholds(
    maximum_overlap_distance_mm=30.0,
    minimum_overlap_points=100_000,
    maximum_symmetric_median_mm=1.5,
    maximum_symmetric_p95_mm=25.0,
    maximum_board_median_mm=1.25,
    maximum_board_p95_mm=2.5,
    maximum_plane_offset_mm=1.0,
    maximum_normal_split_deg=0.6,
    maximum_double_layer_thickness_mm=2.5,
    maximum_diagnostic_translation_mm=1.0,
    maximum_diagnostic_rotation_deg=0.25,
    voxel_size_mm=2.5,
)


def canonical_pair(camera_a: str, camera_b: str) -> tuple[str, str]:
    if camera_a == camera_b:
        raise ValueError("pair cameras must be distinct")
    return tuple(sorted((camera_a, camera_b)))  # type: ignore[return-value]


def pair_key(camera_a: str, camera_b: str) -> str:
    return "__".join(canonical_pair(camera_a, camera_b))


def generate_pairs(camera_ids: list[str] | tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    names = tuple(sorted(camera_ids))
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("N-camera acceptance requires unique camera names and N >= 2")
    return tuple(combinations(names, 2))


def evaluate_ncamera_alignment(
    clouds: Mapping[str, np.ndarray],
    *,
    thresholds: NCameraAcceptanceThresholds | None = None,
    board_masks: Mapping[str, np.ndarray] | None = None,
    declared_no_overlap_pairs: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Evaluate every unordered camera pair without camera-name assumptions."""

    gate = thresholds or NCameraAcceptanceThresholds()
    names = tuple(sorted(clouds))
    pairs = generate_pairs(names)
    no_overlap = {
        canonical_pair(*pair) for pair in (declared_no_overlap_pairs or set())
    }
    points = {name: _points(clouds[name], name) for name in names}
    masks = {} if board_masks is None else dict(board_masks)
    for name, mask in masks.items():
        if name not in points:
            raise ValueError(f"board mask references unknown camera {name!r}")
        value = np.asarray(mask, dtype=bool)
        if value.shape != (len(points[name]),):
            raise ValueError(f"{name}: board mask shape mismatch")
        masks[name] = value

    per_pair = {}
    accepted_edges = []
    for left, right in pairs:
        metrics = _evaluate_pair(
            left,
            right,
            points[left],
            points[right],
            gate,
            left_board=None if left not in masks else points[left][masks[left]],
            right_board=None if right not in masks else points[right][masks[right]],
            declared_no_overlap=(left, right) in no_overlap,
        )
        per_pair[pair_key(left, right)] = metrics
        if metrics["status"] == "PASS":
            accepted_edges.append([left, right])
    connected = _connected(names, accepted_edges)
    statuses = [value["status"] for value in per_pair.values()]
    passed = bool(
        connected
        and all(status in {"PASS", "NOT_APPLICABLE_NO_OVERLAP"} for status in statuses)
    )
    fused = _voxel_downsample(
        np.concatenate([points[name] for name in names]), gate.voxel_size_mm / 1000.0
    )
    counts = {name: len(points[name]) for name in names}
    total = sum(counts.values())
    return {
        "schema_version": "pointcloud-builder.ncamera-alignment-evaluation.v1",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "camera_set": list(names),
        "thresholds": asdict(gate),
        "per_pair": per_pair,
        "all_rig": {
            "camera_graph_nodes": list(names),
            "accepted_pairwise_overlap_edges": accepted_edges,
            "pairwise_overlap_connectivity": {
                "connected": connected,
                "accepted_edge_count": len(accepted_edges),
                "required_node_count": len(names),
            },
            "per_camera_point_contribution": {
                name: {
                    "point_count": counts[name],
                    "fraction": counts[name] / total,
                }
                for name in names
            },
            "all_camera_input_point_count": total,
            "all_camera_fused_point_count": len(fused),
            "all_camera_surface_thickness_mm": _surface_thickness_mm(fused),
            "camera_drop_statistics": {name: 0 for name in names},
            "matcher_statistics": {"evaluated_complete_sets": 1, "dropped_sets": 0},
        },
        "gates": {
            "all_pair_declarations_valid": all(
                status in {"PASS", "NOT_APPLICABLE_NO_OVERLAP"}
                for status in statuses
            ),
            "connected_overlap_graph": connected,
        },
        "diagnostic_residual_writeback": False,
    }


def aggregate_ncamera_evaluations(
    evaluations: list[Mapping[str, Any]],
    *,
    thresholds: NCameraAcceptanceThresholds,
    diagnostic_residuals: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate pose-local evaluations without mixing moving scene geometry."""

    if not evaluations:
        raise ValueError("at least one N-camera evaluation is required")
    camera_set = evaluations[0].get("camera_set")
    pair_keys = tuple(sorted(evaluations[0].get("per_pair", {})))
    if not isinstance(camera_set, list) or len(camera_set) < 2 or not pair_keys:
        raise ValueError("invalid N-camera evaluation")
    for evaluation in evaluations:
        if evaluation.get("camera_set") != camera_set:
            raise ValueError("N-camera evaluation camera sets differ")
        if tuple(sorted(evaluation.get("per_pair", {}))) != pair_keys:
            raise ValueError("N-camera evaluation pair sets differ")

    per_pair: dict[str, Any] = {}
    accepted_edges: list[list[str]] = []
    for key in pair_keys:
        frames = [evaluation["per_pair"][key] for evaluation in evaluations]
        statuses = {frame["status"] for frame in frames}
        if statuses == {"NOT_APPLICABLE_NO_OVERLAP"}:
            valid = all(frame["gates"]["no_measurable_overlap"] for frame in frames)
            per_pair[key] = {
                "status": "NOT_APPLICABLE_NO_OVERLAP" if valid else "FAIL",
                "cameras": list(frames[0]["cameras"]),
                "evaluated_matched_set_count": len(frames),
                "physical_justification_required": True,
                "declared_no_overlap": True,
                "gates": {"no_measurable_overlap": valid},
            }
            continue
        if "NOT_APPLICABLE_NO_OVERLAP" in statuses:
            raise ValueError("pair overlap declaration changed between matched sets")

        overlap = min(
            int(frame["overlap_point_count"]["minimum_per_direction"])
            for frame in frames
        )
        symmetric_median = _median_metric(frames, "symmetric_nn", "median_mm")
        symmetric_p95 = _p95_metric(frames, "symmetric_nn", "p95_mm")
        board_median = _median_metric(frames, "board_interior", "median_mm")
        board_p95 = _p95_metric(frames, "board_interior", "p95_mm")
        plane_offset = _p95_abs_metric(frames, "board_plane", "signed_offset_mm")
        normal_split = _p95_metric(frames, "board_plane", "normal_split_deg")
        double_layer = _p95_metric(
            frames, "board_plane", "double_layer_thickness_mm"
        )
        residual = None if diagnostic_residuals is None else diagnostic_residuals.get(key)
        translation = (
            _median_metric(frames, "diagnostic_residual_se3", "translation_norm_mm")
            if residual is None
            else float(residual["translation_norm_mm"])
        )
        rotation = (
            _median_metric(frames, "diagnostic_residual_se3", "rotation_geodesic_deg")
            if residual is None
            else float(residual["rotation_geodesic_deg"])
        )
        gates = {
            "minimum_overlap_points": overlap >= thresholds.minimum_overlap_points,
            "symmetric_median": symmetric_median
            <= thresholds.maximum_symmetric_median_mm,
            "symmetric_p95": symmetric_p95 <= thresholds.maximum_symmetric_p95_mm,
            "board_available": all(
                frame["board_interior"]["status"] == "AVAILABLE" for frame in frames
            ),
            "board_median": board_median <= thresholds.maximum_board_median_mm,
            "board_p95": board_p95 <= thresholds.maximum_board_p95_mm,
            "plane_offset": plane_offset <= thresholds.maximum_plane_offset_mm,
            "normal_split": normal_split <= thresholds.maximum_normal_split_deg,
            "double_layer_thickness": double_layer
            <= thresholds.maximum_double_layer_thickness_mm,
            "diagnostic_translation": translation
            <= thresholds.maximum_diagnostic_translation_mm,
            "diagnostic_rotation": rotation
            <= thresholds.maximum_diagnostic_rotation_deg,
        }
        cameras = list(frames[0]["cameras"])
        per_pair[key] = {
            "status": "PASS" if all(gates.values()) else "FAIL",
            "cameras": cameras,
            "evaluated_matched_set_count": len(frames),
            "overlap_point_count": {"minimum_per_direction": overlap},
            "symmetric_nn": {
                "median_of_matched_set_medians_mm": symmetric_median,
                "p95_of_matched_set_p95_mm": symmetric_p95,
            },
            "board_interior": {
                "status": "AVAILABLE" if gates["board_available"] else "NOT_RUN",
                "median_of_matched_set_medians_mm": board_median,
                "p95_of_matched_set_p95_mm": board_p95,
            },
            "board_plane": {
                "absolute_signed_offset_p95_mm": plane_offset,
                "normal_split_p95_deg": normal_split,
                "double_layer_thickness_p95_mm": double_layer,
            },
            "diagnostic_residual_se3": {
                "translation_xyz_mm": (
                    list(residual["translation_xyz_mm"])
                    if residual is not None and "translation_xyz_mm" in residual
                    else None
                ),
                "translation_norm_mm": translation,
                "rotation_geodesic_deg": rotation,
                "aggregation": (
                    "aggregate_matched_set_point_to_plane_icp"
                    if residual is not None
                    else "median_of_pose_local_point_to_point_icp"
                ),
                "diagnostic_only": True,
                "written_back": False,
            },
            "gates": gates,
        }
        if all(gates.values()):
            accepted_edges.append(cameras)

    names = tuple(camera_set)
    connected = _connected(names, accepted_edges)
    statuses = [value["status"] for value in per_pair.values()]
    passed = connected and all(
        status in {"PASS", "NOT_APPLICABLE_NO_OVERLAP"} for status in statuses
    )
    contributions = {
        name: sum(
            int(evaluation["all_rig"]["per_camera_point_contribution"][name][
                "point_count"
            ])
            for evaluation in evaluations
        )
        for name in names
    }
    total = sum(contributions.values())
    return {
        "schema_version": "pointcloud-builder.ncamera-alignment-evaluation.v1",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "camera_set": list(names),
        "evaluated_matched_set_count": len(evaluations),
        "thresholds": asdict(thresholds),
        "per_pair": per_pair,
        "all_rig": {
            "camera_graph_nodes": list(names),
            "accepted_pairwise_overlap_edges": accepted_edges,
            "pairwise_overlap_connectivity": {
                "connected": connected,
                "accepted_edge_count": len(accepted_edges),
                "required_node_count": len(names),
            },
            "per_camera_point_contribution": {
                name: {
                    "point_count": contributions[name],
                    "fraction": contributions[name] / total,
                }
                for name in names
            },
            "all_camera_input_point_count": total,
            "camera_drop_statistics": {name: 0 for name in names},
            "matcher_statistics": {"evaluated_complete_sets": len(evaluations)},
        },
        "gates": {
            "all_pair_declarations_valid": all(
                status in {"PASS", "NOT_APPLICABLE_NO_OVERLAP"}
                for status in statuses
            ),
            "connected_overlap_graph": connected,
        },
        "diagnostic_residual_writeback": False,
    }


def diagnostic_point_to_plane_residual(
    anchor_frames: list[np.ndarray], moving_frames: list[np.ndarray]
) -> dict[str, Any]:
    """Fit one diagnostic-only SE(3) over pose-local, voxelized frame clouds."""

    if not anchor_frames or len(anchor_frames) != len(moving_frames):
        raise ValueError("diagnostic residual requires equal non-empty frame lists")
    anchor = np.concatenate(
        [_first_voxel_downsample(_points(value, "anchor"), 0.004) for value in anchor_frames]
    )
    moving = np.concatenate(
        [_first_voxel_downsample(_points(value, "moving"), 0.004) for value in moving_frames]
    )
    import open3d as o3d
    from scipy.spatial import cKDTree

    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(anchor))
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(moving))
    target = target.voxel_down_sample(0.004)
    source = source.voxel_down_sample(0.004)
    normal_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.020, max_nn=40)
    target.estimate_normals(normal_param)
    source.estimate_normals(normal_param)
    target_points = np.asarray(target.points)
    source_points = np.asarray(source.points)
    target_normals = np.asarray(target.normals)
    source_normals = np.asarray(source.normals)
    source_distance, source_index = cKDTree(target_points).query(
        source_points, workers=1
    )
    _, target_index = cKDTree(source_points).query(target_points, workers=1)
    rows = np.arange(source_points.shape[0])
    mutual = target_index[source_index] == rows
    normal_consistent = np.abs(
        np.einsum("ij,ij->i", source_normals, target_normals[source_index])
    ) >= math.cos(math.radians(30.0))
    accepted = mutual & normal_consistent & (source_distance <= 0.030)
    accepted_rows = np.flatnonzero(accepted)
    if accepted_rows.size < 100:
        raise RuntimeError(
            "point-to-plane ICP has too few mutual normal-consistent correspondences"
        )
    trim_count = max(100, math.floor(accepted_rows.size * 0.80))
    chosen = accepted_rows[
        np.argsort(source_distance[accepted_rows], kind="stable")[:trim_count]
    ]
    fit_source = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(source_points[chosen])
    )
    fit_source.normals = o3d.utility.Vector3dVector(source_normals[chosen])
    target_rows = source_index[chosen]
    fit_target = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(target_points[target_rows])
    )
    fit_target.normals = o3d.utility.Vector3dVector(target_normals[target_rows])
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(
        o3d.pipelines.registration.TukeyLoss(k=0.010)
    )
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=80
    )
    result = o3d.pipelines.registration.registration_icp(
        fit_source, fit_target, 0.030, np.eye(4), estimation, criteria
    )
    matrix = np.asarray(result.transformation)
    if result.fitness <= 0 or not np.isfinite(matrix).all():
        raise RuntimeError("robust point-to-plane ICP did not produce a valid fit")
    return {
        "translation_xyz_mm": (matrix[:3, 3] * 1000.0).tolist(),
        "translation_norm_mm": float(np.linalg.norm(matrix[:3, 3]) * 1000.0),
        "rotation_geodesic_deg": _rotation_geodesic_deg(matrix[:3, :3]),
    }


def bind_physical_acceptance(
    evaluation: Mapping[str, Any],
    solution_path: str | Path,
    *,
    source_receipts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a passed N-camera evaluation to one exact candidate solution."""

    solution_source = Path(solution_path).expanduser().resolve()
    solution = load_solution(solution_source)
    expected = sorted(solution.T_workspace_from_camera)
    if evaluation.get("passed") is not True or evaluation.get("status") != "PASS":
        raise ValueError("cannot bind a failed N-camera physical evaluation")
    if evaluation.get("camera_set") != expected:
        raise ValueError("physical evaluation and solution camera sets differ")
    gates = dict(evaluation.get("gates", {}))
    if not gates or not all(value is True for value in gates.values()):
        raise ValueError("physical evaluation gates did not all pass")
    return {
        "schema_version": PHYSICAL_ACCEPTANCE_SCHEMA_VERSION,
        "status": "PASS",
        "passed": True,
        "solution_fingerprint": solution_fingerprint(solution),
        "workspace_frame": solution.workspace_frame,
        "target_identity": solution.target_identity,
        "camera_set": expected,
        "camera_bundle_hashes": solution.camera_bundle_hashes,
        "camera_identities": solution.camera_identities,
        "per_pair": dict(evaluation["per_pair"]),
        "all_rig": dict(evaluation["all_rig"]),
        "thresholds": dict(evaluation["thresholds"]),
        "gates": gates,
        "diagnostic_residual_writeback": False,
        "source_receipts": {
            "solution": {
                "path": str(solution_source),
                "sha256": sha256_file(solution_source),
            },
            **dict(source_receipts or {}),
        },
    }


def summarize_legacy_ab_diagnostic(
    solution_path: str | Path,
    validation_path: str | Path,
    diagnostic_root: str | Path,
    *,
    thresholds: NCameraAcceptanceThresholds = REAL_DUAL_MULTIPOSE_V1_THRESHOLDS,
) -> dict[str, Any]:
    """Deterministically derive formal A/B acceptance from immutable reports.

    Only the pre-residual candidate geometry is accepted. The diagnostic ICP
    transform is reported and remains writeback=false. The invalidated cube
    detector is not consulted.
    """

    solution_source = Path(solution_path).expanduser().resolve()
    validation_source = Path(validation_path).expanduser().resolve()
    root = Path(diagnostic_root).expanduser().resolve()
    solution = load_solution(solution_source)
    validation = _read_json(validation_source)
    require_validated_candidate(solution, validation)
    if sorted(solution.T_workspace_from_camera) != ["camera_a", "camera_b"]:
        raise ValueError("legacy diagnostic summarizer only supports its A/B source")
    manifest_path = root / "candidate" / "manifest.json"
    raw_path = root / "candidate" / "raw_metrics.json"
    residual_path = root / "candidate" / "residual_transform.json"
    invalidation_path = root / "candidate_cube_false_positive_invalidation.json"
    manifest = _read_json(manifest_path)
    raw = _read_json(raw_path)
    residual = _read_json(residual_path)
    invalidation = _read_json(invalidation_path)
    contract = manifest.get("candidate_contract")
    fingerprint = solution_fingerprint(solution)
    if not isinstance(contract, dict) or contract.get("solution_fingerprint") != fingerprint:
        raise ValueError("candidate diagnostic does not bind the exact solution")
    if manifest.get("production_calibration_modified") is not False:
        raise ValueError("legacy diagnostic modified production calibration")
    if residual.get("diagnostic_only") is not True or residual.get(
        "written_back_to_camera_bundle"
    ) is not False:
        raise ValueError("legacy diagnostic residual writeback contract is invalid")
    if (
        invalidation.get("status") != "PARTIALLY_INVALIDATED"
        or invalidation.get("formal_cube_acceptance") != "NOT_RUN"
    ):
        raise ValueError("cube false-positive invalidation receipt is missing")
    frames = raw.get("per_frame")
    if not isinstance(frames, list) or len(frames) < 1:
        raise ValueError("legacy diagnostic has no frame metrics")

    def values(selector):
        result = np.asarray([selector(item["before"]) for item in frames], dtype=float)
        if not np.isfinite(result).all():
            raise ValueError("legacy diagnostic metric is not finite")
        return result

    overlap_a = values(lambda item: item["overlap_count"]["camera_a"])
    overlap_b = values(lambda item: item["overlap_count"]["camera_b"])
    symmetric_median = values(lambda item: item["symmetric"]["median_mm"])
    symmetric_p95 = values(lambda item: item["symmetric"]["p95_mm"])
    board_median = values(lambda item: item["roi"]["board"]["median_mm"])
    board_p95 = values(lambda item: item["roi"]["board"]["p95_mm"])
    plane_offset = values(lambda item: abs(item["board_plane"]["signed_plane_offset_mm"]))
    normal_split = values(lambda item: item["board_plane"]["normal_angular_difference_deg"])
    double_layer = values(
        lambda item: item["board_plane"]["combined_double_layer_thickness_mm"]
    )
    translation = float(residual["translation"]["norm_mm"])
    rotation = float(residual["rotation"]["geodesic_deg"])
    metrics = {
        "status": "PASS",
        "cameras": ["camera_a", "camera_b"],
        "evaluated_frame_count": len(frames),
        "overlap_point_count": {
            "minimum_per_direction": int(min(overlap_a.min(), overlap_b.min())),
            "median_camera_a": float(np.median(overlap_a)),
            "median_camera_b": float(np.median(overlap_b)),
        },
        "symmetric_nn": {
            "median_of_frame_medians_mm": float(np.median(symmetric_median)),
            "p95_of_frame_p95_mm": float(np.quantile(symmetric_p95, 0.95)),
        },
        "board_interior": {
            "median_of_frame_medians_mm": float(np.median(board_median)),
            "p95_of_frame_p95_mm": float(np.quantile(board_p95, 0.95)),
        },
        "board_plane": {
            "absolute_signed_offset_p95_mm": float(np.quantile(plane_offset, 0.95)),
            "normal_split_p95_deg": float(np.quantile(normal_split, 0.95)),
            "double_layer_thickness_p95_mm": float(np.quantile(double_layer, 0.95)),
        },
        "diagnostic_residual_se3": {
            "translation_xyz_mm": [
                float(residual["translation"][key])
                for key in ("dx_mm", "dy_mm", "dz_mm")
            ],
            "translation_norm_mm": translation,
            "rotation_geodesic_deg": rotation,
            "diagnostic_only": True,
            "written_back": False,
        },
        "visual_review": {
            "status": "BOUND_TO_CURRENT_TASK_ACCEPTED_RIGID_GHOSTING_REMOVAL",
            "artifact_sha256": sha256_file(
                root / "candidate" / "rerun" / "cross-camera-alignment.rrd"
            ),
        },
        "invalidated_cube_detector": {
            "status": "EXCLUDED",
            "receipt_sha256": sha256_file(invalidation_path),
        },
    }
    pair_gates = {
        "minimum_overlap_points": metrics["overlap_point_count"][
            "minimum_per_direction"
        ]
        >= thresholds.minimum_overlap_points,
        "symmetric_median": metrics["symmetric_nn"]["median_of_frame_medians_mm"]
        <= thresholds.maximum_symmetric_median_mm,
        "symmetric_p95": metrics["symmetric_nn"]["p95_of_frame_p95_mm"]
        <= thresholds.maximum_symmetric_p95_mm,
        "board_median": metrics["board_interior"]["median_of_frame_medians_mm"]
        <= thresholds.maximum_board_median_mm,
        "board_p95": metrics["board_interior"]["p95_of_frame_p95_mm"]
        <= thresholds.maximum_board_p95_mm,
        "plane_offset": metrics["board_plane"]["absolute_signed_offset_p95_mm"]
        <= thresholds.maximum_plane_offset_mm,
        "normal_split": metrics["board_plane"]["normal_split_p95_deg"]
        <= thresholds.maximum_normal_split_deg,
        "double_layer_thickness": metrics["board_plane"][
            "double_layer_thickness_p95_mm"
        ]
        <= thresholds.maximum_double_layer_thickness_mm,
        "diagnostic_translation": translation
        <= thresholds.maximum_diagnostic_translation_mm,
        "diagnostic_rotation": rotation <= thresholds.maximum_diagnostic_rotation_deg,
    }
    metrics["gates"] = pair_gates
    metrics["status"] = "PASS" if all(pair_gates.values()) else "FAIL"
    evaluation = {
        "status": metrics["status"],
        "passed": all(pair_gates.values()),
        "camera_set": ["camera_a", "camera_b"],
        "thresholds": asdict(thresholds),
        "per_pair": {"camera_a__camera_b": metrics},
        "all_rig": {
            "camera_graph_nodes": ["camera_a", "camera_b"],
            "accepted_pairwise_overlap_edges": (
                [["camera_a", "camera_b"]] if all(pair_gates.values()) else []
            ),
            "pairwise_overlap_connectivity": {
                "connected": all(pair_gates.values()),
                "accepted_edge_count": 1 if all(pair_gates.values()) else 0,
            },
            "camera_drop_statistics": manifest.get("timing", {}).get(
                "per_camera", {}
            ),
            "matcher_statistics": manifest.get("matcher_statistics", {}),
        },
        "gates": {
            "all_pair_declarations_valid": all(pair_gates.values()),
            "connected_overlap_graph": all(pair_gates.values()),
        },
    }
    if not evaluation["passed"]:
        failed = sorted(name for name, passed in pair_gates.items() if not passed)
        raise ValueError("legacy physical acceptance gates failed: " + ", ".join(failed))
    return bind_physical_acceptance(
        evaluation,
        solution_source,
        source_receipts={
            "validation": {
                "path": str(validation_source),
                "sha256": sha256_file(validation_source),
            },
            "candidate_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "candidate_raw_metrics": {
                "path": str(raw_path),
                "sha256": sha256_file(raw_path),
            },
            "candidate_residual": {
                "path": str(residual_path),
                "sha256": sha256_file(residual_path),
            },
            "cube_false_positive_invalidation": {
                "path": str(invalidation_path),
                "sha256": sha256_file(invalidation_path),
            },
        },
    )


def _evaluate_pair(
    left_name: str,
    right_name: str,
    left: np.ndarray,
    right: np.ndarray,
    thresholds: NCameraAcceptanceThresholds,
    *,
    left_board: np.ndarray | None,
    right_board: np.ndarray | None,
    declared_no_overlap: bool,
) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    aa = left
    bb = right
    max_distance = thresholds.maximum_overlap_distance_mm / 1000.0
    a_distance, a_index = cKDTree(bb).query(aa, workers=1)
    b_distance, _b_index = cKDTree(aa).query(bb, workers=1)
    a_mask = np.isfinite(a_distance) & (a_distance <= max_distance)
    b_mask = np.isfinite(b_distance) & (b_distance <= max_distance)
    minimum_overlap = min(int(a_mask.sum()), int(b_mask.sum()))
    if declared_no_overlap:
        valid_declaration = minimum_overlap < thresholds.minimum_overlap_points
        return {
            "status": (
                "NOT_APPLICABLE_NO_OVERLAP" if valid_declaration else "FAIL"
            ),
            "cameras": [left_name, right_name],
            "overlap_point_count": {
                left_name: int(a_mask.sum()),
                right_name: int(b_mask.sum()),
                "minimum_per_direction": minimum_overlap,
            },
            "physical_justification_required": True,
            "declared_no_overlap": True,
            "gates": {"no_measurable_overlap": valid_declaration},
        }
    if minimum_overlap == 0:
        return {
            "status": "FAIL",
            "cameras": [left_name, right_name],
            "overlap_point_count": {"minimum_per_direction": 0},
            "failure_reason": "NO_MEASURABLE_OVERLAP",
            "gates": {"minimum_overlap_points": False},
        }
    distances = np.concatenate((a_distance[a_mask], b_distance[b_mask])) * 1000.0
    board = _board_metrics(left_board, right_board, max_distance)
    plane = _plane_metrics(
        aa[a_mask], bb[b_mask], left_board=left_board, right_board=right_board
    )
    sample_a = aa[a_mask]
    sample_b = bb[a_index[a_mask]]
    if len(sample_a) > 20_000:
        indices = np.linspace(0, len(sample_a) - 1, 20_000).round().astype(int)
        sample_a = sample_a[indices]
        sample_b = sample_b[indices]
    residual = robust_point_to_point_icp(
        sample_a,
        sample_b,
        maximum_correspondence_distance_m=max_distance,
    )["T_anchor_from_moving_residual"]
    rotation_deg = _rotation_geodesic_deg(residual[:3, :3])
    translation_mm = residual[:3, 3] * 1000.0
    gates = {
        "minimum_overlap_points": minimum_overlap
        >= thresholds.minimum_overlap_points,
        "symmetric_median": float(np.median(distances))
        <= thresholds.maximum_symmetric_median_mm,
        "symmetric_p95": float(np.quantile(distances, 0.95))
        <= thresholds.maximum_symmetric_p95_mm,
        "board_available": board["status"] == "AVAILABLE",
        "board_median": board.get("median_mm", math.inf)
        <= thresholds.maximum_board_median_mm,
        "board_p95": board.get("p95_mm", math.inf)
        <= thresholds.maximum_board_p95_mm,
        "plane_offset": abs(plane["signed_offset_mm"])
        <= thresholds.maximum_plane_offset_mm,
        "normal_split": plane["normal_split_deg"]
        <= thresholds.maximum_normal_split_deg,
        "double_layer_thickness": plane["double_layer_thickness_mm"]
        <= thresholds.maximum_double_layer_thickness_mm,
        "diagnostic_translation": float(np.linalg.norm(translation_mm))
        <= thresholds.maximum_diagnostic_translation_mm,
        "diagnostic_rotation": rotation_deg
        <= thresholds.maximum_diagnostic_rotation_deg,
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "cameras": [left_name, right_name],
        "overlap_point_count": {
            left_name: int(a_mask.sum()),
            right_name: int(b_mask.sum()),
            "minimum_per_direction": minimum_overlap,
        },
        "symmetric_nn": {
            "median_mm": float(np.median(distances)),
            "p95_mm": float(np.quantile(distances, 0.95)),
        },
        "board_interior": board,
        "board_plane": plane,
        "diagnostic_residual_se3": {
            "translation_xyz_mm": translation_mm.tolist(),
            "translation_norm_mm": float(np.linalg.norm(translation_mm)),
            "rotation_geodesic_deg": rotation_deg,
            "diagnostic_only": True,
            "written_back": False,
        },
        "gates": gates,
    }


def _board_metrics(
    left: np.ndarray | None, right: np.ndarray | None, maximum_distance_m: float
) -> dict[str, Any]:
    if left is None or right is None or not len(left) or not len(right):
        return {"status": "NOT_RUN", "reason": "BOARD_MASKS_REQUIRED"}
    from scipy.spatial import cKDTree

    left_distance = cKDTree(right).query(left, workers=1)[0]
    right_distance = cKDTree(left).query(right, workers=1)[0]
    values = np.concatenate(
        (
            left_distance[left_distance <= maximum_distance_m],
            right_distance[right_distance <= maximum_distance_m],
        )
    )
    if not len(values):
        return {"status": "NOT_RUN", "reason": "NO_BOARD_OVERLAP"}
    return {
        "status": "AVAILABLE",
        "count": len(values),
        "median_mm": float(np.median(values) * 1000.0),
        "p95_mm": float(np.quantile(values, 0.95) * 1000.0),
    }


def _plane_metrics(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_board: np.ndarray | None,
    right_board: np.ndarray | None,
) -> dict[str, float]:
    aa = left if left_board is None else left_board
    bb = right if right_board is None else right_board
    normal_a, _offset_a, signed_a = _fit_plane(aa)
    normal_b, _offset_b, signed_b = _fit_plane(bb)
    if float(normal_a @ normal_b) < 0:
        normal_b = -normal_b
        signed_b = -signed_b
    center_a = np.median(aa, axis=0)
    signed_b_to_anchor = (bb - center_a) @ normal_a
    combined_points = np.concatenate((aa, bb))
    _combined_normal, _combined_offset, combined = _fit_plane(combined_points)
    return {
        "signed_offset_mm": float(np.median(signed_b_to_anchor) * 1000.0),
        "normal_split_deg": _normal_angle_deg(normal_a, normal_b),
        "plane_thickness_left_mm": float(
            (np.quantile(signed_a, 0.95) - np.quantile(signed_a, 0.05)) * 1000.0
        ),
        "plane_thickness_right_mm": float(
            (np.quantile(signed_b, 0.95) - np.quantile(signed_b, 0.05)) * 1000.0
        ),
        "double_layer_thickness_mm": float(
            (np.quantile(combined, 0.95) - np.quantile(combined, 0.05)) * 1000.0
        ),
    }


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    if len(points) < 6:
        raise ValueError("plane metrics require at least six points")
    center = np.median(points, axis=0)
    _u, _s, vt = np.linalg.svd(points - center, full_matrices=False)
    normal = vt[-1]
    normal /= np.linalg.norm(normal)
    offset = float(center @ normal)
    signed = points @ normal - offset
    return normal, offset, signed


def _connected(names: tuple[str, ...], edges: list[list[str]]) -> bool:
    adjacency = {name: set() for name in names}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen = {names[0]}
    stack = [names[0]]
    while stack:
        current = stack.pop()
        for neighbor in sorted(adjacency[current] - seen):
            seen.add(neighbor)
            stack.append(neighbor)
    return seen == set(names)


def _voxel_downsample(points: np.ndarray, voxel_m: float) -> np.ndarray:
    keys = np.floor(points / voxel_m).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys = keys[order]
    sorted_points = points[order]
    starts = np.concatenate(([0], np.flatnonzero(np.any(np.diff(keys, axis=0), axis=1)) + 1))
    counts = np.diff(np.concatenate((starts, [len(points)])))
    return np.add.reduceat(sorted_points, starts, axis=0) / counts[:, None]


def _first_voxel_downsample(points: np.ndarray, voxel_m: float) -> np.ndarray:
    keys = np.floor(points / voxel_m).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def _surface_thickness_mm(points: np.ndarray) -> float | None:
    if len(points) < 6:
        return None
    _normal, _offset, signed = _fit_plane(points)
    return float((np.quantile(signed, 0.95) - np.quantile(signed, 0.05)) * 1000.0)


def _normal_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    return float(math.degrees(math.acos(float(np.clip(left @ right, -1.0, 1.0)))))


def _rotation_geodesic_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _metric(frames: list[Mapping[str, Any]], group: str, name: str) -> np.ndarray:
    values = np.asarray([frame[group][name] for frame in frames], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite aggregate metric {group}.{name}")
    return values


def _median_metric(frames: list[Mapping[str, Any]], group: str, name: str) -> float:
    return float(np.median(_metric(frames, group, name)))


def _p95_metric(frames: list[Mapping[str, Any]], group: str, name: str) -> float:
    return float(np.quantile(_metric(frames, group, name), 0.95))


def _p95_abs_metric(frames: list[Mapping[str, Any]], group: str, name: str) -> float:
    return float(np.quantile(np.abs(_metric(frames, group, name)), 0.95))


def _points(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) < 6:
        raise ValueError(f"{name}: cloud must be finite Nx3 with N >= 6")
    if not np.isfinite(result).all():
        raise ValueError(f"{name}: cloud must contain finite values")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("artifact root must be a JSON object")
    return raw
