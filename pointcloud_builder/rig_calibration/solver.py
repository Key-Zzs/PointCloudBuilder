"""Gauge-fixed SE(3) bundle adjustment for N fixed cameras and moving target poses."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from pointcloud_builder.rig_calibration.config import RigCalibrationConfig
from pointcloud_builder.rig_calibration.graph import (
    CalibrationPreflightError,
    analyze_observation_graph,
    require_connected_graph,
)
from pointcloud_builder.rig_calibration.initialization import initialize_rig
from pointcloud_builder.rig_calibration.projection import project_target_points
from pointcloud_builder.rig_calibration.quality import (
    assess_pose_diversity,
    jacobian_observability,
    reprojection_metrics,
)
from pointcloud_builder.rig_calibration.se3 import transform_error, validate_transform
from pointcloud_builder.rig_calibration.types import (
    RigCalibrationObservations,
    RigCalibrationSolution,
    RigTargetObservation,
)


@dataclass(frozen=True)
class _ParameterLayout:
    camera_ids: tuple[str, ...]
    variable_pose_ids: tuple[str, ...]
    anchor_pose_id: str

    @property
    def parameter_count(self) -> int:
        return 6 * (len(self.camera_ids) + len(self.variable_pose_ids))


def solve_rig_calibration(
    data: RigCalibrationObservations,
    config: RigCalibrationConfig | None = None,
    *,
    observations_sha256: str | None = None,
) -> RigCalibrationSolution:
    """Solve candidate camera extrinsics without mutating any CameraBundle."""

    config = config or RigCalibrationConfig()
    solve_observations = tuple(
        sorted(
            (item for item in data.observations if item.split == "solve"),
            key=lambda item: item.observation_id,
        )
    )
    _validate_observations(data, solve_observations, config)
    graph = analyze_observation_graph(
        solve_observations, camera_ids=data.camera_ids, split="solve"
    )
    require_connected_graph(graph)
    if config.anchor_pose_id not in {item.pose_id for item in solve_observations}:
        raise CalibrationPreflightError(
            "MISSING_GAUGE_ANCHOR",
            f"anchor pose {config.anchor_pose_id!r} has no solve observation",
        )
    initialization = initialize_rig(data, anchor_pose_id=config.anchor_pose_id)
    diversity = assess_pose_diversity(
        data,
        initialization.camera_poses,
        initialization.target_poses,
        config,
    )
    if not diversity["passed"]:
        raise CalibrationPreflightError(
            "INSUFFICIENT_POSE_DIVERSITY",
            ", ".join(diversity["failure_reasons"]),
        )
    layout = _ParameterLayout(
        camera_ids=data.camera_ids,
        variable_pose_ids=tuple(
            pose_id
            for pose_id in sorted(initialization.target_poses)
            if pose_id != config.anchor_pose_id
        ),
        anchor_pose_id=config.anchor_pose_id,
    )
    initial_parameters = _pack_parameters(
        initialization.camera_poses, initialization.target_poses, layout
    )
    initial_residuals, initial_corner_errors = _residuals(
        initial_parameters,
        layout,
        data,
        solve_observations,
    )
    try:
        from scipy.optimize import least_squares
    except ImportError as error:
        raise RuntimeError(
            "rig calibration requires the optional scipy dependency"
        ) from error
    optimized = least_squares(
        lambda values: _residuals(
            values,
            layout,
            data,
            solve_observations,
            return_corner_errors=False,
            robust_loss=config.robust_loss,
            loss_scale_px=config.loss_scale_px,
        ),
        initial_parameters,
        loss="linear",
        max_nfev=config.max_nfev,
        xtol=config.optimizer_xtol,
        ftol=config.optimizer_ftol,
        gtol=config.optimizer_gtol,
        method="trf",
        x_scale="jac",
    )
    camera_poses, target_poses = _unpack_parameters(optimized.x, layout)
    _validate_solution_transforms(camera_poses, target_poses)
    final_residuals, final_corner_errors = _residuals(
        optimized.x, layout, data, solve_observations
    )
    final_metrics = reprojection_metrics(final_corner_errors)
    per_camera, per_pose = _grouped_metrics(
        optimized.x, layout, data, solve_observations
    )
    corrections: dict[str, dict[str, Any]] = {}
    for camera_id in layout.camera_ids:
        optimizer_delta = transform_error(
            camera_poses[camera_id], initialization.camera_poses[camera_id]
        )
        if camera_id in data.initial_camera_poses:
            provision_delta = transform_error(
                camera_poses[camera_id], data.initial_camera_poses[camera_id]
            )
            provision_reference = "provided_fixed_mount_initial"
        else:
            provision_delta = None
            provision_reference = "NOT_AVAILABLE_NO_FIXED_PROVISION"
        corrections[camera_id] = {
            "provision_reference": provision_reference,
            "from_provision": provision_delta,
            "from_optimizer_initialization": optimizer_delta,
        }
    observability = jacobian_observability(optimized.jac)
    passed = bool(
        optimized.success
        and final_metrics["p95_px"] is not None
        and final_metrics["p95_px"] <= config.final_reprojection_p95_px
        and diversity["passed"]
        and observability["rank"] == layout.parameter_count
        and observability["condition_number"] is not None
        and observability["condition_number"] <= config.max_condition_number
    )
    validation = {
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
        "candidate_only": True,
        "production_applied": False,
        "solve_pose_split": "solve",
        "holdout": "AVAILABLE"
        if any(item.split == "holdout" for item in data.observations)
        else "NOT_RUN",
        "failure_reasons": _failure_reasons(
            optimized.success,
            final_metrics,
            config,
            observability,
            layout.parameter_count,
            config.max_condition_number,
        ),
    }
    return RigCalibrationSolution(
        workspace_frame=data.workspace_frame,
        anchor_pose_id=config.anchor_pose_id,
        camera_frames={
            camera_id: data.projection_models[camera_id].frame
            for camera_id in layout.camera_ids
        },
        target_identity=data.target_identity,
        camera_bundle_hashes=data.camera_bundle_hashes,
        camera_identities=data.camera_identities,
        T_workspace_from_camera=camera_poses,
        T_workspace_from_target=target_poses,
        initial_camera_poses=initialization.camera_poses,
        camera_corrections=corrections,
        quality_counts={
            "camera_count": len(layout.camera_ids),
            "pose_count": len(initialization.target_poses),
            "observation_count": len(solve_observations),
            "corner_count": sum(len(item.point_ids) for item in solve_observations),
        },
        reprojection={
            "initial": reprojection_metrics(initial_corner_errors),
            "final": final_metrics,
            "initial_scalar_residual_count": len(initial_residuals),
            "final_scalar_residual_count": len(final_residuals),
        },
        per_camera_metrics=per_camera,
        per_pose_metrics=per_pose,
        pose_diversity=diversity,
        graph=graph.to_dict(),
        optimizer={
            "success": bool(optimized.success),
            "status": int(optimized.status),
            "message": str(optimized.message),
            "nfev": int(optimized.nfev),
            "njev": int(optimized.njev) if optimized.njev is not None else None,
            "cost": float(optimized.cost),
            "optimality": float(optimized.optimality),
            "parameterization": "rotation_vector_plus_translation",
            "robust_loss": config.robust_loss,
            "loss_scale_px": config.loss_scale_px,
            "robust_block": "rho(squared_l2_corner_error)",
            "scipy_loss": "linear_after_block_robust_mapping",
            "gauge_anchor": f"T_workspace_from_target[{config.anchor_pose_id}]=I",
        },
        observability=observability,
        validation=validation,
        config=config.to_dict(),
        bootstrap_qualifications=data.bootstrap_qualifications,
        pose_plan_sha256=data.pose_plan_sha256,
        pose_plan_summary=data.pose_plan_summary,
        observations_sha256=observations_sha256,
    )


def _validate_observations(
    data: RigCalibrationObservations,
    observations: tuple[RigTargetObservation, ...],
    config: RigCalibrationConfig,
) -> None:
    if not observations:
        raise CalibrationPreflightError("NO_SOLVE_OBSERVATIONS", "solve split is empty")
    failed_quality = [
        item.observation_id
        for item in observations
        if item.quality.get("passed") is not True
    ]
    if failed_quality:
        raise CalibrationPreflightError(
            "OBSERVATION_QUALITY_FAILED", ", ".join(sorted(failed_quality))
        )
    too_few = [
        item.observation_id
        for item in observations
        if len(item.point_ids) < config.min_corners_per_observation
    ]
    if too_few:
        raise CalibrationPreflightError("TOO_FEW_CORNERS", ", ".join(sorted(too_few)))
    counts = defaultdict(int)
    for item in observations:
        counts[item.camera_id] += 1
    insufficient = [
        camera_id
        for camera_id in data.camera_ids
        if counts[camera_id] < config.min_observations_per_camera
    ]
    if insufficient:
        raise CalibrationPreflightError(
            "CAMERA_WITH_INSUFFICIENT_OBSERVATIONS", ", ".join(insufficient)
        )


def _pack_parameters(
    camera_poses: dict[str, np.ndarray],
    target_poses: dict[str, np.ndarray],
    layout: _ParameterLayout,
) -> np.ndarray:
    return np.concatenate(
        [
            _transform_to_parameters(camera_poses[camera_id])
            for camera_id in layout.camera_ids
        ]
        + [
            _transform_to_parameters(target_poses[pose_id])
            for pose_id in layout.variable_pose_ids
        ]
    )


def _unpack_parameters(
    parameters: np.ndarray, layout: _ParameterLayout
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    offset = 0
    cameras: dict[str, np.ndarray] = {}
    poses: dict[str, np.ndarray] = {layout.anchor_pose_id: np.eye(4, dtype=np.float64)}
    for camera_id in layout.camera_ids:
        cameras[camera_id] = _parameters_to_transform(parameters[offset : offset + 6])
        offset += 6
    for pose_id in layout.variable_pose_ids:
        poses[pose_id] = _parameters_to_transform(parameters[offset : offset + 6])
        offset += 6
    return cameras, poses


def _transform_to_parameters(matrix: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return np.concatenate(
        (Rotation.from_matrix(matrix[:3, :3]).as_rotvec(), matrix[:3, 3])
    )


def _parameters_to_transform(parameters: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
    result[:3, 3] = parameters[3:6]
    return result


def _residuals(
    parameters: np.ndarray,
    layout: _ParameterLayout,
    data: RigCalibrationObservations,
    observations: Iterable[RigTargetObservation],
    *,
    return_corner_errors: bool = True,
    robust_loss: str | None = None,
    loss_scale_px: float = 1.0,
) -> Any:
    cameras, poses = _unpack_parameters(parameters, layout)
    scalar_residuals: list[np.ndarray] = []
    corner_errors: list[np.ndarray] = []
    for observation in observations:
        projected, in_front = project_target_points(
            observation.object_points_m,
            poses[observation.pose_id],
            cameras[observation.camera_id],
            data.projection_models[observation.camera_id],
        )
        residual = projected - observation.image_points_px
        invalid = ~in_front | ~np.isfinite(residual).all(axis=1)
        residual[invalid] = 1e4
        corner_error = np.linalg.norm(residual, axis=1)
        corner_error[invalid] = 1e4
        if robust_loss is None:
            scalar_residuals.append(residual.reshape(-1))
        else:
            scalar_residuals.append(
                _block_robust_residuals(residual, robust_loss, loss_scale_px)
            )
        if return_corner_errors:
            corner_errors.append(corner_error)
    scalars = np.concatenate(scalar_residuals)
    if not return_corner_errors:
        return scalars
    return scalars, np.concatenate(corner_errors)


def _block_robust_residuals(
    residuals: np.ndarray,
    loss: str,
    scale_px: float,
) -> np.ndarray:
    """Map 2-D corner blocks so linear least squares equals rho(||e||^2)."""

    squared = np.sum(residuals * residuals, axis=1) / (scale_px * scale_px)
    if loss == "huber":
        rho = np.where(squared <= 1.0, squared, 2.0 * np.sqrt(squared) - 1.0)
    elif loss == "cauchy":
        rho = np.log1p(squared)
    else:
        raise ValueError(f"unsupported block robust loss: {loss!r}")
    ratio = np.ones_like(squared)
    nonzero = squared > np.finfo(np.float64).eps
    ratio[nonzero] = np.sqrt(rho[nonzero] / squared[nonzero])
    return (residuals * ratio[:, None]).reshape(-1)


def _grouped_metrics(
    parameters: np.ndarray,
    layout: _ParameterLayout,
    data: RigCalibrationObservations,
    observations: Iterable[RigTargetObservation],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    cameras, poses = _unpack_parameters(parameters, layout)
    by_camera: dict[str, list[float]] = defaultdict(list)
    by_pose: dict[str, list[float]] = defaultdict(list)
    for observation in observations:
        projected, in_front = project_target_points(
            observation.object_points_m,
            poses[observation.pose_id],
            cameras[observation.camera_id],
            data.projection_models[observation.camera_id],
        )
        errors = np.linalg.norm(projected - observation.image_points_px, axis=1)
        errors[~in_front | ~np.isfinite(errors)] = 1e4
        by_camera[observation.camera_id].extend(errors.tolist())
        by_pose[observation.pose_id].extend(errors.tolist())
    return (
        {
            key: reprojection_metrics(values)
            for key, values in sorted(by_camera.items())
        },
        {key: reprojection_metrics(values) for key, values in sorted(by_pose.items())},
    )


def _validate_solution_transforms(
    cameras: dict[str, np.ndarray], poses: dict[str, np.ndarray]
) -> None:
    for camera_id, matrix in cameras.items():
        validate_transform(matrix, name=f"T_workspace_from_camera[{camera_id!r}]")
    for pose_id, matrix in poses.items():
        validate_transform(matrix, name=f"T_workspace_from_target[{pose_id!r}]")


def _failure_reasons(
    optimizer_success: bool,
    final_metrics: dict[str, Any],
    config: RigCalibrationConfig,
    observability: dict[str, Any],
    parameter_count: int,
    max_condition_number: float,
) -> list[str]:
    reasons: list[str] = []
    if not optimizer_success:
        reasons.append("optimizer_did_not_converge")
    if (
        final_metrics["p95_px"] is None
        or final_metrics["p95_px"] > config.final_reprojection_p95_px
    ):
        reasons.append("final_reprojection_p95_exceeds_gate")
    if observability["rank"] != parameter_count:
        reasons.append("rank_deficient_jacobian")
    condition = observability["condition_number"]
    if condition is None or condition > max_condition_number:
        reasons.append("jacobian_condition_number_exceeds_gate")
    return reasons
