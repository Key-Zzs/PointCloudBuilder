"""Quantitative projection parity against the librealsense reference API."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pointcloud_builder.camera_model import CameraIntrinsics
from pointcloud_builder.integrations.camera_rig import (
    calibration_from_camera_bundle,
    ffs_calibration_from_camera_bundle,
)
from pointcloud_builder.integrations.camera_rig.calibration_adapter import (
    resolve_bundle_transform,
)
from pointcloud_builder.local_paths import require_repo_local_path
from pointcloud_builder.projection import (
    ProjectionModelError,
    deproject_pixels,
    project_points,
)

PROJECTION_P95_GATE_PX = 0.10
PROJECTION_MAX_GATE_PX = 0.25


def audit_camera_bundle_projection(
    bundle: Any,
    *,
    camera_label: str,
    bundle_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit one real CameraBundle without exposing device identity on stdout."""

    calibration = calibration_from_camera_bundle(bundle)
    propagation_report = _audit_propagation(bundle, calibration)
    stream_reports: dict[str, Any] = {}
    for stream_name in ("color", "depth", "ir_left", "ir_right"):
        stream_reports[stream_name] = audit_projection_model(
            calibration.intrinsics[stream_name],
            reference_model=bundle.intrinsics[stream_name],
        )
    ffs_calibration = ffs_calibration_from_camera_bundle(bundle)
    ffs_report = _audit_ffs_contract(ffs_calibration, bundle)
    T_color_from_ir_left = calibration.transform(
        calibration.intrinsic_frames["ir_left"],
        calibration.intrinsic_frames["color"],
    ).matrix
    reference_T_color_from_ir_left = resolve_bundle_transform(
        bundle,
        bundle.intrinsics["ir_left"].frame,
        bundle.intrinsics["color"].frame,
    ).matrix
    rgb_report = audit_rgb_projection_chain(
        ffs_calibration.left_intrinsics,
        calibration.intrinsics["color"],
        T_color_from_ir_left,
        ir_reference_model=bundle.intrinsics["ir_left"],
        color_reference_model=bundle.intrinsics["color"],
        reference_T_color_from_ir_left=reference_T_color_from_ir_left,
    )
    propagation_pass = bool(propagation_report["gate_pass"])
    projection_pass = all(
        report["projection"]["gate_pass"] for report in stream_reports.values()
    )
    deprojection_pass = all(
        report["deprojection"]["status"] == "NOT_APPLICABLE"
        or report["deprojection"]["gate_pass"]
        for report in stream_reports.values()
    )
    round_trip_pass = all(
        report["round_trip"]["status"] == "NOT_APPLICABLE"
        or report["round_trip"]["gate_pass"]
        for report in stream_reports.values()
    )
    report: dict[str, Any] = {
        "schema_version": "pointcloud-builder.projection-parity.v1",
        "camera_label": camera_label,
        "bundle_sha256": _sha256_file(bundle_path) if bundle_path is not None else None,
        "reference": "pyrealsense2 rs2_project_point_to_pixel / rs2_deproject_pixel_to_point",
        "grid": {
            "width": 20,
            "height": 15,
            "points_per_depth": len(_pixel_grid(calibration.intrinsics["color"], 20, 15)),
            "depths_m": [0.25, 1.0, 3.0],
            "principal_point_included": True,
        },
        "gates": {
            "projection_p95_px": PROJECTION_P95_GATE_PX,
            "projection_max_px": PROJECTION_MAX_GATE_PX,
        },
        "streams": stream_reports,
        "propagation": propagation_report,
        "rgb_ir_left_to_color": rgb_report,
        "ffs_rectification": ffs_report,
        "acceptance": {
            "CAMERARIG_TO_PCB_CALIBRATION_PROPAGATION": (
                "PASS" if propagation_pass else "FAIL"
            ),
            "RAW_STREAM_PROJECTION_PARITY": "PASS"
            if projection_pass and deprojection_pass and round_trip_pass
            else "FAIL",
            "FFS_RECTIFICATION_CONTRACT": "PASS" if ffs_report["gate_pass"] else "FAIL",
            "RGB_PIXEL_PROJECTION_PARITY": "PASS" if rgb_report["gate_pass"] else "FAIL",
            "DOUBLE_RECTIFICATION": "ABSENT" if ffs_report["gate_pass"] else "PRESENT",
            "MODEL_PARITY": "PASS"
            if propagation_pass
            and projection_pass
            and deprojection_pass
            and round_trip_pass
            and rgb_report["gate_pass"]
            and ffs_report["gate_pass"]
            else "FAIL",
            "FACTORY_INTRINSIC_PHYSICAL_ACCURACY": "NOT_FULLY_VALIDATED",
        },
    }
    return report


def audit_projection_model(
    model: CameraIntrinsics,
    *,
    reference_model: Any | None = None,
    grid_width: int = 20,
    grid_height: int = 15,
    depths_m: Iterable[float] = (0.25, 1.0, 3.0),
) -> dict[str, Any]:
    """Compare one PCB model with the scalar librealsense reference."""

    rs = _realsense()
    reference = _to_realsense_intrinsics(reference_model or model, rs)
    pixels = _pixel_grid(model, grid_width, grid_height)
    depth_values = tuple(depths_m)
    depths = np.repeat(np.asarray(depth_values, dtype=np.float32), len(pixels))
    pixels_tiled = np.tile(pixels, (len(depth_values), 1))
    normalized_x = (pixels_tiled[:, 0] - model.cx) / model.fx
    normalized_y = (pixels_tiled[:, 1] - model.cy) / model.fy
    points = np.column_stack(
        (normalized_x * depths, normalized_y * depths, depths)
    ).astype(np.float32)
    pcb_projection = (
        project_points(torch.from_numpy(points), model).pixels_px.detach().cpu().numpy()
    )
    reference_projection = np.asarray(
        [rs.rs2_project_point_to_pixel(reference, point.tolist()) for point in points],
        dtype=np.float64,
    )
    projection_error = np.linalg.norm(
        pcb_projection.astype(np.float64) - reference_projection, axis=1
    )
    projection_metrics = _metrics(projection_error, unit="px")
    projection_metrics["error_vs_image_radius"] = _radius_metrics(
        pixels_tiled, projection_error, model
    )
    projection_metrics["gate_pass"] = bool(
        projection_metrics["p95"] <= PROJECTION_P95_GATE_PX
        and projection_metrics["max"] <= PROJECTION_MAX_GATE_PX
    )

    try:
        pcb_points = (
            deproject_pixels(
                torch.from_numpy(pixels_tiled), torch.from_numpy(depths), model
            )
            .points_camera.detach()
            .cpu()
            .numpy()
        )
    except ProjectionModelError as error:
        deprojection: dict[str, Any] = {
            "status": "NOT_APPLICABLE",
            "reason": str(error),
        }
        round_trip: dict[str, Any] = {
            "status": "NOT_APPLICABLE",
            "reason": str(error),
        }
    else:
        reference_points = np.asarray(
            [
                rs.rs2_deproject_pixel_to_point(reference, pixel.tolist(), float(depth))
                for pixel, depth in zip(pixels_tiled, depths, strict=True)
            ],
            dtype=np.float64,
        )
        deprojection_error_mm = 1000.0 * np.linalg.norm(
            pcb_points.astype(np.float64) - reference_points, axis=1
        )
        deprojection = _metrics(deprojection_error_mm, unit="mm")
        deprojection["gate_pass"] = bool(deprojection["max"] <= 0.01)
        deprojection["status"] = "PASS" if deprojection["gate_pass"] else "FAIL"
        round_trip_pixels = (
            project_points(torch.from_numpy(pcb_points), model)
            .pixels_px.detach()
            .cpu()
            .numpy()
        )
        round_trip_error = np.linalg.norm(
            round_trip_pixels.astype(np.float64) - pixels_tiled.astype(np.float64), axis=1
        )
        round_trip = _metrics(round_trip_error, unit="px")
        round_trip["gate_pass"] = bool(round_trip["max"] <= PROJECTION_MAX_GATE_PX)
        round_trip["status"] = "PASS" if round_trip["gate_pass"] else "FAIL"

    projection_status = "PASS" if projection_metrics["gate_pass"] else "FAIL"
    return {
        "projection_model": _model_snapshot(model),
        "projection": {"status": projection_status, **projection_metrics},
        "deprojection": deprojection,
        "round_trip": round_trip,
    }


def audit_rgb_projection_chain(
    ir_left_model: CameraIntrinsics,
    color_model: CameraIntrinsics,
    T_color_from_ir_left: np.ndarray,
    *,
    ir_reference_model: Any | None = None,
    color_reference_model: Any | None = None,
    reference_T_color_from_ir_left: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare IR pixel/depth through color projection against librealsense."""

    rs = _realsense()
    ir_reference = _to_realsense_intrinsics(ir_reference_model or ir_left_model, rs)
    color_reference = _to_realsense_intrinsics(
        color_reference_model or color_model, rs
    )
    matrix = np.asarray(T_color_from_ir_left, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("T_color_from_ir_left must be a 4x4 matrix")
    reference_extrinsics = rs.extrinsics()
    reference_matrix = np.asarray(
        reference_T_color_from_ir_left
        if reference_T_color_from_ir_left is not None
        else matrix,
        dtype=np.float64,
    )
    if reference_matrix.shape != (4, 4):
        raise ValueError("reference_T_color_from_ir_left must be a 4x4 matrix")
    reference_extrinsics.rotation = (
        reference_matrix[:3, :3].reshape(-1, order="F").tolist()
    )
    reference_extrinsics.translation = reference_matrix[:3, 3].tolist()
    pixels = _pixel_grid(ir_left_model, 20, 15)
    depths = np.repeat(np.asarray((0.25, 1.0, 3.0), dtype=np.float32), len(pixels))
    pixels = np.tile(pixels, (3, 1))
    pcb_ir = deproject_pixels(
        torch.from_numpy(pixels), torch.from_numpy(depths), ir_left_model
    ).points_camera
    rotation = torch.tensor(matrix[:3, :3].copy(), dtype=pcb_ir.dtype)
    translation = torch.tensor(matrix[:3, 3].copy(), dtype=pcb_ir.dtype)
    pcb_color = pcb_ir @ rotation.T + translation
    pcb_pixels = project_points(pcb_color, color_model).pixels_px.detach().cpu().numpy()
    reference_pixels: list[list[float]] = []
    for pixel, depth in zip(pixels, depths, strict=True):
        point_ir = rs.rs2_deproject_pixel_to_point(
            ir_reference, pixel.tolist(), float(depth)
        )
        point_color = rs.rs2_transform_point_to_point(reference_extrinsics, point_ir)
        reference_pixels.append(
            rs.rs2_project_point_to_pixel(color_reference, point_color)
        )
    reference_array = np.asarray(reference_pixels, dtype=np.float64)
    finite = np.isfinite(pcb_pixels).all(axis=1) & np.isfinite(reference_array).all(axis=1)
    errors = np.linalg.norm(
        pcb_pixels[finite].astype(np.float64) - reference_array[finite], axis=1
    )
    metrics = _metrics(errors, unit="px")
    gate_pass = bool(
        finite.any()
        and metrics["p95"] <= PROJECTION_P95_GATE_PX
        and metrics["max"] <= PROJECTION_MAX_GATE_PX
    )
    metrics.update(
        {
            "valid_count": int(finite.sum()),
            "status": "PASS" if gate_pass else "FAIL",
            "gate_pass": gate_pass,
            "visibility_and_sampling": "NOT_EVALUATED_GEOMETRY_ONLY",
        }
    )
    return metrics


def write_projection_report(report: dict[str, Any], output: str | Path) -> None:
    path = require_repo_local_path(output, label="real projection report")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _audit_propagation(bundle: Any, calibration: Any) -> dict[str, Any]:
    per_stream: dict[str, Any] = {}
    for stream_name in ("color", "depth", "ir_left", "ir_right"):
        source = bundle.intrinsics[stream_name]
        adapted = calibration.intrinsics[stream_name]
        checks = {
            "width": adapted.width == int(source.width),
            "height": adapted.height == int(source.height),
            "fx": adapted.fx == float(source.fx),
            "fy": adapted.fy == float(source.fy),
            "cx": adapted.cx == float(source.cx),
            "cy": adapted.cy == float(source.cy),
            "distortion_model": (
                adapted.distortion_model
                == _normalized_distortion_name(source.distortion_model)
            ),
            "distortion_coeffs": (
                adapted.distortion_coeffs
                == tuple(float(value) for value in source.distortion_coeffs)
            ),
            "frame": adapted.frame == str(source.frame),
            "pixel_geometry_raw": adapted.pixel_geometry == "raw",
        }
        per_stream[stream_name] = {
            "gate_pass": all(checks.values()),
            "checks": checks,
        }
    gate_pass = all(value["gate_pass"] for value in per_stream.values())
    return {
        "status": "PASS" if gate_pass else "FAIL",
        "gate_pass": gate_pass,
        "source": "CameraRig CameraBundle intrinsics",
        "per_stream": per_stream,
    }


def _audit_ffs_contract(calibration: Any, bundle: Any) -> dict[str, Any]:
    left = calibration.left_intrinsics
    right = calibration.right_intrinsics
    source_left = bundle.intrinsics["ir_left"]
    source_right = bundle.intrinsics["ir_right"]
    left_source_match = _rectified_model_matches_identity_source(left, source_left)
    right_source_match = _rectified_model_matches_identity_source(right, source_right)
    gate_pass = bool(
        calibration.rectification_identity
        and left.pixel_geometry == "rectified"
        and right.pixel_geometry == "rectified"
        and left.distortion_model == right.distortion_model == "none"
        and not left.distortion_coeffs
        and not right.distortion_coeffs
        and left_source_match
        and right_source_match
    )
    return {
        "status": "PASS" if gate_pass else "FAIL",
        "gate_pass": gate_pass,
        "rectification_mode": calibration.rectification_mode,
        "rectification_identity": calibration.rectification_identity,
        "resolved_depth_projection_model": _model_snapshot(left),
        "source_to_rectified_left_match": left_source_match,
        "source_to_rectified_right_match": right_source_match,
        "double_rectification": "ABSENT" if gate_pass else "PRESENT",
    }


def _rectified_model_matches_identity_source(derived: Any, source: Any) -> bool:
    return bool(
        derived.width == int(source.width)
        and derived.height == int(source.height)
        and derived.fx == float(source.fx)
        and derived.fy == float(source.fy)
        and derived.cx == float(source.cx)
        and derived.cy == float(source.cy)
        and derived.frame == str(source.frame)
        and derived.pixel_geometry == "rectified"
        and derived.distortion_model == "none"
        and not derived.distortion_coeffs
        and not any(abs(float(value)) > 1e-12 for value in source.distortion_coeffs)
    )


def _pixel_grid(model: CameraIntrinsics, width: int, height: int) -> np.ndarray:
    xs = np.linspace(1.0, model.width - 2.0, width, dtype=np.float32)
    ys = np.linspace(1.0, model.height - 2.0, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    grid = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
    required = np.asarray(
        [
            (model.cx, model.cy),
            ((model.width - 1.0) / 2.0, (model.height - 1.0) / 2.0),
            (1.0, model.cy),
            (model.width - 2.0, model.cy),
            (model.cx, 1.0),
            (model.cx, model.height - 2.0),
        ],
        dtype=np.float32,
    )
    return np.unique(np.vstack((grid, required)), axis=0)


def _metrics(values: np.ndarray, *, unit: str) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0, "unit": unit, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(finite),
        "unit": unit,
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def _radius_metrics(
    pixels: np.ndarray,
    errors: np.ndarray,
    model: CameraIntrinsics,
) -> list[dict[str, Any]]:
    radius = np.linalg.norm(
        np.column_stack(
            (
                (pixels[:, 0] - model.cx) / model.fx,
                (pixels[:, 1] - model.cy) / model.fy,
            )
        ),
        axis=1,
    )
    edges = np.linspace(0.0, float(radius.max()) + 1e-12, 6)
    result: list[dict[str, Any]] = []
    for index in range(5):
        selected = (radius >= edges[index]) & (radius <= edges[index + 1])
        result.append(
            {
                "radius_min": float(edges[index]),
                "radius_max": float(edges[index + 1]),
                **_metrics(errors[selected], unit="px"),
            }
        )
    return result


def _model_snapshot(model: CameraIntrinsics) -> dict[str, Any]:
    return {
        "width": model.width,
        "height": model.height,
        "fx": model.fx,
        "fy": model.fy,
        "cx": model.cx,
        "cy": model.cy,
        "distortion_model": model.distortion_model,
        "distortion_coeffs": list(model.distortion_coeffs),
        "pixel_geometry": model.pixel_geometry,
        "frame": model.frame,
    }


def _to_realsense_intrinsics(model: CameraIntrinsics, rs: Any) -> Any:
    enum_names = {
        "none": "none",
        "brown-conrady": "brown_conrady",
        "modified-brown-conrady": "modified_brown_conrady",
        "inverse-brown-conrady": "inverse_brown_conrady",
        "ftheta": "ftheta",
        "kannala-brandt4": "kannala_brandt4",
    }
    result = rs.intrinsics()
    result.width = model.width
    result.height = model.height
    result.fx = model.fx
    result.fy = model.fy
    result.ppx = model.cx
    result.ppy = model.cy
    distortion_name = _normalized_distortion_name(model.distortion_model)
    result.model = getattr(rs.distortion, enum_names[distortion_name])
    coefficients = list(model.distortion_coeffs)
    result.coeffs = (coefficients + [0.0] * 5)[:5]
    return result


def _normalized_distortion_name(value: object) -> str:
    name = str(value).strip().lower().replace("_", "-")
    aliases = {
        "brown-conrady": "brown-conrady",
        "modified-brown-conrady": "modified-brown-conrady",
        "inverse-brown-conrady": "inverse-brown-conrady",
        "kannala-brandt4": "kannala-brandt4",
        "ftheta": "ftheta",
        "none": "none",
    }
    try:
        return aliases[name]
    except KeyError as error:
        raise ValueError(f"unsupported RealSense distortion model: {value!r}") from error


def _realsense() -> Any:
    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise RuntimeError(
            "projection parity requires the optional pyrealsense2 reference backend"
        ) from error
    return rs


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()
