from __future__ import annotations

import numpy as np
import pytest
import torch

rs = pytest.importorskip("pyrealsense2")

from pointcloud_builder.camera_model import CameraIntrinsics  # noqa: E402
from pointcloud_builder.projection import deproject_pixels, project_points  # noqa: E402
from pointcloud_builder.projection_parity import (  # noqa: E402
    _pixel_grid,
    audit_projection_model,
)

MODELS = [
    ("brown-conrady", rs.distortion.brown_conrady, (0.1, -0.03, 0.001, -0.002, 0.005)),
    (
        "modified-brown-conrady",
        rs.distortion.modified_brown_conrady,
        (0.1, -0.03, 0.001, -0.002, 0.005),
    ),
    (
        "inverse-brown-conrady",
        rs.distortion.inverse_brown_conrady,
        (0.1, -0.03, 0.001, -0.002, 0.005),
    ),
    ("ftheta", rs.distortion.ftheta, (0.7, 0.0, 0.0, 0.0, 0.0)),
    (
        "kannala-brandt4",
        rs.distortion.kannala_brandt4,
        (0.01, -0.002, 0.0003, -0.00004, 0.0),
    ),
]


def _models(name: str, enum: object, coefficients: tuple[float, ...]):
    pcb = CameraIntrinsics(
        640,
        480,
        615.25,
        613.75,
        319.5,
        239.5,
        name,
        coefficients,
        "raw",
        "synthetic/optical",
    )
    reference = rs.intrinsics()
    reference.width = pcb.width
    reference.height = pcb.height
    reference.fx = pcb.fx
    reference.fy = pcb.fy
    reference.ppx = pcb.cx
    reference.ppy = pcb.cy
    reference.model = enum
    reference.coeffs = (list(coefficients) + [0.0] * 5)[:5]
    return pcb, reference


@pytest.mark.parametrize(("name", "enum", "coefficients"), MODELS)
def test_projection_matches_librealsense_nonzero_models(
    name: str, enum: object, coefficients: tuple[float, ...]
) -> None:
    pcb, reference = _models(name, enum, coefficients)
    points = np.asarray(
        [[-0.4, -0.3, 1.0], [-0.2, 0.25, 0.8], [0.0, 0.0, 1.0], [0.35, 0.2, 1.2]],
        dtype=np.float32,
    )
    actual = project_points(torch.from_numpy(points), pcb).pixels_px.numpy()
    expected = np.asarray(
        [rs.rs2_project_point_to_pixel(reference, point.tolist()) for point in points]
    )
    np.testing.assert_allclose(actual, expected, atol=6e-5, rtol=0.0)


@pytest.mark.parametrize(
    ("name", "enum", "coefficients"),
    [MODELS[0], MODELS[2], MODELS[3], MODELS[4]],
)
def test_applicable_deprojection_matches_librealsense_nonzero_models(
    name: str, enum: object, coefficients: tuple[float, ...]
) -> None:
    pcb, reference = _models(name, enum, coefficients)
    pixels = np.asarray(
        [[5.0, 5.0], [160.0, 120.0], [319.5, 239.5], [634.0, 474.0]],
        dtype=np.float32,
    )
    actual = deproject_pixels(
        torch.from_numpy(pixels), torch.ones(len(pixels)), pcb
    ).points_camera.numpy()
    expected = np.asarray(
        [rs.rs2_deproject_pixel_to_point(reference, pixel.tolist(), 1.0) for pixel in pixels]
    )
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0.0)


def test_strong_brown_edge_deprojection_matches_librealsense_gate() -> None:
    coefficients = (0.4, 0.2, 0.02, -0.02, 0.1)
    pcb, reference = _models(
        "brown-conrady", rs.distortion.brown_conrady, coefficients
    )
    pixels = _pixel_grid(pcb, 20, 15)
    actual = deproject_pixels(
        torch.from_numpy(pixels), torch.ones(len(pixels)), pcb
    ).points_camera.numpy()
    expected = np.asarray(
        [rs.rs2_deproject_pixel_to_point(reference, pixel.tolist(), 1.0) for pixel in pixels]
    )
    error_mm = 1000.0 * np.linalg.norm(actual - expected, axis=1)
    assert float(error_mm.max()) <= 0.01


def test_reference_model_is_independent_and_mismatch_fails_gate() -> None:
    adapted, _reference = _models("none", rs.distortion.none, ())
    source = CameraIntrinsics(
        **{
            **adapted.__dict__,
            "fx": adapted.fx + 5.0,
        }
    )
    report = audit_projection_model(adapted, reference_model=source)
    assert report["projection"]["status"] == "FAIL"
    assert report["projection"]["gate_pass"] is False


def test_parity_grid_explicitly_contains_principal_point() -> None:
    model, _reference = _models("none", rs.distortion.none, ())
    grid = _pixel_grid(model, 20, 15)
    assert np.any(np.all(grid == np.asarray([model.cx, model.cy]), axis=1))
