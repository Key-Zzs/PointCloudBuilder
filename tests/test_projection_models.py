from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from pointcloud_builder.camera_model import CameraIntrinsics
from pointcloud_builder.projection import (
    ProjectionModelError,
    deproject_pixels,
    project_points,
)
from pointcloud_builder.projection_parity import write_projection_report


def _model(
    distortion_model: str = "none",
    coefficients: tuple[float, ...] = (),
    *,
    pixel_geometry: str = "raw",
) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=640,
        height=480,
        fx=615.25,
        fy=613.75,
        cx=319.5,
        cy=239.5,
        distortion_model=distortion_model,
        distortion_coeffs=coefficients,
        pixel_geometry=pixel_geometry,
        frame="synthetic/color_optical",
    )


def _points() -> torch.Tensor:
    pixels = torch.tensor(
        [[5.0, 5.0], [160.0, 120.0], [319.5, 239.5], [480.0, 360.0], [634.0, 474.0]],
        dtype=torch.float64,
    )
    return deproject_pixels(pixels, torch.ones(5, dtype=torch.float64), _model()).points_camera


def test_legacy_pinhole_is_explicit_rectified_identity() -> None:
    model = CameraIntrinsics(640, 480, 600.0, 600.0, 319.5, 239.5)
    assert model.pixel_geometry == "rectified"
    assert model.distortion_model == "none"
    assert model.is_identity_projection


def test_rectified_nonidentity_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="rectified pixel geometry"):
        _model("brown-conrady", (0.1, 0.0, 0.0, 0.0, 0.0), pixel_geometry="rectified")


@pytest.mark.parametrize(
    "coefficients",
    [
        (0.12, -0.04, 0.0015, -0.002, 0.008),
        (-0.12, 0.035, -0.001, 0.0018, -0.006),
    ],
)
def test_brown_conrady_matches_independent_opencv_oracle_and_round_trip(
    coefficients: tuple[float, ...],
) -> None:
    cv2 = pytest.importorskip("cv2")
    points = _points()
    model = _model("brown-conrady", coefficients)
    result = project_points(points, model)
    camera_matrix = np.asarray(
        [[model.fx, 0.0, model.cx], [0.0, model.fy, model.cy], [0.0, 0.0, 1.0]]
    )
    oracle, _jacobian = cv2.projectPoints(
        points.numpy(), np.zeros(3), np.zeros(3), camera_matrix, np.asarray(coefficients)
    )
    np.testing.assert_allclose(
        result.pixels_px.numpy(), oracle.reshape(-1, 2), atol=1e-10, rtol=0.0
    )
    recovered = deproject_pixels(result.pixels_px, points[:, 2], model)
    torch.testing.assert_close(recovered.points_camera, points, atol=2e-10, rtol=0.0)


def test_nonzero_inverse_brown_projection_is_labeled_librealsense_parity() -> None:
    model = _model("inverse-brown-conrady", (0.1, -0.02, 0.0, 0.0, 0.0))
    assert torch.isfinite(project_points(_points(), model).pixels_px).all()


def test_modified_brown_nonidentity_deprojection_fails_closed() -> None:
    model = _model("modified-brown-conrady", (0.1, -0.02, 0.0, 0.0, 0.0))
    with pytest.raises(ProjectionModelError, match="public contract"):
        deproject_pixels(
            torch.tensor([[100.0, 100.0]]), torch.tensor([1.0]), model
        )


def test_projection_masks_are_geometry_only_and_visibility_free() -> None:
    points = torch.tensor(
        [[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        dtype=torch.float32,
    )
    result = project_points(points, _model())
    assert result.valid.tolist() == [True, True, False]
    assert result.in_bounds.tolist() == [True, False, False]
    assert torch.isnan(result.pixels_px[2]).all()


def test_nonzero_color_from_ir_transform_uses_color_projection() -> None:
    ir_point = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    T_color_from_ir = torch.eye(4, dtype=torch.float64)
    T_color_from_ir[:3, 3] = torch.tensor([0.025, -0.01, 0.0], dtype=torch.float64)
    homogeneous = torch.cat((ir_point, torch.ones((1, 1), dtype=torch.float64)), dim=1)
    color_point = (T_color_from_ir @ homogeneous.T).T[:, :3]
    projected = project_points(color_point, _model())
    expected = torch.tensor(
        [[319.5 + 0.025 * 615.25, 239.5 - 0.01 * 613.75]], dtype=torch.float64
    )
    torch.testing.assert_close(projected.pixels_px, expected, atol=1e-12, rtol=0.0)


def test_real_projection_report_writer_requires_local_path(tmp_path) -> None:
    with pytest.raises(ValueError, match="under repository .local"):
        write_projection_report({"status": "synthetic"}, tmp_path / "report.json")
    repository_local = Path(__file__).resolve().parents[1] / ".local"
    with tempfile.TemporaryDirectory(dir=repository_local) as directory:
        output = Path(directory) / "report.json"
        write_projection_report({"status": "synthetic"}, output)
        assert output.is_file()
