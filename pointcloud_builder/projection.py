"""Frame-aware camera projection and deprojection.

The formulas intentionally follow librealsense's public projection contract for
models where PCB has an audited implementation.  Unsupported direction/model
combinations fail closed instead of being approximated as a pinhole camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from pointcloud_builder.camera_model import CameraIntrinsics
from pointcloud_builder.types import Tensor


class ProjectionModelError(ValueError):
    """Raised when a projection direction has no audited implementation."""


@dataclass(frozen=True)
class ProjectionResult:
    """Projected pixels and masks with visibility kept out of the contract."""

    pixels_px: Tensor
    finite: Tensor
    in_front: Tensor
    in_bounds: Tensor

    @property
    def valid(self) -> Tensor:
        return self.finite & self.in_front


@dataclass(frozen=True)
class DeprojectionResult:
    """Deprojected camera-frame points and their numerical validity mask."""

    points_camera: Tensor
    valid: Tensor


def project_points(
    points_camera: Tensor,
    projection_model: CameraIntrinsics,
) -> ProjectionResult:
    """Project ``[...,3]`` camera-frame points to ``[...,2]`` pixels.

    This computes geometry only.  It deliberately does not perform z-buffering,
    occlusion handling, or image sampling.
    """

    points = _as_float_tensor(points_camera, final_dimension=3, name="points_camera")
    z = points[..., 2]
    finite_point = torch.isfinite(points).all(dim=-1)
    in_front = finite_point & (z > 0.0)
    safe_z = torch.where(in_front, z, torch.ones_like(z))
    x = points[..., 0] / safe_z
    y = points[..., 1] / safe_z
    x, y = _distort_for_projection(x, y, projection_model)
    u = x * projection_model.fx + projection_model.cx
    v = y * projection_model.fy + projection_model.cy
    pixels = torch.stack((u, v), dim=-1)
    finite = finite_point & torch.isfinite(pixels).all(dim=-1)
    in_bounds = (
        finite
        & in_front
        & (u >= 0.0)
        & (u < projection_model.width)
        & (v >= 0.0)
        & (v < projection_model.height)
    )
    nan = torch.full_like(pixels, float("nan"))
    pixels = torch.where((finite & in_front)[..., None], pixels, nan)
    return ProjectionResult(
        pixels_px=pixels,
        finite=finite,
        in_front=in_front,
        in_bounds=in_bounds,
    )


def deproject_pixels(
    pixels_px: Tensor,
    depth: Tensor,
    projection_model: CameraIntrinsics,
) -> DeprojectionResult:
    """Deproject pixels plus metric optical-axis depth into camera coordinates."""

    pixels = _as_float_tensor(pixels_px, final_dimension=2, name="pixels_px")
    depths = torch.as_tensor(depth, dtype=pixels.dtype, device=pixels.device)
    try:
        pixels, depths = torch.broadcast_tensors(pixels, depths[..., None])
    except RuntimeError as error:
        raise ValueError("pixels_px and depth shapes are not broadcast compatible") from error
    depths = depths[..., 0]
    x = (pixels[..., 0] - projection_model.cx) / projection_model.fx
    y = (pixels[..., 1] - projection_model.cy) / projection_model.fy
    x, y = _undistort_for_deprojection(x, y, projection_model)
    points = torch.stack((depths * x, depths * y, depths), dim=-1)
    valid = (
        torch.isfinite(pixels).all(dim=-1)
        & torch.isfinite(depths)
        & (depths > 0.0)
        & torch.isfinite(points).all(dim=-1)
    )
    points = torch.where(valid[..., None], points, torch.zeros_like(points))
    return DeprojectionResult(points_camera=points, valid=valid)


def _distort_for_projection(
    x: Tensor,
    y: Tensor,
    model: CameraIntrinsics,
) -> tuple[Tensor, Tensor]:
    if model.is_identity_projection:
        return x, y
    coefficients = _five_coefficients(model)
    name = model.distortion_model
    if name == "brown-conrady":
        return _brown_forward(x, y, coefficients, modified=False)
    if name == "modified-brown-conrady":
        return _brown_forward(x, y, coefficients, modified=True)
    if name == "inverse-brown-conrady":
        # This is deliberately librealsense API parity, including its use of
        # the modified-Brown forward path for this enum.
        return _brown_forward(x, y, coefficients, modified=True)
    if name == "ftheta":
        r = torch.sqrt(x * x + y * y)
        coefficient = coefficients[0]
        denominator = math.atan(2.0 * math.tan(coefficient / 2.0))
        if abs(denominator) <= 1e-12:
            raise ProjectionModelError("ftheta coefficient produces a singular projection")
        rd = (1.0 / coefficient) * torch.atan(2.0 * r * math.tan(coefficient / 2.0))
        scale = torch.where(r > 1e-12, rd / r, torch.ones_like(r))
        return x * scale, y * scale
    if name == "kannala-brandt4":
        r = torch.sqrt(x * x + y * y)
        theta = torch.atan(r)
        theta2 = theta * theta
        series = 1.0 + theta2 * (
            coefficients[0]
            + theta2
            * (coefficients[1] + theta2 * (coefficients[2] + theta2 * coefficients[3]))
        )
        rd = theta * series
        scale = torch.where(r > 1e-12, rd / r, torch.ones_like(r))
        return x * scale, y * scale
    raise ProjectionModelError(f"unsupported projection model: {name!r}")


def _undistort_for_deprojection(
    x: Tensor,
    y: Tensor,
    model: CameraIntrinsics,
) -> tuple[Tensor, Tensor]:
    if model.is_identity_projection:
        return x, y
    coefficients = _five_coefficients(model)
    name = model.distortion_model
    if name == "brown-conrady":
        return _brown_deprojection(x, y, coefficients)
    if name == "modified-brown-conrady":
        raise ProjectionModelError(
            "non-identity modified-brown-conrady deprojection is not supported by "
            "the librealsense public contract"
        )
    if name == "inverse-brown-conrady":
        return _inverse_brown_deprojection(x, y, coefficients)
    if name == "ftheta":
        rd = torch.sqrt(x * x + y * y)
        coefficient = coefficients[0]
        rd_safe = torch.clamp(rd, min=torch.finfo(rd.dtype).eps)
        r = torch.tan(coefficient * rd_safe) / math.atan(2.0 * math.tan(coefficient / 2.0))
        scale = torch.where(rd > 1e-12, r / rd_safe, torch.ones_like(rd))
        return x * scale, y * scale
    if name == "kannala-brandt4":
        rd = torch.sqrt(x * x + y * y)
        theta = rd.clone()
        for _ in range(4):
            theta2 = theta * theta
            f = theta * (
                1.0
                + theta2
                * (
                    coefficients[0]
                    + theta2
                    * (
                        coefficients[1]
                        + theta2 * (coefficients[2] + theta2 * coefficients[3])
                    )
                )
            ) - rd
            derivative = 1.0 + theta2 * (
                3.0 * coefficients[0]
                + theta2
                * (
                    5.0 * coefficients[1]
                    + theta2 * (7.0 * coefficients[2] + 9.0 * theta2 * coefficients[3])
                )
            )
            theta = theta - f / derivative
        r = torch.tan(theta)
        scale = torch.where(rd > 1e-12, r / rd, torch.ones_like(rd))
        return x * scale, y * scale
    raise ProjectionModelError(f"unsupported deprojection model: {name!r}")


def _brown_forward(
    x: Tensor,
    y: Tensor,
    coefficients: tuple[float, float, float, float, float],
    *,
    modified: bool,
) -> tuple[Tensor, Tensor]:
    k1, k2, p1, p2, k3 = coefficients
    r2 = x * x + y * y
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    if modified:
        xr = x * radial
        yr = y * radial
        return (
            xr + 2.0 * p1 * xr * yr + p2 * (r2 + 2.0 * xr * xr),
            yr + 2.0 * p2 * xr * yr + p1 * (r2 + 2.0 * yr * yr),
        )
    return (
        x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
        y * radial + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y),
    )


def _brown_deprojection(
    xd: Tensor,
    yd: Tensor,
    coefficients: tuple[float, float, float, float, float],
) -> tuple[Tensor, Tensor]:
    """Mirror librealsense's ten-step Brown-Conrady deprojection path."""

    k1, k2, p1, p2, k3 = coefficients
    x = xd.clone()
    y = yd.clone()
    for _ in range(10):
        r2 = x * x + y * y
        inverse_radial = 1.0 / (1.0 + r2 * (k1 + r2 * (k2 + r2 * k3)))
        delta_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        delta_y = 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)
        x = (xd - delta_x) * inverse_radial
        y = (yd - delta_y) * inverse_radial
    return x, y


def _inverse_brown_deprojection(
    xd: Tensor,
    yd: Tensor,
    coefficients: tuple[float, float, float, float, float],
) -> tuple[Tensor, Tensor]:
    """Mirror librealsense's ten-step inverse-Brown deprojection path."""

    k1, k2, p1, p2, k3 = coefficients
    x = xd.clone()
    y = yd.clone()
    for _ in range(10):
        r2 = x * x + y * y
        inverse_radial = 1.0 / (1.0 + r2 * (k1 + r2 * (k2 + r2 * k3)))
        xq = x / inverse_radial
        yq = y / inverse_radial
        delta_x = 2.0 * p1 * xq * yq + p2 * (r2 + 2.0 * xq * xq)
        delta_y = 2.0 * p2 * xq * yq + p1 * (r2 + 2.0 * yq * yq)
        x = (xd - delta_x) * inverse_radial
        y = (yd - delta_y) * inverse_radial
    return x, y


def _five_coefficients(
    model: CameraIntrinsics,
) -> tuple[float, float, float, float, float]:
    coefficients = model.distortion_coeffs
    if len(coefficients) != 5:
        raise ProjectionModelError(
            f"{model.distortion_model} requires exactly five distortion coefficients"
        )
    return coefficients  # type: ignore[return-value]


def _as_float_tensor(value: Tensor, *, final_dimension: int, name: str) -> Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 0 or tensor.shape[-1] != final_dimension:
        raise ValueError(f"{name} must have final dimension {final_dimension}")
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32)
    return tensor
