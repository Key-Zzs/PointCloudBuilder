"""Small explicit SE(3) helpers using column-vector transform semantics."""

from __future__ import annotations

import math

import numpy as np


def validate_transform(value: np.ndarray, *, name: str = "transform") -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64).copy()
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9, rtol=0.0):
        raise ValueError(f"{name} must have a homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0.0):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-7):
        raise ValueError(f"{name} rotation determinant must be +1")
    matrix.setflags(write=False)
    return matrix


def inverse(T_target_from_source: np.ndarray) -> np.ndarray:
    matrix = np.asarray(T_target_from_source, dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return result


def compose(T_a_from_b: np.ndarray, T_b_from_c: np.ndarray) -> np.ndarray:
    return np.asarray(T_a_from_b, dtype=np.float64) @ np.asarray(
        T_b_from_c, dtype=np.float64
    )


def transform_points(T_target_from_source: np.ndarray, points: np.ndarray) -> np.ndarray:
    matrix = np.asarray(T_target_from_source, dtype=np.float64)
    values = np.asarray(points, dtype=np.float64)
    return values @ matrix[:3, :3].T + matrix[:3, 3]


def transform_error(T_estimated: np.ndarray, T_expected: np.ndarray) -> dict[str, float]:
    delta = np.asarray(T_estimated) @ inverse(np.asarray(T_expected))
    rotation = delta[:3, :3]
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return {
        "translation_mm": float(1000.0 * np.linalg.norm(delta[:3, 3])),
        "rotation_deg": float(math.degrees(math.acos(cosine))),
    }
