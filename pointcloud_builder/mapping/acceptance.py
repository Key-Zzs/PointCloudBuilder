"""Fail-closed production acceptance with an explicitly optional native baseline."""

from __future__ import annotations

from typing import Any


REQUIRED_FFS_PRODUCTION_GATES = (
    "ffs_recording",
    "ffs_offline_tsdf",
    "ffs_save_load",
    "ffs_live_static_map",
    "freeze",
    "dynamic_overlay",
    "guarded_synthetic",
    "rerun",
    "postprocess",
    "timing",
)


def evaluate_m9_production_acceptance(
    production_gates: dict[str, bool],
    *,
    legacy_target_validated: bool,
    native_status: str = "DEGRADED_GEOMETRY",
    historical_native_board_thickness_mm: float = 3.342,
) -> dict[str, Any]:
    """Return independent production and optional-native verdicts.

    The 500 x 700 mm board is recorded as a deferred deployment preset and is
    deliberately absent from the required production gate set.
    """

    unknown = sorted(set(production_gates) - set(REQUIRED_FFS_PRODUCTION_GATES))
    missing = sorted(set(REQUIRED_FFS_PRODUCTION_GATES) - set(production_gates))
    if unknown or missing:
        raise ValueError(
            f"production gate contract mismatch; missing={missing}, unknown={unknown}"
        )
    if any(not isinstance(value, bool) for value in production_gates.values()):
        raise ValueError("production gates must be booleans")
    if native_status not in {"PASS", "DEGRADED_GEOMETRY", "NOT_RUN"}:
        raise ValueError(
            "native baseline status must be PASS, DEGRADED_GEOMETRY, or NOT_RUN"
        )
    production_passed = legacy_target_validated and all(production_gates.values())
    return {
        "schema_version": "pointcloud-builder.m9-production-acceptance.v1",
        "deployment_target": {
            "current": "validated_legacy_charuco",
            "status": "PASS" if legacy_target_validated else "FAIL",
            "large_board_500x700": {
                "role": "future_deployment_preset",
                "status": "DEFERRED",
                "production_required": False,
            },
        },
        "production_tsdf": {
            "depth_source": "ffs_stereo",
            "recommended_backend": "tensorrt_plugin",
            "gates": dict(production_gates),
            "status": "PASS" if production_passed else "FAIL",
        },
        "native_tsdf": {
            "role": "optional_baseline",
            "production_required": False,
            "status": native_status,
            "historical_board_thickness_mm": historical_native_board_thickness_mm,
        },
        "M8.5_DEPLOYMENT_READINESS": (
            "PASS" if legacy_target_validated else "FAIL"
        ),
        "M9_FFS_PRODUCTION_MAPPING": "PASS" if production_passed else "FAIL",
        "M9_NATIVE_OPTIONAL_BASELINE": native_status,
        "OVERALL_PRODUCTION_STATUS": "PASS" if production_passed else "FAIL",
    }
