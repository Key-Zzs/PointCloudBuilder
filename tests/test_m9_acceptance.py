from __future__ import annotations

import pytest

from pointcloud_builder.mapping.acceptance import (
    REQUIRED_FFS_PRODUCTION_GATES,
    evaluate_m9_production_acceptance,
)
from pointcloud_builder.mapping.provenance import (
    validate_production_ffs_provenance,
)


def _gates(value: bool = True) -> dict[str, bool]:
    return {name: value for name in REQUIRED_FFS_PRODUCTION_GATES}


def test_native_degraded_and_large_board_deferred_do_not_block_production() -> None:
    report = evaluate_m9_production_acceptance(
        _gates(), legacy_target_validated=True, native_status="DEGRADED_GEOMETRY"
    )
    assert report["deployment_target"]["large_board_500x700"] == {
        "role": "future_deployment_preset",
        "status": "DEFERRED",
        "production_required": False,
    }
    assert report["M9_NATIVE_OPTIONAL_BASELINE"] == "DEGRADED_GEOMETRY"
    assert report["OVERALL_PRODUCTION_STATUS"] == "PASS"


def test_required_ffs_failure_still_fails_closed() -> None:
    gates = _gates()
    gates["ffs_offline_tsdf"] = False
    report = evaluate_m9_production_acceptance(gates, legacy_target_validated=True)
    assert report["M9_FFS_PRODUCTION_MAPPING"] == "FAIL"
    assert report["OVERALL_PRODUCTION_STATUS"] == "FAIL"


def test_current_legacy_target_remains_a_required_deployment_gate() -> None:
    report = evaluate_m9_production_acceptance(
        _gates(), legacy_target_validated=False, native_status="PASS"
    )
    assert report["M8.5_DEPLOYMENT_READINESS"] == "FAIL"
    assert report["OVERALL_PRODUCTION_STATUS"] == "FAIL"


def test_native_not_run_is_non_blocking() -> None:
    report = evaluate_m9_production_acceptance(
        _gates(), legacy_target_validated=True, native_status="NOT_RUN"
    )
    assert report["M9_NATIVE_OPTIONAL_BASELINE"] == "NOT_RUN"
    assert report["OVERALL_PRODUCTION_STATUS"] == "PASS"


def test_production_backend_provenance_rejects_non_plugin_backend() -> None:
    provenance = {
        "camera_a": {
            "depth_source": "ffs_stereo",
            "backend": "pytorch",
            "precision": "fp16",
            "artifact_id": "candidate",
            "pipeline_config_sha256": "a" * 64,
        }
    }
    with pytest.raises(ValueError, match="tensorrt_plugin"):
        validate_production_ffs_provenance(provenance, ("camera_a",))
