from __future__ import annotations

from typing import Any

from camera_rig.targets.charuco.quality import CharucoQualityThresholds

from tools.calibration import capture_multicamera_target_poses as capture


def test_production_detector_uses_uncertainty_validated(monkeypatch: Any) -> None:
    recorded: dict[str, object] = {}

    class FakeDetector:
        def __init__(self, target: object, *, thresholds: object) -> None:
            recorded["target"] = target
            recorded["thresholds"] = thresholds

    monkeypatch.setattr(capture, "CharucoDetector", FakeDetector)
    target = object()

    detector = capture._production_detector(target)

    assert isinstance(detector, FakeDetector)
    assert recorded["target"] is target
    thresholds = recorded["thresholds"]
    assert isinstance(thresholds, CharucoQualityThresholds)
    assert thresholds.policy == "uncertainty_validated"
    assert thresholds.minimum_charuco_corners == 12
    assert thresholds.minimum_corner_fraction == 0.5
    assert thresholds.absolute_minimum_coverage_ratio == 0.0
    assert thresholds.minimum_marker_perimeter_px == 20.0


def test_pose_capture_result_reports_each_camera_without_device_identity() -> None:
    pose_log = {
        "pose_id": "pose_3",
        "cameras": {
            "camera_b": {
                "accepted": False,
                "corner_count": 8,
                "quality": {"failure_reasons": ["INSUFFICIENT_CORNERS"]},
            },
            "camera_a": {
                "accepted": True,
                "corner_count": 24,
                "quality": {"failure_reasons": []},
            },
        },
    }

    assert capture._pose_capture_result(pose_log) == (
        "POSE_RESULT=pose_3; camera_a=ACCEPTED(corners=24); "
        "camera_b=REJECTED(corners=8,reasons=INSUFFICIENT_CORNERS)"
    )
