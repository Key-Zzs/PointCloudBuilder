from __future__ import annotations

import numpy as np

from pointcloud_builder.rig_calibration.physical_acceptance import (
    NCameraAcceptanceThresholds,
    aggregate_ncamera_evaluations,
    evaluate_ncamera_alignment,
    generate_pairs,
)


def _plane(x_low: float = -0.3, x_high: float = 0.3) -> np.ndarray:
    x, y = np.meshgrid(
        np.arange(x_low, x_high + 0.005, 0.01),
        np.arange(-0.2, 0.205, 0.01),
        indexing="xy",
    )
    return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))


def _thresholds(minimum_overlap_points: int = 100) -> NCameraAcceptanceThresholds:
    return NCameraAcceptanceThresholds(
        maximum_overlap_distance_mm=15.0,
        minimum_overlap_points=minimum_overlap_points,
        maximum_symmetric_median_mm=0.2,
        maximum_symmetric_p95_mm=0.2,
        maximum_board_median_mm=0.2,
        maximum_board_p95_mm=0.2,
        maximum_plane_offset_mm=0.2,
        maximum_normal_split_deg=0.2,
        maximum_double_layer_thickness_mm=0.2,
        maximum_diagnostic_translation_mm=0.2,
        maximum_diagnostic_rotation_deg=0.2,
        voxel_size_mm=2.5,
    )


def _evaluate(clouds, *, no_overlap=None, minimum_overlap_points=100):
    masks = {name: np.ones(len(points), dtype=bool) for name, points in clouds.items()}
    return evaluate_ncamera_alignment(
        clouds,
        thresholds=_thresholds(minimum_overlap_points),
        board_masks=masks,
        declared_no_overlap_pairs=no_overlap,
    )


def test_pair_generation_is_generic_for_two_three_and_four_cameras() -> None:
    assert generate_pairs(("left", "right")) == (("left", "right"),)
    assert generate_pairs(("rear", "left", "right")) == (
        ("left", "rear"),
        ("left", "right"),
        ("rear", "right"),
    )
    assert len(generate_pairs(("one", "two", "three", "four"))) == 6


def test_two_camera_physical_acceptance_passes() -> None:
    cloud = _plane()
    report = _evaluate({"north": cloud, "south": cloud.copy()})
    assert report["passed"] is True
    assert list(report["per_pair"]) == ["north__south"]


def test_three_camera_all_pairs_and_order_invariance() -> None:
    cloud = _plane()
    forward = _evaluate({"cam-z": cloud, "left": cloud.copy(), "overhead": cloud.copy()})
    reverse = _evaluate({"overhead": cloud.copy(), "left": cloud, "cam-z": cloud.copy()})
    assert forward == reverse
    assert forward["passed"] is True
    assert len(forward["per_pair"]) == 3
    assert len(forward["all_rig"]["accepted_pairwise_overlap_edges"]) == 3


def test_three_camera_chain_overlap_is_connected() -> None:
    # A/B share [-0.10, 0.10], B/C share [0.90, 1.10], A/C have no overlap.
    clouds = {
        "alpha": _plane(-1.0, 0.1),
        "bridge": _plane(-0.1, 1.1),
        "charlie": _plane(0.9, 2.0),
    }
    report = _evaluate(
        clouds,
        no_overlap={("alpha", "charlie")},
        minimum_overlap_points=80,
    )
    assert report["passed"] is True
    assert report["per_pair"]["alpha__charlie"]["status"] == (
        "NOT_APPLICABLE_NO_OVERLAP"
    )
    assert report["all_rig"]["pairwise_overlap_connectivity"]["connected"] is True


def test_disconnected_overlap_graph_fails() -> None:
    clouds = {
        "one": _plane(-1.0, -0.5),
        "two": _plane(-1.0, -0.5),
        "three": _plane(0.5, 1.0),
        "four": _plane(0.5, 1.0),
    }
    report = _evaluate(
        clouds,
        no_overlap={
            ("one", "three"),
            ("one", "four"),
            ("two", "three"),
            ("two", "four"),
        },
    )
    assert report["passed"] is False
    assert report["gates"]["connected_overlap_graph"] is False


def test_four_camera_smoke_produces_six_pairs() -> None:
    cloud = _plane()
    names = ("front", "rear", "port", "starboard")
    report = _evaluate({name: cloud.copy() for name in names})
    assert report["passed"] is True
    assert len(report["per_pair"]) == 6


def test_pose_local_sequence_aggregation_keeps_generic_pair_contract() -> None:
    cloud = _plane()
    thresholds = _thresholds()
    evaluations = [
        _evaluate({"alpha": cloud.copy(), "bridge": cloud.copy(), "zulu": cloud.copy()})
        for _ in range(3)
    ]
    residuals = {
        key: {"translation_norm_mm": 0.0, "rotation_geodesic_deg": 0.0}
        for key in evaluations[0]["per_pair"]
    }
    report = aggregate_ncamera_evaluations(
        evaluations,
        thresholds=thresholds,
        diagnostic_residuals=residuals,
    )
    assert report["passed"] is True
    assert report["evaluated_matched_set_count"] == 3
    assert len(report["per_pair"]) == 3
    assert all(
        value["diagnostic_residual_se3"]["written_back"] is False
        for value in report["per_pair"].values()
    )
