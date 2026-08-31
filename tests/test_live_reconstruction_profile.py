from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from pointcloud_builder.rig import load_rig_config

ROOT = Path(__file__).parents[1]
PATH = ROOT / "tools/mapping/run_live_reconstruction_profile.py"
SPEC = importlib.util.spec_from_file_location("run_live_reconstruction_profile", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("profile", ["raw", "dense", "compact"])
def test_public_profile_satisfies_live_acceptance_contract(profile: str) -> None:
    path = (
        ROOT
        / "configs/mapping"
        / {
            "raw": "raw_rgb_concatenation_example.yaml",
            "dense": "dense_rgb_reconstruction_example.yaml",
            "compact": "compact_rgb_reconstruction_example.yaml",
        }[profile]
    )
    MODULE._validate_profile_config(load_rig_config(path), profile)


def test_profile_mismatch_fails_before_hardware_access() -> None:
    dense = load_rig_config(
        ROOT / "configs/mapping/dense_rgb_reconstruction_example.yaml"
    )
    with pytest.raises(ValueError, match="raw profile"):
        MODULE._validate_profile_config(dense, "raw")


def test_three_camera_dense_profile_is_not_rejected_by_camera_count() -> None:
    config = load_rig_config(
        ROOT / "configs/mapping/live_rig_three_camera_example.yaml"
    )
    MODULE._validate_profile_config(config, "dense")


def test_rgb_metrics_and_exact_row_preservation() -> None:
    source = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0, 0.2, 0.0], [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]]
    )
    selected = source[[1]]
    metrics = MODULE._point_metrics(selected)
    assert metrics["channels"] == 6
    assert metrics["finite"]
    assert metrics["rgb_in_unit_interval"]
    assert MODULE._rows_are_subset(selected, source)
    assert not MODULE._rows_are_subset(selected + 0.01, source)
