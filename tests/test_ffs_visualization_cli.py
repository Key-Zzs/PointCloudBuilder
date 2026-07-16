from __future__ import annotations

import subprocess
import sys

import numpy as np
import torch

from pointcloud_builder.frame_io import load_frame
from scripts.visualize_ffs_stereo_pipeline import save_pipeline_artifacts


def test_shared_loader_supports_stereo_npz(tmp_path) -> None:
    path = tmp_path / "stereo.npz"
    np.savez(path, left_ir=np.zeros((2, 2), dtype=np.uint8), right_ir=np.ones((2, 2), dtype=np.uint8), color=np.zeros((2, 2, 3), dtype=np.uint8))
    frame = load_frame(path)
    assert set(("left_ir", "right_ir", "rgb")) <= set(frame)
    assert "color" not in frame


def test_stereo_no_show_writes_all_offline_artifacts(tmp_path) -> None:
    h, w = 4, 4
    perception = {
        "left_ir": torch.zeros((h, w)),
        "right_ir": torch.ones((h, w)),
        "disparity": torch.full((h, w), 2.0),
        "depth": torch.full((h, w), 1.0),
        "valid_mask": torch.ones((h, w), dtype=torch.bool),
        "raw": torch.ones((8, 3)),
        "cropped": torch.ones((4, 3)),
        "sampled": torch.ones((2, 3)),
    }
    result = save_pipeline_artifacts(perception, {"ffs": {"timing_ms": {"inference": 1.0}}}, tmp_path, no_show=True)
    assert result["point_counts"] == {"raw": 8, "cropped": 4, "sampled": 2}
    for name in (
        "left_ir.png",
        "right_ir.png",
        "disparity.npy",
        "disparity.png",
        "depth_m.npy",
        "depth.png",
        "valid_mask.png",
        "invalid_disparity_mask.png",
        "remove_invisible_mask.png",
        "z_range_invalid_mask.png",
        "raw.ply",
        "cropped.ply",
        "sampled.ply",
        "metadata.json",
        "timing.json",
        "stage_counts.json",
    ):
        assert (tmp_path / name).is_file(), name
    assert result["stage_counts"]["visualization_contract"] == {"denoise_cloud": False, "zfar_m": 100.0, "zfar_applied": False}


def test_legacy_visualizer_no_show_does_not_require_gui(tmp_path) -> None:
    config = "configs/example_head_depth_raw.yaml"
    input_path = tmp_path / "frame.npz"
    np.savez(input_path, depth=np.ones((480, 640), dtype=np.float32))
    subprocess.run(
        [sys.executable, "scripts/visualize_raw_pointcloud.py", "--config", config, "--input", str(input_path), "--no-show"],
        check=True,
        capture_output=True,
        text=True,
    )
