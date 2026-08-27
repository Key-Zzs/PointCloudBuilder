"""Offline frame loader shared by legacy and stereo visualization scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_frame(path: str | Path) -> dict[str, Any]:
    """Load legacy or CameraRig NPZ/NPY frames into PCB's canonical keys."""

    input_path = Path(path)
    if input_path.suffix == ".npz":
        with np.load(input_path) as data:
            frame: dict[str, Any] = {}
            for key in (
                "depth",
                "left_ir",
                "right_ir",
                "ir_left",
                "ir_right",
                "rgb",
                "color",
                "timestamp",
                "global_frame_index",
            ):
                if key in data:
                    frame[key] = data[key]
            _normalize_camera_rig_keys(frame)
            if not frame:
                raise ValueError(f"NPZ has no supported frame fields: {input_path}")
            return frame
    if input_path.suffix == ".npy":
        data = np.load(input_path, allow_pickle=True)
        if data.shape == () and isinstance(data.item(), dict):
            raw = dict(data.item())
            _normalize_camera_rig_keys(raw)
            return raw
        return {"depth": data}
    raise ValueError(f"Unsupported input file extension: {input_path.suffix}")


def _normalize_camera_rig_keys(frame: dict[str, Any]) -> None:
    aliases = {
        "color": "rgb",
        "ir_left": "left_ir",
        "ir_right": "right_ir",
    }
    for source, target in aliases.items():
        if source in frame and target not in frame:
            frame[target] = frame.pop(source)
