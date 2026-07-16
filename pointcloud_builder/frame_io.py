"""Offline frame loader shared by legacy and stereo visualization scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_frame(path: str | Path) -> dict[str, Any]:
    """Load legacy depth/RGB or new left/right-IR/RGB NPZ/NPY frames."""

    input_path = Path(path)
    if input_path.suffix == ".npz":
        with np.load(input_path) as data:
            frame: dict[str, Any] = {}
            for key in ("depth", "left_ir", "right_ir", "rgb", "color", "timestamp", "global_frame_index"):
                if key in data:
                    frame[key] = data[key]
            if "color" in frame and "rgb" not in frame:
                frame["rgb"] = frame.pop("color")
            if not frame:
                raise ValueError(f"NPZ has no supported frame fields: {input_path}")
            return frame
    if input_path.suffix == ".npy":
        data = np.load(input_path, allow_pickle=True)
        if data.shape == () and isinstance(data.item(), dict):
            raw = dict(data.item())
            if "color" in raw and "rgb" not in raw:
                raw["rgb"] = raw.pop("color")
            return raw
        return {"depth": data}
    raise ValueError(f"Unsupported input file extension: {input_path.suffix}")
