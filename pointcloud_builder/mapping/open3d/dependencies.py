"""Lazy optional dependency boundary for Open3D."""

from __future__ import annotations

import importlib
import importlib.metadata
from typing import Any


def require_open3d() -> Any:
    try:
        version = importlib.metadata.version("open3d")
        module = importlib.import_module("open3d")
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise RuntimeError(
            'TSDF mapping requires the optional extra: pip install ".[tsdf]"'
        ) from error
    if version != "0.19.0" or getattr(module, "__version__", None) != version:
        raise RuntimeError("TSDF backend is validated only with open3d==0.19.0")
    return module
