"""Lazy dependency gate for the official Rerun SDK distribution."""

from __future__ import annotations

import importlib
import importlib.metadata
from typing import Any


def require_rerun() -> Any:
    """Import the official SDK only inside the logger process."""

    try:
        version = importlib.metadata.version("rerun-sdk")
        module = importlib.import_module("rerun")
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise RuntimeError(
            'Rerun visualization requires the official extra: pip install ".[rerun]"'
        ) from error
    if getattr(module, "__version__", None) != version:
        raise RuntimeError(
            "rerun module version differs from the rerun-sdk distribution"
        )
    return module
