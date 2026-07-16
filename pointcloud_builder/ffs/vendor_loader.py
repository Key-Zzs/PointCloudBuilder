"""Scoped importer for the original FFS pickle module names."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def vendor_root() -> Path:
    return Path(__file__).resolve().parents[2] / "ffs_reproduction" / "vendor"


def _is_ffs_module(name: str) -> bool:
    return name == "core" or name.startswith("core.") or name == "Utils" or name.startswith("foundation_stereo_ori")


@contextmanager
def scoped_vendor_imports(root: Path | None = None) -> Iterator[None]:
    """Temporarily expose vendor modules under the names used by the pickle.

    Existing modules are restored byte-for-byte and modules imported only by
    this context are removed when it exits.  Model objects retain their class
    objects after loading; their forward path does not require a global FFS
    source checkout or a permanent ``sys.path`` entry.
    """

    root = (root or vendor_root()).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"FFS vendor root does not exist: {root}")
    old_path = list(sys.path)
    saved = {name: module for name, module in sys.modules.items() if _is_ffs_module(name)}
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        for name in list(sys.modules):
            if _is_ffs_module(name):
                if name in saved:
                    sys.modules[name] = saved[name]
                else:
                    sys.modules.pop(name, None)
        sys.path[:] = old_path
