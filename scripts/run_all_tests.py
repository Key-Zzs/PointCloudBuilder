"""Run the package test suite."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    result = subprocess.run([sys.executable, "-m", "pytest"], check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
