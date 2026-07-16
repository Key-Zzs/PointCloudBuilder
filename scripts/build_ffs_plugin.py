#!/usr/bin/env python3
"""Configure/build the FFS GWC plugin with an explicit SM120 target."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import sysconfig
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensorrt-root", type=Path, default=None)
    parser.add_argument("--tensorrt-python-dir", type=Path, default=Path(sysconfig.get_paths()["purelib"]))
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "ffs_reproduction/build")
    parser.add_argument("--cuda-arch", default="120")
    args = parser.parse_args()
    source_dir = REPO_ROOT / "ffs_reproduction/cpp"
    build_dir = args.build_dir.expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    configure = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}",
        f"-DTENSORRT_PYTHON_DIR={args.tensorrt_python_dir.expanduser().resolve()}",
    ]
    cuda_root = Path(sys.executable).resolve().parent.parent
    cuda_compiler = Path(sys.executable).resolve().parent / "nvcc"
    if cuda_compiler.is_file():
        configure.append(f"-DCMAKE_CUDA_COMPILER={cuda_compiler}")
    if (cuda_root / "targets/x86_64-linux/include/cuda_runtime.h").is_file():
        configure.append(f"-DCUDAToolkit_ROOT={cuda_root}")
    if args.tensorrt_root is not None:
        configure.append(f"-DTENSORRT_ROOT={args.tensorrt_root.expanduser().resolve()}")
    env = os.environ.copy()
    if cuda_compiler.is_file():
        env["CUDACXX"] = str(cuda_compiler)
    subprocess.run(configure, check=True, cwd=REPO_ROOT, env=env)
    subprocess.run(["cmake", "--build", str(build_dir), "--target", "ffs_gwc_plugin", "-j2"], check=True, cwd=REPO_ROOT, env=env)
    library = build_dir / "libffs_gwc_plugin.so"
    if not library.is_file():
        raise FileNotFoundError(f"Plugin build did not produce {library}")
    print(library)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
