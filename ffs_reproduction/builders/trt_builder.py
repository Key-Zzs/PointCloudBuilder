"""TensorRT 10 Python Builder API helpers; ``trtexec`` is not used."""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from pointcloud_builder.ffs.tensorrt_common import load_plugin_library, require_tensorrt


def build_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    fp16: bool = True,
    plugin_library: str | Path | None = None,
    workspace_bytes: int | None = None,
    workspace_gib: float = 8.0,
    builder_optimization_level: int = 3,
    config_path: str | Path | None = None,
    error_path: str | Path | None = None,
    metadata_root: str | Path | None = None,
) -> dict[str, Any]:
    """Parse ONNX, build, serialize and inspect one fixed-shape engine.

    A failed precision/tactic build is never retried with another precision.
    When ``error_path`` is supplied, the phase and complete traceback are
    persisted before the original exception is re-raised.
    """

    onnx_path, engine_path = Path(onnx_path).resolve(), Path(engine_path).resolve()
    metadata_root_path = Path(metadata_root).expanduser().resolve() if metadata_root is not None else engine_path.parent
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file does not exist: {onnx_path}")
    requested_workspace_bytes = int(workspace_bytes) if workspace_bytes is not None else int(float(workspace_gib) * (1024**3))
    if requested_workspace_bytes <= 0:
        raise ValueError("workspace_gib/workspace_bytes must be positive")
    if not 0 <= int(builder_optimization_level) <= 5:
        raise ValueError("builder_optimization_level must be between 0 and 5")
    phase = "initialization"
    started = time.perf_counter()
    trt: Any | None = None
    try:
        trt = require_tensorrt()
        if plugin_library is not None:
            phase = "plugin_registration"
            load_plugin_library(plugin_library)
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        phase = "onnx_parser"
        if not parser.parse_from_file(str(onnx_path)):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("TensorRT ONNX parser failed:\n" + "\n".join(errors))
        build_config = builder.create_builder_config()
        build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, requested_workspace_bytes)
        if hasattr(build_config, "set_builder_optimization_level"):
            build_config.set_builder_optimization_level(int(builder_optimization_level))
        elif hasattr(build_config, "builder_optimization_level"):
            build_config.builder_optimization_level = int(builder_optimization_level)
        elif int(builder_optimization_level) != 3:
            raise RuntimeError("This TensorRT Python API has no builder optimization level setter or property")
        if fp16:
            build_config.set_flag(trt.BuilderFlag.FP16)
        phase = "engine_build"
        serialized = builder.build_serialized_network(network, build_config)
        if serialized is None:
            raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")
        payload = bytes(serialized)
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(payload)
        phase = "engine_inspection"
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(payload)
        if engine is None:
            raise RuntimeError(f"TensorRT built an engine that cannot be deserialized: {engine_path}")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        input_names = [network.get_input(i).name for i in range(network.num_inputs)]
        output_names = [network.get_output(i).name for i in range(network.num_outputs)]
        return {
            "status": "success",
            "engine_path": _portable_path(engine_path, metadata_root_path),
            "config_path": _portable_path(Path(config_path).expanduser().resolve(), metadata_root_path) if config_path else None,
            "input_names": input_names,
            "output_names": output_names,
            "input_shapes": [list(network.get_input(i).shape) for i in range(network.num_inputs)],
            "output_shapes": [list(network.get_output(i).shape) for i in range(network.num_outputs)],
            "input_dtypes": [str(network.get_input(i).dtype) for i in range(network.num_inputs)],
            "output_dtypes": [str(network.get_output(i).dtype) for i in range(network.num_outputs)],
            "network_layers": int(getattr(network, "num_layers", 0)),
            "trt_version": getattr(trt, "__version__", "unknown"),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            "precision": "fp16" if fp16 else "fp32",
            "fp16": bool(fp16),
            "builder_optimization_level": int(builder_optimization_level),
            "workspace_bytes": requested_workspace_bytes,
            "workspace_gib": requested_workspace_bytes / (1024**3),
            "build_time_ms": elapsed_ms,
            "engine_size_bytes": len(payload),
            "memory": {
                "activation_device_memory_bytes": _first_int_attr(
                    engine, "device_memory_size_v2", "device_memory_size", "device_memory_size_for_profile"
                ),
                "persistent_device_memory_bytes": _first_int_attr(
                    engine, "persistent_device_memory_size", "persistent_device_memory_size_bytes"
                ),
                "workspace_limit_bytes": requested_workspace_bytes,
            },
            "parser": {"status": "success", "num_errors": 0},
        }
    except Exception as exc:
        if error_path is not None:
            error_file = Path(error_path).expanduser().resolve()
            error_file.parent.mkdir(parents=True, exist_ok=True)
            error_payload = {
                "status": "failed",
                "phase": phase,
                "onnx_path": _portable_path(onnx_path, metadata_root_path),
                "engine_path": _portable_path(engine_path, metadata_root_path),
                "config_path": _portable_path(Path(config_path).expanduser().resolve(), metadata_root_path) if config_path else None,
                "requested_precision": "fp16" if fp16 else "fp32",
                "builder_optimization_level": int(builder_optimization_level),
                "workspace_bytes": requested_workspace_bytes,
                "workspace_gib": requested_workspace_bytes / (1024**3),
                "trt_version": getattr(trt, "__version__", "unavailable"),
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
                "error_type": type(exc).__name__,
                "error": _portable_text(str(exc), metadata_root_path),
                "traceback": _portable_text(traceback.format_exc(), metadata_root_path),
            }
            error_file.write_text(json.dumps(error_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise


def _portable_path(path: Path, metadata_root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), metadata_root.resolve())).as_posix()


def _portable_text(value: str, metadata_root: Path) -> str:
    """Keep complete diagnostics while removing machine-specific path prefixes."""

    value = value.replace(str(metadata_root), ".")
    return value.replace(str(Path.home()), "~")


def _first_int_attr(value: Any, *names: str) -> int | None:
    for name in names:
        candidate = getattr(value, name, None)
        if callable(candidate):
            try:
                candidate = candidate(0)
            except TypeError:
                candidate = None
        if candidate is not None:
            try:
                return int(candidate)
            except (TypeError, ValueError):
                continue
    return None
