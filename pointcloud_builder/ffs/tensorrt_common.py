"""Small TensorRT 10 runtime wrapper with persistent contexts and buffers."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, Mapping

import torch

from pointcloud_builder.ffs.manifest import sha256_file


_LOADED_PLUGIN_LIBRARIES: list[ctypes.CDLL] = []


def require_tensorrt() -> Any:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT is required for this FFS backend; install the pinned TensorRT 10 CUDA 13 package in dp3"
        ) from exc
    return trt


def load_plugin_library(path: str | Path) -> ctypes.CDLL:
    """Load a plugin before parser/engine operations and verify its registry entry."""

    trt = require_tensorrt()
    plugin_path = Path(path).expanduser().resolve()
    if not plugin_path.is_file():
        raise FileNotFoundError(f"FFS plugin library does not exist: {plugin_path}")
    init_plugins = getattr(trt, "init_libnvinfer_plugins", None)
    if init_plugins is not None:
        init_plugins(trt.Logger(trt.Logger.WARNING), "")
    library = ctypes.CDLL(str(plugin_path), mode=ctypes.RTLD_GLOBAL)
    register = getattr(library, "ffs_register_gwc_plugin", None)
    if register is None:
        register = getattr(library, "registerFFSGWCPlugin", None)
    if register is not None:
        register.restype = ctypes.c_bool
        if not bool(register()):
            raise RuntimeError(f"FFS plugin registration function failed: {plugin_path}")
    _LOADED_PLUGIN_LIBRARIES.append(library)
    registry = trt.get_plugin_registry()
    creator = registry.get_plugin_creator("FFSGWCVolume", "1", "")
    if creator is None:
        raise RuntimeError("FFSGWCVolume v1 plugin creator is not registered after library load")
    return library


class TensorRTEngine:
    """Deserialize one fixed-shape TensorRT 10 engine and reuse its buffers."""

    def __init__(
        self,
        path: str | Path,
        *,
        input_shapes: Mapping[str, tuple[int, ...]],
        expected_inputs: tuple[str, ...],
        expected_outputs: tuple[str, ...],
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"TensorRT engine does not exist: {self.path}")
        self.sha256 = sha256_file(self.path)
        self.trt = require_tensorrt()
        logger = self.trt.Logger(self.trt.Logger.WARNING)
        with self.path.open("rb") as handle:
            serialized = handle.read()
        self.runtime = self.trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(serialized)
        if self.engine is None:
            raise RuntimeError(f"TensorRT failed to deserialize engine: {self.path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"TensorRT failed to create execution context: {self.path}")
        self.input_names = tuple(
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == self.trt.TensorIOMode.INPUT
        )
        self.output_names = tuple(
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == self.trt.TensorIOMode.OUTPUT
        )
        if self.input_names != expected_inputs:
            raise ValueError(f"Engine inputs {self.input_names} != required {expected_inputs}")
        if self.output_names != expected_outputs:
            raise ValueError(f"Engine outputs {self.output_names} != required {expected_outputs}")
        self._outputs: dict[str, torch.Tensor] = {}
        for name in self.input_names:
            if name not in input_shapes:
                raise ValueError(f"Missing fixed input shape for TensorRT tensor {name}")
            shape = tuple(int(x) for x in input_shapes[name])
            if not self.context.set_input_shape(name, shape):
                raise ValueError(f"TensorRT rejected input shape {name}={shape}")
        for name in self.output_names:
            shape = tuple(int(x) for x in self.context.get_tensor_shape(name))
            if any(x <= 0 for x in shape):
                raise ValueError(f"TensorRT output shape is not fixed after setup: {name}={shape}")
            self._outputs[name] = torch.empty(shape, device="cuda", dtype=self.torch_dtype(self.engine.get_tensor_dtype(name)))

    @staticmethod
    def torch_dtype(data_type: Any) -> torch.dtype:
        trt = require_tensorrt()
        mapping = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.BF16: torch.bfloat16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BOOL: torch.bool,
        }
        if data_type not in mapping:
            raise TypeError(f"Unsupported TensorRT dtype: {data_type}")
        return mapping[data_type]

    def infer(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if set(inputs) != set(self.input_names):
            raise ValueError(f"TensorRT inputs {tuple(inputs)} != engine inputs {self.input_names}")
        for name in self.input_names:
            tensor = inputs[name]
            if tensor.device.type != "cuda":
                raise ValueError(f"TensorRT input {name} must be CUDA, got {tensor.device}")
            tensor = tensor.contiguous()
            expected_shape = tuple(int(x) for x in self.context.get_tensor_shape(name))
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(f"TensorRT input {name} shape {tuple(tensor.shape)} != {expected_shape}")
            expected_dtype = self.torch_dtype(self.engine.get_tensor_dtype(name))
            if tensor.dtype != expected_dtype:
                tensor = tensor.to(dtype=expected_dtype)
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        for name, output in self._outputs.items():
            self.context.set_tensor_address(name, int(output.data_ptr()))
        stream = torch.cuda.current_stream().cuda_stream
        if not self.context.execute_async_v3(stream):
            raise RuntimeError(f"TensorRT execution failed for {self.path}")
        return self._outputs


def timed_cuda_sections(fn: Any) -> tuple[Any, dict[str, float]]:
    """Run a function and return CUDA-event elapsed sections when available."""

    if not torch.cuda.is_available():
        import time

        start = time.perf_counter()
        result = fn()
        return result, {"total_ms": (time.perf_counter() - start) * 1000.0}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    return result, {"total_ms": float(start.elapsed_time(end))}
