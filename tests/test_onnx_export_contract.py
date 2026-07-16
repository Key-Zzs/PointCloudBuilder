from __future__ import annotations

import onnx
import pytest
from onnx import TensorProto, helper

from ffs_reproduction.exporters.onnx_export import ONNX_EXPORT_CONTRACT, validate_onnx_contract


def _write_model(path, *, dynamic: bool = False) -> None:
    shape = [1, 3, 480, 640]
    if dynamic:
        shape = [1, 3, 480, 0]
    left = helper.make_tensor_value_info("left_image", TensorProto.FLOAT, shape)
    right = helper.make_tensor_value_info("right_image", TensorProto.FLOAT, shape)
    disparity = helper.make_tensor_value_info("disparity", TensorProto.FLOAT, [1, 1, 480, 640])
    node = helper.make_node("Identity", ["left_image"], ["disparity"])
    graph = helper.make_graph([node], "single", [left, right], [disparity])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, path)


def test_export_contract_pins_legacy_static_exporter_and_checks_io(tmp_path) -> None:
    assert ONNX_EXPORT_CONTRACT["dynamo"] is False
    path = tmp_path / "single.onnx"
    _write_model(path)
    result = validate_onnx_contract(
        path,
        input_names=("left_image", "right_image"),
        output_names=("disparity",),
    )
    assert result["opset"] == 17
    assert result["input_shapes"] == [[1, 3, 480, 640], [1, 3, 480, 640]]


def test_export_contract_rejects_dynamic_shape(tmp_path) -> None:
    path = tmp_path / "dynamic.onnx"
    _write_model(path, dynamic=True)
    with pytest.raises(ValueError, match="static positive shape"):
        validate_onnx_contract(
            path,
            input_names=("left_image", "right_image"),
            output_names=("disparity",),
        )


def test_target_tensorrt_parser_accepts_checked_onnx_when_available(tmp_path) -> None:
    trt = pytest.importorskip("tensorrt")
    path = tmp_path / "single.onnx"
    _write_model(path)
    logger = trt.Logger(trt.Logger.ERROR)
    network = trt.Builder(logger).create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    assert parser.parse_from_file(str(path)), [str(parser.get_error(i)) for i in range(parser.num_errors)]
