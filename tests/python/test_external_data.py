from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from minimax_music3_webgpu.external_data import repack_external_data


def test_repack_external_data_copies_each_initializer_without_crossing_shards(
    tmp_path: Path,
) -> None:
    source_data = b"abcd" + b"efghij" + b"klmnopqrst"
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(source_data)
    initializers = [
        _external_tensor("first", 0, 4),
        _external_tensor("second", 4, 6),
        _external_tensor("third", 10, 10),
    ]
    model_path = tmp_path / "source.onnx"
    onnx.save_model(_model(initializers), model_path)

    repacked = repack_external_data(model_path, tmp_path / "packed", max_file_bytes=12)
    model = onnx.load_model(repacked.model_path, load_external_data=False)

    assert len(repacked.shards) == 2
    for initializer in model.graph.initializer:
        location, offset, length = _external_fields(initializer)
        assert offset + length <= 12
        assert (repacked.model_path.parent / location).read_bytes()[offset : offset + length] == source_data[
            {"first": 0, "second": 4, "third": 10}[initializer.name] : {"first": 4, "second": 10, "third": 20}[initializer.name]
        ]


def test_repack_external_data_rejects_initializer_larger_than_limit(tmp_path: Path) -> None:
    (tmp_path / "source.bin").write_bytes(b"0123456789abc")
    model_path = tmp_path / "source.onnx"
    onnx.save_model(_model([_external_tensor("too_large", 0, 13)]), model_path)

    with pytest.raises(ValueError, match="too_large"):
        repack_external_data(model_path, tmp_path / "packed", max_file_bytes=12)


def _external_tensor(name: str, offset: int, length: int) -> TensorProto:
    tensor = TensorProto(name=name, data_type=TensorProto.UINT8)
    tensor.dims.extend([length])
    tensor.data_location = TensorProto.EXTERNAL
    tensor.external_data.extend(
        [
            onnx.StringStringEntryProto(key="location", value="source.bin"),
            onnx.StringStringEntryProto(key="offset", value=str(offset)),
            onnx.StringStringEntryProto(key="length", value=str(length)),
        ]
    )
    return tensor


def _model(initializers: list[TensorProto]) -> onnx.ModelProto:
    graph = helper.make_graph([], "test", [], [], initializer=initializers)
    return helper.make_model(graph)


def _external_fields(tensor: TensorProto) -> tuple[str, int, int]:
    fields = {entry.key: entry.value for entry in tensor.external_data}
    return fields["location"], int(fields["offset"]), int(fields["length"])
