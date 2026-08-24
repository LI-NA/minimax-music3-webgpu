import runpy
from pathlib import Path

import onnx
from onnx import helper


def test_generated_matmul_nbits_fixture_uses_the_b32_symmetric_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)

    runpy.run_path(str(repository / "tools" / "generate-browser-fixtures.py"))

    model = onnx.load_model(tmp_path / "test-fixtures" / "matmul-nbits.onnx")
    node = next(node for node in model.graph.node if node.op_type == "MatMulNBits")
    attributes = {item.name: helper.get_attribute_value(item) for item in node.attribute}
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    assert attributes == {
        "K": 128,
        "N": 2,
        "accuracy_level": 4,
        "bits": 4,
        "block_size": 32,
    }
    assert tuple(initializers["weight"].dims) == (2, 4, 16)
    assert tuple(initializers["scales"].dims) == (2, 4)
