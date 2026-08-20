import numpy as np
import onnx
import onnxruntime as ort
import torch

from minimax_music3_webgpu import reduced_head
from minimax_music3_webgpu.reduced_head import export_reduced_head, select_reduced_head_rows


def test_select_reduced_head_rows_preserves_exact_semantic_and_end_rows() -> None:
    weights = np.arange(20 * 3, dtype=np.float16).reshape(20, 3)

    semantic, end = select_reduced_head_rows(
        weights,
        semantic_start=5,
        semantic_count=4,
        end_token_id=2,
    )

    np.testing.assert_array_equal(semantic, weights[5:9])
    np.testing.assert_array_equal(end, weights[2:3])


def test_export_reduced_head_uses_ort_compatible_opset_and_external_initializers(
    tmp_path, monkeypatch
) -> None:
    weights = np.arange(8 * 2, dtype=np.float32).reshape(8, 2)
    monkeypatch.setattr(reduced_head, "HIDDEN_SIZE", 2)
    monkeypatch.setattr(reduced_head, "SEMANTIC_TOKEN_START", 3)
    monkeypatch.setattr(reduced_head, "SEMANTIC_TOKEN_COUNT", 3)
    monkeypatch.setattr(reduced_head, "AUDIO_END_TOKEN_ID", 1)
    monkeypatch.setattr(reduced_head, "ARTIFACT_FILE_LIMIT", 16)
    monkeypatch.setattr(reduced_head, "safe_open", _safe_open(weights))

    exported = export_reduced_head(tmp_path / "source.safetensors", tmp_path / "packed")
    model = onnx.load_model(exported.model_path, load_external_data=False)
    session = ort.InferenceSession(str(exported.model_path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"hidden_states": np.array([[[1, 2], [3, 4]]], dtype=np.float16)})

    assert model.opset_import[0].version == 13
    initializers = {item.name: item for item in model.graph.initializer}
    assert {"semantic_weight", "end_weight"} <= initializers.keys()
    for name in ("semantic_weight", "end_weight"):
        assert initializers[name].data_location == onnx.TensorProto.EXTERNAL
        assert initializers[name].raw_data == b""
    np.testing.assert_array_equal(outputs[0], np.array([[46, 60, 74]], dtype=np.float16))
    np.testing.assert_array_equal(outputs[1], np.array([[18]], dtype=np.float16))
    np.testing.assert_array_equal(outputs[2], np.array([[3, 4]], dtype=np.float16))
    assert session.get_outputs()[2].name == "last_state"
    assert session.get_outputs()[2].shape == [None, 2]


def _safe_open(weights: np.ndarray):
    class Source:
        def get_slice(self, name: str):
            assert name == "lm_head.weight"
            return Slice(weights)

    class Slice:
        def __init__(self, value: np.ndarray) -> None:
            self.value = value

        def get_shape(self) -> tuple[int, int]:
            return self.value.shape

        def __getitem__(self, index):
            return torch.from_numpy(self.value[index])

    class Context:
        def __enter__(self):
            return Source()

        def __exit__(self, *args) -> None:
            return None

    return lambda *args, **kwargs: Context()
