import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from diffusers.models.transformers.minimax_music3_rvq_depth_decoder import MiniMaxMusic3RVQDepthDecoder
from onnx import TensorProto

from minimax_music3_webgpu.rvq_depth import (
    _expected_shapes,
    export_feedback_graph,
    export_rvq_embedding_table,
    export_rvq_depth_from_state_dict,
    inspect_rvq_source,
    validate_rvq_metadata,
    validate_rvq_graph,
)
from minimax_music3_webgpu import rvq_depth
from minimax_music3_webgpu.manifest import emit_rvq_manifest
from minimax_music3_webgpu.embedding import EmbeddingShard, EmbeddingTableReceipt
from minimax_music3_webgpu.external_data import repack_external_data


def test_rvq_metadata_requires_exact_topology_and_original_key_mapping() -> None:
    config = {
        "hidden_size": 4096,
        "num_layers": 4,
        "num_attention_heads": 16,
        "intermediate_size": 6144,
        "audio_vocab_size": 1024,
        "num_codebooks": 8,
        "max_position_embeddings": 16,
    }
    shapes = _expected_shapes(config)

    validate_rvq_metadata(config, shapes)

    assert len(shapes) == 47
    assert shapes["audio_embeddings.weight"] == (7168, 4096)
    assert shapes["layers.3.down_proj.weight"] == (4096, 6144)
    assert [shapes[f"audio_heads.{index}.weight"] for index in range(7)] == [(1024, 4096)] * 7


@pytest.mark.real
def test_real_rvq_source_matches_pinned_receipt() -> None:
    config = Path("artifacts/source/rvq_depth_decoder/config.json")
    weights = Path("artifacts/source/rvq_depth_decoder/diffusion_pytorch_model.safetensors")
    receipt = Path("artifacts/receipts/source-acoustic.json")
    if not all(path.is_file() for path in (config, weights, receipt)):
        pytest.skip("pinned acoustic source is not downloaded")
    report = inspect_rvq_source(
        config_path=config,
        weights_path=weights,
    )

    assert report.config == {
        "hidden_size": 4096,
        "num_layers": 4,
        "num_attention_heads": 16,
        "intermediate_size": 6144,
        "audio_vocab_size": 1024,
        "num_codebooks": 8,
        "max_position_embeddings": 16,
    }
    assert report.tensor_count == 47
    assert report.audio_embedding_shape == (7168, 4096)
    assert report.head_shapes == ((1024, 4096),) * 7
    assert report.source_sha256 == json.loads(
        receipt.read_text(encoding="utf-8")
    )["files"][3]["sha256"]


def test_zero_padded_fixed_graph_matches_prefix_reference_for_lengths_2_through_8(tmp_path) -> None:
    contract = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "rvq-contract.json").read_text(encoding="utf-8")
    )
    torch.manual_seed(contract["seed"])
    model = MiniMaxMusic3RVQDepthDecoder(
        hidden_size=contract["hiddenSize"],
        num_layers=contract["layers"],
        num_attention_heads=contract["heads"],
        intermediate_size=contract["intermediateSize"],
        audio_vocab_size=contract["vocabSize"],
        num_codebooks=8,
        max_position_embeddings=16,
    ).eval()
    graph = tmp_path / "rvq-depth.onnx"
    export_rvq_depth_from_state_dict(
        model.state_dict(),
        graph,
        hidden_size=contract["hiddenSize"],
        intermediate_size=contract["intermediateSize"],
        num_heads=contract["heads"],
        num_layers=contract["layers"],
        vocab_size=contract["vocabSize"],
    )
    session = ort.InferenceSession(graph.read_bytes(), providers=["CPUExecutionProvider"])
    generator = np.random.default_rng(contract["seed"])
    global_hidden = generator.standard_normal((2, contract["hiddenSize"])).astype(np.float16)
    semantic = generator.standard_normal((2, contract["hiddenSize"])).astype(np.float16)
    residuals = generator.standard_normal((2, 6, contract["hiddenSize"])).astype(np.float16)

    with torch.no_grad():
        for depth_index in range(7):
            raw = np.concatenate(
                [global_hidden[:, None], semantic[:, None], residuals[:, :depth_index]], axis=1
            )
            projected = model.projection(torch.from_numpy(raw).to(torch.float32))
            hidden = model(projected)[:, -1]
            expected_logits = torch.stack([head(hidden) for head in model.audio_heads], dim=1).numpy()
            padded = residuals.copy()
            padded[:, depth_index:] = 0
            actual_hidden, actual_logits = session.run(
                None,
                {
                    "global_last_hidden": global_hidden,
                    "semantic_embedding": semantic,
                    "residual_embeddings": padded,
                    "depth_index": np.array(depth_index, dtype=np.int32),
                },
            )
            np.testing.assert_allclose(actual_hidden, hidden.numpy(), rtol=contract["relativeTolerance"], atol=contract["absoluteTolerance"])
            np.testing.assert_allclose(actual_logits, expected_logits, rtol=contract["relativeTolerance"], atol=contract["absoluteTolerance"])


def test_exported_graph_has_fixed_topology_and_feedback_matches_reference(tmp_path) -> None:
    contract = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "rvq-contract.json").read_text(encoding="utf-8")
    )
    torch.manual_seed(contract["seed"])
    model = MiniMaxMusic3RVQDepthDecoder(
        hidden_size=contract["hiddenSize"],
        num_layers=4,
        num_attention_heads=contract["heads"],
        intermediate_size=contract["intermediateSize"],
        audio_vocab_size=contract["vocabSize"],
        num_codebooks=8,
        max_position_embeddings=16,
    )
    rvq = export_rvq_depth_from_state_dict(
        model.state_dict(), tmp_path / "rvq-depth.onnx",
        hidden_size=contract["hiddenSize"], intermediate_size=contract["intermediateSize"],
        num_heads=contract["heads"], num_layers=4, vocab_size=contract["vocabSize"],
    )
    report = validate_rvq_graph(rvq, hidden_size=contract["hiddenSize"], vocab_size=contract["vocabSize"])
    assert report.softmax_nodes == 4
    assert report.rms_norms == 9
    assert report.head_matmuls == 7
    assert not report.dynamic_shape_ops

    feedback = export_feedback_graph(tmp_path / "feedback.onnx", hidden_size=contract["hiddenSize"])
    session = ort.InferenceSession(feedback.read_bytes(), providers=["CPUExecutionProvider"])
    generator = np.random.default_rng(contract["seed"])
    semantic = generator.standard_normal((2, 1, contract["hiddenSize"])).astype(np.float16)
    residual = generator.standard_normal((2, 7, contract["hiddenSize"])).astype(np.float16)
    actual = session.run(None, {"semantic_rows": semantic, "residual_rows": residual})[0]
    expected = (semantic.astype(np.float32) + residual.astype(np.float32).sum(axis=1, keepdims=True)) / np.sqrt(8)
    np.testing.assert_allclose(actual, expected.astype(np.float16), rtol=0.002, atol=0.002)


def test_rvq_depth_index_path_is_int32(tmp_path) -> None:
    torch.manual_seed(1)
    model = MiniMaxMusic3RVQDepthDecoder(
        hidden_size=8, num_layers=4, num_attention_heads=2, intermediate_size=12,
        audio_vocab_size=5, num_codebooks=8, max_position_embeddings=16,
    )
    graph = export_rvq_depth_from_state_dict(
        model.state_dict(), tmp_path / "rvq-depth.onnx",
        hidden_size=8, intermediate_size=12, num_heads=2, num_layers=4, vocab_size=5,
    )

    inferred = onnx.shape_inference.infer_shapes(onnx.load(graph))
    inputs = {value.name: value.type.tensor_type.elem_type for value in inferred.graph.input}
    values = {value.name: value.type.tensor_type.elem_type for value in inferred.graph.value_info}
    initializers = {value.name: value.data_type for value in inferred.graph.initializer}

    assert inputs["depth_index"] == TensorProto.INT32
    assert initializers["one"] == TensorProto.INT32
    assert values["current_position"] == TensorProto.INT32


def test_audio_embedding_table_uses_row_shard_receipt_and_exact_fp16_layout(tmp_path, monkeypatch) -> None:
    rows = np.arange(7 * 4, dtype=np.float32).reshape(7, 4)

    class Source:
        def get_slice(self, name):
            assert name == "audio_embeddings.weight"
            return Slice()

    class Slice:
        def get_shape(self):
            return rows.shape

        def __getitem__(self, index):
            return torch.from_numpy(rows[index])

    class Context:
        def __enter__(self):
            return Source()

        def __exit__(self, *_):
            return None

    monkeypatch.setattr(rvq_depth, "safe_open", lambda *_, **__: Context())
    receipt = export_rvq_embedding_table(
        tmp_path / "source.safetensors", tmp_path / "embedding", rows=7, columns=4, max_file_bytes=4 * 2 * 3
    )

    assert [(item.row_start, item.row_count) for item in receipt.shards] == [(0, 3), (3, 3), (6, 1)]
    restored = np.concatenate([
        np.fromfile(item.path, dtype=np.float16).reshape(item.row_count, 4) for item in receipt.shards
    ])
    np.testing.assert_array_equal(restored, rows.astype(np.float16))


def test_rvq_manifest_declares_both_gpu_outputs_and_exact_row_table(tmp_path) -> None:
    torch.manual_seed(1)
    model = MiniMaxMusic3RVQDepthDecoder(
        hidden_size=8, num_layers=4, num_attention_heads=2, intermediate_size=12,
        audio_vocab_size=5, num_codebooks=8, max_position_embeddings=16,
    )
    graph = export_rvq_depth_from_state_dict(
        model.state_dict(), tmp_path / "rvq-depth.onnx",
        hidden_size=8, intermediate_size=12, num_heads=2, num_layers=4, vocab_size=5,
    )
    feedback = export_feedback_graph(tmp_path / "feedback.onnx", hidden_size=8)
    table_file = tmp_path / "embedding.fp16"
    table_file.write_bytes(bytes(7 * 8 * 2))
    table = EmbeddingTableReceipt((
        EmbeddingShard(table_file, 0, 7, 8, 16, table_file.stat().st_size, rvq_depth._sha256(table_file)),
    ))

    manifest_path = emit_rvq_manifest(
        tmp_path / "manifest.json", rvq_depth=graph, feedback=feedback, embedding=table,
        rows=7, columns=8,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["rvqDepth"]["gpuOutputs"] == ["depth_hidden"]
    assert payload["feedback"]["gpuOutputs"] == ["inputs_embeds"]
    assert payload["rvqEmbedding"] == {
        "rows": 7,
        "columns": 8,
        "rowBytes": 16,
        "shards": [{
            "path": "embedding.fp16", "bytes": 112,
            "sha256": rvq_depth._sha256(table_file), "rowStart": 0, "rowCount": 7,
        }],
    }


def test_rvq_repack_keeps_shape_constants_inline_for_ort_shape_inference(tmp_path) -> None:
    torch.manual_seed(2)
    model = MiniMaxMusic3RVQDepthDecoder(
        hidden_size=8, num_layers=4, num_attention_heads=2, intermediate_size=12,
        audio_vocab_size=5, num_codebooks=8, max_position_embeddings=16,
    )
    source = export_rvq_depth_from_state_dict(
        model.state_dict(), tmp_path / "raw" / "rvq-depth.onnx",
        hidden_size=8, intermediate_size=12, num_heads=2, num_layers=4, vocab_size=5,
        external_data=True,
    )
    packed = repack_external_data(source, tmp_path / "packed", 1024, inline_threshold=32)

    session = ort.InferenceSession(str(packed.model_path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[3].name == "depth_index"
