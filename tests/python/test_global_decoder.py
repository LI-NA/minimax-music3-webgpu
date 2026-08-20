import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from minimax_music3_webgpu.global_decoder import builder_arguments, validate_global_decoder
from minimax_music3_webgpu.manifest import emit_manifest, _kv_pairs, _source_shard
from minimax_music3_webgpu.paths import ArtifactPaths
from minimax_music3_webgpu import manifest


def test_builder_arguments_target_webgpu_q4(tmp_path) -> None:
    args = builder_arguments(tmp_path / "language_model", tmp_path / "output", tmp_path / "cache")

    assert "-p" in args and args[args.index("-p") + 1] == "int4"
    assert "-e" in args and args[args.index("-e") + 1] == "webgpu"
    assert "exclude_embeds=true" in args
    assert "exclude_lm_head=true" in args
    assert "include_hidden_states=true" not in args
    assert "block_size=128" in args
    assert "accuracy_level=4" in args
    assert "is_symmetric=true" in args
    assert "fuse_qk_norm_gqa=true" in args


def test_builder_arguments_can_disable_qk_norm_fusion_for_cpu_smoke(tmp_path) -> None:
    args = builder_arguments(tmp_path / "language_model", tmp_path / "output", tmp_path / "cache", fuse_qk_norm_gqa=False)

    assert "fuse_qk_norm_gqa=false" in args


@pytest.mark.parametrize("layers", [1, 36])
def test_validate_global_decoder_enforces_cache_and_q4_graph_invariants(tmp_path, layers) -> None:
    model_path = tmp_path / "global_decoder.onnx"
    _write_decoder_fixture(model_path, layers)

    report = validate_global_decoder(model_path, expected_layers=layers)

    assert report.attention_nodes == layers
    assert report.past_inputs == layers * 2
    assert report.present_outputs == layers * 2
    assert report.hidden_output == "hidden_states"


def test_validate_global_decoder_rejects_full_embedding_initializer(tmp_path) -> None:
    model_path = tmp_path / "global_decoder.onnx"
    _write_decoder_fixture(model_path, 1, initializer_name="model.embed_tokens.weight")

    with pytest.raises(ValueError, match="token embedding"):
        validate_global_decoder(model_path, expected_layers=1)


def test_kv_pairs_match_layer_and_cache_identity(tmp_path) -> None:
    path = tmp_path / "decoder.onnx"
    _write_decoder_fixture(path, 2)
    model = onnx.load_model(path)
    pairs = _kv_pairs(model)

    assert pairs == [
        ("past_key_values.0.key", "present.0.key"),
        ("past_key_values.0.value", "present.0.value"),
        ("past_key_values.1.key", "present.1.key"),
        ("past_key_values.1.value", "present.1.value"),
    ]


@pytest.mark.parametrize("target", ["../outside.safetensors", "C:/outside.safetensors"])
def test_source_shard_rejects_unsafe_index_target(tmp_path, target) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"lm_head.weight": target}}))

    with pytest.raises(ValueError, match="safetensors index path"):
        _source_shard(tmp_path, "lm_head.weight")


def test_emit_manifest_rejects_duplicate_and_missing_files(tmp_path) -> None:
    graph = tmp_path / "global_decoder.onnx"
    external = tmp_path / "weights.bin"
    graph.write_bytes(b"graph")
    external.write_bytes(b"weights")
    payload = {
        "graph": graph,
        "external_data": [("weights.bin", external)],
        "embedding_shards": [(0, 1, external)],
        "tokenizer_files": [("tokenizer.json", external)],
        "license_file": external,
        "kv_pairs": [("past_key_values.0.key", "present.0.key")],
    }
    with pytest.raises(ValueError, match="duplicate"):
        emit_manifest(tmp_path / "manifest.json", **payload)
    payload["embedding_shards"] = [(0, 1, tmp_path / "missing.fp16")]
    with pytest.raises(ValueError, match="missing"):
        emit_manifest(tmp_path / "manifest.json", **payload)


@pytest.mark.parametrize(("layers", "release_name"), [(1, "global-one-layer"), (36, "global")])
def test_emit_global_release_assembles_all_browser_artifacts(tmp_path, monkeypatch, layers, release_name) -> None:
    paths = ArtifactPaths(tmp_path, tmp_path / "source", tmp_path / "work", tmp_path / "release", tmp_path / "receipts")
    language = paths.source / "language_model"
    language.mkdir(parents=True)
    (language / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"model.embed_tokens.weight": "model.safetensors", "lm_head.weight": "model.safetensors"}}))
    (language / "model.safetensors").write_bytes(b"weights")
    (paths.source / "tokenizer").mkdir()
    (paths.source / "tokenizer" / "tokenizer.json").write_text("{}")
    (paths.source / "LICENSE").write_text("license")
    release = paths.release / release_name
    packed = paths.work / f"global-packed-{layers}"
    packed.mkdir(parents=True)
    _write_external_decoder_fixture(packed / "global_decoder.onnx")
    monkeypatch.setattr(manifest, "VOCAB_SIZE", 1)
    monkeypatch.setattr(manifest, "HIDDEN_SIZE", 2)

    def export_embedding(_source, output):
        output.mkdir(parents=True)
        file = output / "embedding-000.fp16"
        file.write_bytes(b"fp16")
        return SimpleNamespace(shards=(SimpleNamespace(path=file, row_start=0, row_count=1),))

    def export_head(_source, output):
        output.mkdir(parents=True)
        file = output / "reduced-head.onnx"
        _write_head_fixture(file)
        return SimpleNamespace(model_path=file)

    monkeypatch.setattr(manifest, "export_embedding_table", export_embedding)
    monkeypatch.setattr(manifest, "export_reduced_head", export_head)

    result = manifest.emit_global_release(paths, layers)
    payload = json.loads(result.read_text())

    assert result == release / "manifest.json"
    assert payload["schemaVersion"] == 1
    assert payload["graph"]["externalData"][0]["onnxLocation"] == "weights.bin"
    assert payload["reducedHead"]["path"] == "reduced-head/reduced-head.onnx"
    assert (release / "embedding" / "embedding-000.fp16").is_file()
    assert (release / "reduced-head" / "reduced-head.onnx").is_file()
    assert (release / "tokenizer" / "tokenizer.json").is_file()
    assert (release / "LICENSE").is_file()
    assert (paths.receipts / f"global-release-{layers}.json").is_file()


def test_emit_global_release_keeps_existing_release_when_assembly_fails(tmp_path, monkeypatch) -> None:
    paths = ArtifactPaths(tmp_path, tmp_path / "source", tmp_path / "work", tmp_path / "release", tmp_path / "receipts")
    packed = paths.work / "global-packed-1"
    packed.mkdir(parents=True)
    _write_external_decoder_fixture(packed / "global_decoder.onnx")
    language = paths.source / "language_model"
    language.mkdir(parents=True)
    (language / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"model.embed_tokens.weight": "model.safetensors", "lm_head.weight": "model.safetensors"}}))
    (language / "model.safetensors").write_bytes(b"weights")
    (paths.source / "tokenizer").mkdir()
    (paths.source / "LICENSE").write_text("license")
    release = paths.release / "global-one-layer"
    release.mkdir(parents=True)
    original = release / "existing.bin"
    original.write_bytes(b"unchanged")

    monkeypatch.setattr(manifest, "export_embedding_table", lambda *_: (_ for _ in ()).throw(RuntimeError("packing failed")))

    with pytest.raises(RuntimeError, match="packing failed"):
        manifest.emit_global_release(paths, 1)

    assert original.read_bytes() == b"unchanged"

def test_prompt_contract_has_exact_40_token_rows() -> None:
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "prompt-contract.json").read_text(
            encoding="utf-8"
        )
    )
    conditional = fixture["conditional"]
    unconditional = fixture["unconditional"]

    assert len(conditional) == len(unconditional) == 40
    assert conditional == [151644,151671,11646,198,65,5187,374,220,24,21,198,95275,8778,25407,151672,151673,28463,921,58,4450,921,9707,198,58,6150,355,921,38102,198,80987,198,58,13709,1457,82,10011,60,151674,151645,151669]
    assert unconditional == [151644] + [151654] * 37 + [151645, 151669]


def _write_decoder_fixture(path, layers: int, initializer_name: str = "qweight") -> None:
    inputs = [helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT16, [1, None, 64])]
    outputs = [helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT16, [1, None, 64])]
    nodes = []
    for index in range(layers):
        for cache in ("key", "value"):
            inputs.append(helper.make_tensor_value_info(f"past_key_values.{index}.{cache}", TensorProto.FLOAT16, [1, 2, None, 16]))
            outputs.append(helper.make_tensor_value_info(f"present.{index}.{cache}", TensorProto.FLOAT16, [1, 2, None, 16]))
        nodes.append(helper.make_node("GroupQueryAttention", ["inputs_embeds"], [f"layer_{index}"], domain="com.microsoft"))
    nodes.append(helper.make_node("MatMulNBits", ["inputs_embeds", initializer_name], ["hidden_states"], domain="com.microsoft", bits=4, block_size=128))
    initializer = numpy_helper.from_array(np.zeros((1,), dtype=np.uint8), initializer_name)
    model = helper.make_model(helper.make_graph(nodes, "decoder", inputs, outputs, [initializer]), opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)])
    onnx.save_model(model, path)


def _write_external_decoder_fixture(path) -> None:
    _write_decoder_fixture(path, 1)
    model = onnx.load_model(path)
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True, location="weights.bin", size_threshold=0)


def _write_head_fixture(path) -> None:
    model = helper.make_model(helper.make_graph([], "head", [], [
        helper.make_tensor_value_info("semantic_logits", TensorProto.FLOAT16, [1, 1]),
        helper.make_tensor_value_info("end_logit", TensorProto.FLOAT16, [1, 1]),
    ]))
    onnx.save_model(model, path)
