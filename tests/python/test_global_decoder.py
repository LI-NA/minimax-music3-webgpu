import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from minimax_music3_webgpu.global_decoder import builder_arguments, rewrite_attention_mask_for_gqa, validate_global_decoder
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
    assert report.explicit_sequence_inputs == ("seqlens_k", "total_seq_len")


def test_validate_global_decoder_rejects_attention_mask_and_mask_reformat_nodes(tmp_path) -> None:
    model_path = tmp_path / "global_decoder.onnx"
    _write_decoder_fixture(model_path, 1, include_attention_mask=True)

    with pytest.raises(ValueError, match="attention_mask"):
        validate_global_decoder(model_path, expected_layers=1)


def test_validate_global_decoder_requires_explicit_gqa_sequence_inputs(tmp_path) -> None:
    model_path = tmp_path / "global_decoder.onnx"
    _write_decoder_fixture(model_path, 1, explicit_sequence_inputs=False)

    with pytest.raises(ValueError, match="explicit sequence"):
        validate_global_decoder(model_path, expected_layers=1)


def test_validate_global_decoder_rejects_retained_mask_reformat_nodes(tmp_path) -> None:
    model_path = tmp_path / "global_decoder.onnx"
    _write_decoder_fixture(model_path, 1, include_mask_reformat=True)

    with pytest.raises(ValueError, match="mask reformat"):
        validate_global_decoder(model_path, expected_layers=1)


def test_rewrite_attention_mask_for_gqa_removes_only_mask_bookkeeping(tmp_path) -> None:
    model_path = tmp_path / "global_decoder.onnx"
    _write_decoder_fixture(
        model_path,
        1,
        include_attention_mask=True,
        explicit_sequence_inputs=False,
        gqa_uses_mask_reformat=True,
    )

    rewrite_attention_mask_for_gqa(model_path)

    model = onnx.load_model(model_path, load_external_data=False)
    assert [item.name for item in model.graph.input if item.name == "attention_mask"] == []
    assert {item.name for item in model.graph.input} >= {"seqlens_k", "total_seq_len"}
    gqa = next(node for node in model.graph.node if node.op_type == "GroupQueryAttention")
    assert gqa.input[5:7] == ["seqlens_k", "total_seq_len"]
    assert not any("mask" in name.lower() for node in model.graph.node for name in (*node.input, *node.output))
    validate_global_decoder(model_path, expected_layers=1)


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
    release.mkdir(parents=True)
    (release / "previous.bin").write_bytes(b"previous release")
    receipt = paths.receipts / f"global-release-{layers}.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"generation":"previous"}', encoding="utf-8")
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
    assert payload["webgpu"]["requiredLimits"] == {
        "maxStorageBufferBindingSize": 134217728,
        "maxStorageBuffersPerShaderStage": 9,
    }
    assert payload["reducedHead"]["path"] == "reduced-head/reduced-head.onnx"
    assert (release / "embedding" / "embedding-000.fp16").is_file()
    assert (release / "reduced-head" / "reduced-head.onnx").is_file()
    assert (release / "tokenizer" / "tokenizer.json").is_file()
    assert (release / "LICENSE").is_file()
    assert receipt.is_file()
    archives = list((paths.root / "archive" / release_name).iterdir())
    assert len(archives) == 1
    assert (archives[0] / "release" / "previous.bin").read_bytes() == b"previous release"
    assert (archives[0] / "receipt.json").read_text(encoding="utf-8") == '{"generation":"previous"}'


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


def test_promote_directory_archives_prior_release_and_receipt(tmp_path) -> None:
    release = tmp_path / "release" / "global"
    staging = tmp_path / "release" / ".global-next.staging"
    receipt = tmp_path / "receipts" / "global-release-36.json"
    release.mkdir(parents=True)
    staging.mkdir(parents=True)
    receipt.parent.mkdir(parents=True)
    (release / "manifest.json").write_bytes(b"old manifest")
    (release / "weights.bin").write_bytes(b"old weights")
    (staging / "manifest.json").write_bytes(b"new manifest")
    receipt.write_text('{"generation":"old"}', encoding="utf-8")

    manifest._promote_directory(
        staging,
        release,
        receipt_path=receipt,
        receipt_payload={"generation": "new"},
    )

    archives = list((tmp_path / "archive" / "global").iterdir())
    assert len(archives) == 1
    assert (archives[0] / "release" / "manifest.json").read_bytes() == b"old manifest"
    assert (archives[0] / "release" / "weights.bin").read_bytes() == b"old weights"
    assert (archives[0] / "receipt.json").read_text(encoding="utf-8") == '{"generation":"old"}'
    assert (release / "manifest.json").read_bytes() == b"new manifest"
    assert json.loads(receipt.read_text(encoding="utf-8")) == {"generation": "new"}


def test_archive_file_copies_prior_release_without_linking_active_files(tmp_path) -> None:
    release = tmp_path / "release" / "global"
    release.mkdir(parents=True)
    previous = release / "manifest.json"
    previous.write_bytes(b"old")
    archived_copy = tmp_path / "archived-copy"

    manifest._archive_file(previous, archived_copy)

    assert archived_copy.read_bytes() == b"old"
    assert not previous.samefile(archived_copy)


def test_archive_file_rejects_an_unverified_copy(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "archive" / "source.bin"
    source.write_bytes(b"expected")

    def corrupt_copy(_source: Path, target: Path) -> None:
        target.write_bytes(b"corrupt")

    monkeypatch.setattr(manifest.shutil, "copy2", corrupt_copy)

    with pytest.raises(OSError, match="archive verification failed"):
        manifest._archive_file(source, destination)


def test_promote_directory_never_removes_prior_archive_generations(tmp_path) -> None:
    release = tmp_path / "release" / "global"
    release.mkdir(parents=True)
    (release / "manifest.json").write_bytes(b"first")

    for value in (b"second", b"third"):
        staging = tmp_path / "release" / f".global-{value.decode()}.staging"
        staging.mkdir()
        (staging / "manifest.json").write_bytes(value)
        manifest._promote_directory(staging, release)

    archives = list((tmp_path / "archive" / "global").iterdir())
    archived_manifests = {
        (archive / "release" / "manifest.json").read_bytes()
        for archive in archives
    }
    assert archived_manifests == {b"first", b"second"}


def test_promote_directory_rejects_paths_outside_artifact_release_root(tmp_path) -> None:
    release = tmp_path / "published" / "global"
    staging = tmp_path / "published" / ".global-next.staging"
    release.mkdir(parents=True)
    staging.mkdir(parents=True)
    (release / "manifest.json").write_bytes(b"old")
    (staging / "manifest.json").write_bytes(b"new")

    with pytest.raises(ValueError, match="artifact release root"):
        manifest._promote_directory(staging, release)

    assert (release / "manifest.json").read_bytes() == b"old"
    assert (staging / "manifest.json").read_bytes() == b"new"


def test_promote_directory_retains_backup_when_rollback_restore_fails(tmp_path, monkeypatch) -> None:
    release = tmp_path / "release" / "global"
    staging = tmp_path / "release" / ".global-next.staging"
    receipt = tmp_path / "receipts" / "global-release-36.json"
    release.mkdir(parents=True)
    staging.mkdir(parents=True)
    receipt.parent.mkdir(parents=True)
    (release / "manifest.json").write_bytes(b"old")
    (staging / "manifest.json").write_bytes(b"new")
    receipt.write_text('{"generation":"old"}', encoding="utf-8")
    original_replace = Path.replace

    def fail_backup_restore(source: Path, target: Path) -> Path:
        if source.name.endswith(".backup") and target == release:
            raise OSError("restore failed")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_backup_restore)
    monkeypatch.setattr(
        manifest,
        "_atomic_json",
        lambda *_: (_ for _ in ()).throw(OSError("receipt failed")),
    )

    with pytest.raises(OSError, match="restore failed"):
        manifest._promote_directory(
            staging,
            release,
            receipt_path=receipt,
            receipt_payload={"generation": "new"},
        )

    backups = list((tmp_path / "release").glob(".global-*.backup"))
    assert len(backups) == 1
    assert (backups[0] / "manifest.json").read_bytes() == b"old"
    assert receipt.read_text(encoding="utf-8") == '{"generation":"old"}'


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


def _write_decoder_fixture(path, layers: int, initializer_name: str = "qweight", include_attention_mask: bool = False,
                           explicit_sequence_inputs: bool = True, include_mask_reformat: bool = False,
                           gqa_uses_mask_reformat: bool = False) -> None:
    inputs = [helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT16, [1, None, 64])]
    if include_attention_mask:
        inputs.append(helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, None]))
    if explicit_sequence_inputs:
        inputs.extend([
            helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [2]),
            helper.make_tensor_value_info("total_seq_len", TensorProto.INT32, []),
        ])
    outputs = [helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT16, [1, None, 64])]
    nodes = []
    for index in range(layers):
        for cache in ("key", "value"):
            inputs.append(helper.make_tensor_value_info(f"past_key_values.{index}.{cache}", TensorProto.FLOAT16, [1, 2, None, 16]))
            outputs.append(helper.make_tensor_value_info(f"present.{index}.{cache}", TensorProto.FLOAT16, [1, 2, None, 16]))
        gqa_inputs = ["inputs_embeds", "", "", "", "", "mask_sequence", "mask_sum"] if gqa_uses_mask_reformat else (["inputs_embeds", "", "", "", "", "seqlens_k", "total_seq_len"] if explicit_sequence_inputs else ["inputs_embeds"])
        nodes.append(
            helper.make_node(
                "GroupQueryAttention",
                gqa_inputs,
                [f"layer_{index}", f"present.{index}.key", f"present.{index}.value"],
                domain="com.microsoft",
            )
        )
    if include_attention_mask or include_mask_reformat:
        mask_source = "attention_mask" if include_attention_mask else "inputs_embeds"
        nodes.extend([
            helper.make_node("Shape", [mask_source], ["mask_shape"]),
            helper.make_node("Gather", ["mask_shape", "qweight"], ["mask_length"]),
            helper.make_node("Cast", ["mask_length"], ["mask_length_i32"], to=TensorProto.INT32),
            helper.make_node("ReduceSum", ["mask_length_i32"], ["mask_sum"], keepdims=0),
            helper.make_node("Sub", ["mask_sum", "mask_length_i32"], ["mask_sequence"]),
        ])
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
        helper.make_tensor_value_info("last_state", TensorProto.FLOAT16, [2, 4096]),
    ]))
    onnx.save_model(model, path)
