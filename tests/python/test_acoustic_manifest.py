import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import onnx
from onnx import TensorProto, helper

from minimax_music3_webgpu.acoustic_manifest import (
    BuiltAcousticGraphs,
    build_music_5s_release,
    build_music_variable_release,
)
from minimax_music3_webgpu.cli import main
from minimax_music3_webgpu.constants import (
    ACOUSTIC_SOURCE_FILES,
    DIFFUSERS_REVISION,
    MODEL_ID,
    MODEL_REVISION,
)
from minimax_music3_webgpu.paths import ArtifactPaths


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file(root: Path, relative: str, data: bytes = b"fixture") -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": relative, "bytes": len(data), "sha256": _sha(path)}


def _graph(root: Path, relative: str, output: str) -> dict:
    external_relative = f"{Path(relative).parent.as_posix()}/weights.bin"
    external = _file(root, external_relative, b"\x00\x00\x00\x00")
    external["onnxLocation"] = "weights.bin"
    tensor = onnx.TensorProto()
    tensor.name = output
    tensor.data_type = TensorProto.FLOAT
    tensor.dims.append(1)
    tensor.data_location = TensorProto.EXTERNAL
    for key, value in (("location", "weights.bin"), ("offset", "0"), ("length", "4")):
        field = tensor.external_data.add()
        field.key = key
        field.value = value
    model = helper.make_model(
        helper.make_graph(
            [],
            Path(relative).stem,
            [],
            [helper.make_tensor_value_info(output, TensorProto.FLOAT, [1])],
            initializer=[tensor],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    graph_path = root / relative
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, graph_path)
    graph = {"path": relative, "bytes": graph_path.stat().st_size, "sha256": _sha(graph_path)}
    return {**graph, "externalData": [external], "gpuOutputs": [output]}


def _base() -> dict:
    return {
        "schemaVersion": 1,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "diffusersRevision": DIFFUSERS_REVISION,
        },
        "webgpu": {
            "requiredFeatures": ["shader-f16"],
            "requiredLimits": {"maxStorageBufferBindingSize": 134217728},
        },
    }


def _standalone_releases(paths: ArtifactPaths) -> None:
    global_root = paths.release / "global"
    global_root.mkdir(parents=True)
    embedding = _file(global_root, "embedding/embedding.fp16", b"1234")
    embedding.update({"rowStart": 0, "rowCount": 1})
    tokenizer = _file(global_root, "tokenizer/tokenizer.json", b"{}")
    license_file = _file(global_root, "LICENSE", b"license")
    (global_root / "manifest.json").write_text(json.dumps({
        **_base(),
        "quantization": {"bits": 4, "blockSize": 128, "accuracyLevel": 4, "symmetric": True},
        "graph": _graph(global_root, "global.onnx", "hidden_states"),
        "reducedHead": _graph(global_root, "head/head.onnx", "last_state"),
        "embedding": {"rows": 1, "columns": 2, "rowBytes": 4, "shards": [embedding]},
        "tokenizerFiles": [tokenizer],
        "licenseFile": license_file,
        "kvPairs": [{"pastInput": "past.0", "presentOutput": "present.0"}],
    }))

    rvq_root = paths.release / "rvq"
    rvq_root.mkdir()
    rvq_embedding = _file(rvq_root, "embedding/embedding.fp16", b"1234")
    rvq_embedding.update({"rowStart": 0, "rowCount": 1})
    (rvq_root / "manifest.json").write_text(json.dumps({
        **_base(),
        "rvqDepth": _graph(rvq_root, "rvq/rvq.onnx", "depth_hidden"),
        "feedback": {**_file(rvq_root, "feedback.onnx", b"onnx"), "externalData": [], "gpuOutputs": ["inputs_embeds"]},
        "rvqEmbedding": {"rows": 1, "columns": 2, "rowBytes": 4, "shards": [rvq_embedding]},
    }))

    condition_root = paths.release / "condition"
    condition_root.mkdir()
    (condition_root / "manifest.json").write_text(json.dumps({
        **_base(),
        "conditionEncoder": _graph(condition_root, "condition/condition.onnx", "condition"),
    }))

    flow_root = paths.release / "flow"
    flow_root.mkdir()
    (flow_root / "manifest.json").write_text(json.dumps({
        **_base(),
        "quantization": {"bits": 4, "blockSize": 128, "accuracyLevel": 4, "symmetric": True},
        "slice": {"semanticFrames": 125, "latentLength": 430, "flowSteps": 30, "flowGuidance": 1.7},
        "flow": _graph(flow_root, "flow/flow.onnx", "next_latents"),
    }))

    vocoder_root = paths.release / "vocoder"
    vocoder_root.mkdir()
    (vocoder_root / "manifest.json").write_text(json.dumps({
        **_base(),
        "slice": {"latentChannels": 128, "latentLength": 430, "outputSamples": 220160, "sampleRate": 44100, "channels": 2},
        "precision": {"convolution": "float16", "fp32Snakes": ["blocks.0.snake1", "blocks.1.snake1"]},
        "vocoder": _graph(vocoder_root, "vocoder/vocoder.onnx", "waveform"),
    }))


def _acoustic_source_receipt(paths: ArtifactPaths) -> None:
    paths.receipts.mkdir(parents=True, exist_ok=True)
    files = []
    for relative in ACOUSTIC_SOURCE_FILES:
        source = paths.source / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        data = relative.encode("utf-8")
        source.write_bytes(data)
        files.append({
            "path": relative,
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    (paths.receipts / "source-acoustic.json").write_text(json.dumps({
        "repository_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": files,
    }), encoding="utf-8")


def _variable_graphs(root: Path, corrupt_range: str | None = None) -> BuiltAcousticGraphs:
    def graph(name: str, inputs: list, output, initializers=()) -> Path:
        path = root / f"{name}.onnx"
        path.parent.mkdir(parents=True, exist_ok=True)
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Identity", [inputs[0].name], [output.name])],
                name,
                inputs,
                [output],
                initializer=list(initializers),
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        onnx.save_model(model, path)
        return path

    def external_tensor(name: str, location: str, offset: int = 0, length: int = 4):
        tensor = onnx.TensorProto()
        tensor.name = name
        tensor.data_type = TensorProto.FLOAT
        tensor.dims.append(1)
        tensor.data_location = TensorProto.EXTERNAL
        for key, value in (
            ("location", location),
            ("offset", str(offset)),
            ("length", str(length)),
        ):
            field = tensor.external_data.add()
            field.key = key
            field.value = value
        return tensor

    weights = root / "condition.weights.bin"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"\x00" * 8)
    condition_initializers = []
    for index, offset in enumerate((0, 2 if corrupt_range == "overlap" else 4)):
        condition_initializers.append(external_tensor(
            f"condition_weight_{index}",
            weights.name,
            offset,
            3 if corrupt_range == "length" and index == 0 else 4,
        ))

    condition = graph(
        "condition",
        [
            helper.make_tensor_value_info("frame_hiddens", TensorProto.FLOAT16, [1, 200, 32768]),
            helper.make_tensor_value_info("nearest_index", TensorProto.INT64, [689]),
            helper.make_tensor_value_info("active_latent_mask", TensorProto.FLOAT16, [1, 689, 1]),
        ],
        helper.make_tensor_value_info("condition", TensorProto.FLOAT16, [1, 689, 2048]),
        condition_initializers,
    )
    flow_weights = root / "flow.weights.bin"
    flow_weights.write_bytes(b"\x01" * 4)
    flow = graph(
        "flow",
        [
            helper.make_tensor_value_info("latents", TensorProto.FLOAT16, [1, 128, 689]),
            helper.make_tensor_value_info("condition", TensorProto.FLOAT16, [1, 689, 2048]),
            helper.make_tensor_value_info("timestep", TensorProto.FLOAT16, [1]),
            helper.make_tensor_value_info("dt", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("active_latent_mask", TensorProto.FLOAT16, [1, 689, 1]),
            helper.make_tensor_value_info("key_attention_bias", TensorProto.FLOAT16, [1, 1, 1, 690]),
            helper.make_tensor_value_info("noise_prompt", TensorProto.FLOAT16, [1, 128, 172]),
            helper.make_tensor_value_info("previous_latent", TensorProto.FLOAT16, [1, 128, 172]),
            helper.make_tensor_value_info("overlap_enabled", TensorProto.FLOAT16, [1]),
            helper.make_tensor_value_info("guidance", TensorProto.FLOAT16, [1]),
        ],
        helper.make_tensor_value_info("next_latents", TensorProto.FLOAT16, [1, 128, 689]),
        [external_tensor("flow_weight", flow_weights.name)],
    )
    vocoder_weights = root / "vocoder.weights.bin"
    vocoder_weights.write_bytes(b"\x02" * 4)
    vocoder = graph(
        "vocoder",
        [helper.make_tensor_value_info("latents", TensorProto.FLOAT16, [1, 64, "latent_length"])],
        helper.make_tensor_value_info("waveform", TensorProto.FLOAT, [1, 1, "sample_length"]),
        [external_tensor("vocoder_weight", vocoder_weights.name)],
    )
    return BuiltAcousticGraphs(
        condition_encoder=condition,
        flow=flow,
        vocoder=vocoder,
        fp32_snakes=("blocks.0.snake1", "blocks.1.snake1"),
    )


def _build_variable(paths: ArtifactPaths, *, corrupt_range: str | None = None) -> Path:
    return build_music_variable_release(
        paths,
        _build_graphs=lambda staging: _variable_graphs(staging / "acoustic", corrupt_range),
        _preflight=lambda _: None,
        _validate_components=lambda _: None,
    )


def test_build_music_variable_publishes_the_official_maximum_acoustic_contract(
    tmp_path: Path,
) -> None:
    paths = ArtifactPaths.from_root(tmp_path, repository_root=tmp_path)
    _standalone_releases(paths)
    existing = paths.release / "music-5s"
    existing.mkdir()
    sentinel = existing / "sentinel.bin"
    sentinel.write_bytes(b"unchanged-five-second-release")
    before = sentinel.read_bytes()
    manifest_path = _build_variable(paths)
    manifest = json.loads(manifest_path.read_text())

    assert manifest["acoustic"] == {
        "maxSemanticFrames": 200,
        "windowFrames": 200,
        "hopFrames": 100,
        "overlapLatents": 172,
        "leftCrop": 86,
        "rightCrop": 258,
        "samplesPerLatent": 512,
        "maxLatentLength": 689,
        "flowSteps": 30,
        "flowGuidance": 1.7,
    }
    assert manifest["conditionEncoder"]["inputs"][-2:] == [
        {"name": "nearest_index", "dtype": "int64", "shape": [689]},
        {"name": "active_latent_mask", "dtype": "float16", "shape": [1, 689, 1]},
    ]
    assert len(manifest["flow"]["inputs"]) == 10
    assert manifest["flow"]["inputs"][-1] == {
        "name": "guidance",
        "dtype": "float16",
        "shape": [1],
    }
    assert manifest["flow"]["gpuOutputs"] == ["next_latents"]
    assert manifest["vocoder"]["inputs"] == [
        {"name": "latents", "dtype": "float16", "shape": [1, 64, "L"], "maxShape": [1, 64, 689]}
    ]
    assert manifest["vocoder"]["outputs"] == [
        {"name": "waveform", "dtype": "float32", "shape": [1, 1, "512L"], "maxShape": [1, 1, 352768]}
    ]
    assert manifest["webgpu"] == {
        "requiredFeatures": ["shader-f16"],
        "requiredLimits": {
            "maxStorageBufferBindingSize": 134217728,
            "maxStorageBuffersPerShaderStage": 9,
        },
    }
    assert sentinel.read_bytes() == before
    assert (paths.receipts / "music-variable.json").is_file()
    assert not any((paths.release / name).exists() for name in (
        "condition-variable", "flow-variable", "vocoder-variable"
    ))
    assert os.path.samefile(
        paths.release / "global" / "global.onnx",
        manifest_path.parent / manifest["graph"]["path"],
    )
    assert os.path.samefile(
        paths.release / "rvq" / "rvq" / "rvq.onnx",
        manifest_path.parent / manifest["rvqDepth"]["path"],
    )


def test_music_5s_replacement_archives_the_prior_release(tmp_path: Path) -> None:
    paths = ArtifactPaths.from_root(tmp_path, repository_root=tmp_path)
    _standalone_releases(paths)
    first = build_music_5s_release(paths)
    before = _release_bytes(first.parent)

    build_music_5s_release(paths)

    archives = [
        path
        for path in (paths.root / "archive" / "music-5s").iterdir()
        if path.is_dir()
    ]
    assert len(archives) == 1
    assert _release_bytes(archives[0] / "release") == before


def test_music_5s_restore_failure_preserves_backup_and_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from minimax_music3_webgpu import acoustic_manifest as acoustic_module

    release = tmp_path / "music-5s"
    staging = tmp_path / ".music-5s-next.staging"
    release.mkdir()
    staging.mkdir()
    (release / "old.bin").write_bytes(b"old release")
    (staging / "new.bin").write_bytes(b"new release")
    replace = Path.replace

    def fail_publish_and_restore(source: Path, target: Path) -> Path:
        if source == staging:
            raise OSError("primary publication failure")
        if source.name.startswith(".music-5s-") and source.name.endswith(".backup"):
            raise OSError("rollback restore failure")
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_publish_and_restore)

    with pytest.raises(OSError, match="primary publication failure"):
        acoustic_module._promote(staging, release)

    backups = list(tmp_path.glob(".music-5s-*.backup"))
    assert len(backups) == 1
    assert _release_bytes(backups[0]) == {"old.bin": b"old release"}


def _release_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_successful_replacement_archives_prior_release_and_reuses_unchanged_acoustic_files(
    tmp_path: Path,
) -> None:
    paths = ArtifactPaths.from_root(tmp_path, repository_root=tmp_path)
    _standalone_releases(paths)
    first_manifest_path = _build_variable(paths)
    prior_release = _release_bytes(first_manifest_path.parent)
    prior_receipt = (paths.receipts / "music-variable.json").read_bytes()

    second_manifest_path = _build_variable(paths)
    second_manifest = json.loads(second_manifest_path.read_text())
    archives = [path for path in (paths.root / "archive" / "music-variable").iterdir() if path.is_dir()]

    assert len(archives) == 1
    archive = archives[0]
    assert _release_bytes(archive / "release") == prior_release
    assert (archive / "receipt.json").read_bytes() == prior_receipt
    archived_manifest = json.loads((archive / "release" / "manifest.json").read_text())
    for key in ("conditionEncoder", "vocoder"):
        assert os.path.samefile(
            second_manifest_path.parent / second_manifest[key]["path"],
            archive / "release" / archived_manifest[key]["path"],
        )
    assert not os.path.samefile(
        second_manifest_path.parent / second_manifest["flow"]["path"],
        archive / "release" / archived_manifest["flow"]["path"],
    )
    assert os.path.samefile(
        second_manifest_path.parent / second_manifest["flow"]["externalData"][0]["path"],
        archive / "release" / archived_manifest["flow"]["externalData"][0]["path"],
    )
    archived_bytes = _release_bytes(archive / "release")

    _build_variable(paths)

    retained = [path for path in (paths.root / "archive" / "music-variable").iterdir() if path.is_dir()]
    assert len(retained) == 2
    assert archive in retained
    assert _release_bytes(archive / "release") == archived_bytes


def test_pinned_builder_skips_all_matching_acoustic_exports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from minimax_music3_webgpu import acoustic_manifest as acoustic_module

    paths = ArtifactPaths.from_root(tmp_path, repository_root=tmp_path)
    _standalone_releases(paths)
    prior_manifest_path = _build_variable(paths)
    prior = json.loads(prior_manifest_path.read_text())
    monkeypatch.setattr(
        acoustic_module,
        "export_condition_encoder",
        lambda *_args, **_kwargs: pytest.fail("condition exporter must be skipped"),
    )
    monkeypatch.setattr(
        acoustic_module,
        "load_vocoder_state_dict",
        lambda *_args, **_kwargs: pytest.fail("vocoder loader must be skipped"),
    )
    monkeypatch.setattr(
        acoustic_module,
        "open_flow_state",
        lambda *_args, **_kwargs: pytest.fail("flow loader must be skipped"),
    )

    next_manifest_path = build_music_variable_release(
        paths,
        _preflight=lambda _: None,
        _validate_components=lambda _: None,
    )
    built_manifest = json.loads(next_manifest_path.read_text())
    archives = list((paths.root / "archive" / "music-variable").iterdir())

    for name in ("conditionEncoder", "flow", "vocoder"):
        current = next_manifest_path.parent / built_manifest[name]["path"]
        archived = archives[0] / "release" / prior[name]["path"]
        assert current.read_bytes() == archived.read_bytes()
        assert not os.path.samefile(current, archived)


def test_pinned_builder_rebuilds_when_component_fingerprint_is_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from minimax_music3_webgpu import acoustic_manifest as acoustic_module

    paths = ArtifactPaths.from_root(tmp_path, repository_root=tmp_path)
    _standalone_releases(paths)
    prior_manifest_path = _build_variable(paths)
    prior = json.loads(prior_manifest_path.read_text())
    prior["conditionEncoder"]["buildFingerprint"] = "stale"
    monkeypatch.setattr(
        acoustic_module,
        "export_condition_encoder",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("condition rebuild")),
    )

    with pytest.raises(RuntimeError, match="condition rebuild"):
        acoustic_module._build_pinned_acoustic_graphs(
            paths.release / ".next.staging",
            prior_manifest_path.parent,
            prior,
        )


def test_pinned_builder_rejects_corruption_while_copying_prior_acoustic_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from minimax_music3_webgpu import acoustic_manifest as acoustic_module

    paths = ArtifactPaths.from_root(tmp_path, repository_root=tmp_path)
    _standalone_releases(paths)
    _build_variable(paths)
    before = _release_bytes(paths.release / "music-variable")
    receipt = paths.receipts / "music-variable.json"
    receipt_before = receipt.read_bytes()
    copy2 = acoustic_module.shutil.copy2

    def corrupt_copy(source: Path, destination: Path) -> Path:
        result = copy2(source, destination)
        if destination.name == "condition.weights.bin":
            data = destination.read_bytes()
            destination.write_bytes(bytes([data[0] ^ 0xFF]) + data[1:])
        return result

    monkeypatch.setattr(acoustic_module.shutil, "copy2", corrupt_copy)

    with pytest.raises(ValueError, match="copied prior acoustic artifact integrity"):
        build_music_variable_release(
            paths,
            _preflight=lambda _: None,
            _validate_components=lambda _: None,
        )

    assert _release_bytes(paths.release / "music-variable") == before
    assert receipt.read_bytes() == receipt_before


def test_restore_failure_preserves_backup_and_primary_receipt_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from minimax_music3_webgpu import acoustic_manifest as acoustic_module

    paths = ArtifactPaths.from_root(tmp_path, repository_root=tmp_path)
    _standalone_releases(paths)
    _build_variable(paths)
    before = _release_bytes(paths.release / "music-variable")
    receipt = paths.receipts / "music-variable.json"
    receipt_before = receipt.read_bytes()
    atomic_json = acoustic_module._atomic_json

    def fail_receipt(path: Path, payload: dict) -> None:
        if path == receipt:
            raise OSError("primary receipt failure")
        atomic_json(path, payload)

    replace = Path.replace

    def fail_backup_restore(source: Path, target: Path) -> Path:
        if source.name.startswith(".music-variable-") and source.name.endswith(".backup"):
            raise OSError("rollback restore failure")
        return replace(source, target)

    restored_receipts: list[Path] = []
    restore_receipt = acoustic_module._restore_archived_receipt

    def record_receipt_restore(source: Path, target: Path) -> None:
        restored_receipts.append(source)
        restore_receipt(source, target)

    monkeypatch.setattr(acoustic_module, "_atomic_json", fail_receipt)
    monkeypatch.setattr(Path, "replace", fail_backup_restore)
    monkeypatch.setattr(acoustic_module, "_restore_archived_receipt", record_receipt_restore)

    with pytest.raises(OSError, match="primary receipt failure"):
        _build_variable(paths)

    backups = list(paths.release.glob(".music-variable-*.backup"))
    archives = [
        path
        for path in (paths.root / "archive" / "music-variable").iterdir()
        if path.is_dir()
    ]
    assert len(backups) == 1
    assert _release_bytes(backups[0]) == before
    assert len(archives) == 1
    assert _release_bytes(archives[0] / "release") == before
    assert (archives[0] / "receipt.json").read_bytes() == receipt_before
    assert receipt.read_bytes() == receipt_before
    assert restored_receipts == [archives[0] / "receipt.json"]


def test_build_music_variable_late_validation_failure_preserves_previous_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = ArtifactPaths.from_root(tmp_path, repository_root=tmp_path)
    _standalone_releases(paths)
    _build_variable(paths)
    before = _release_bytes(paths.release / "music-variable")
    receipt_before = (paths.receipts / "music-variable.json").read_bytes()
    monkeypatch.setattr(
        "minimax_music3_webgpu.acoustic_manifest._validate_assembled",
        lambda *_: (_ for _ in ()).throw(ValueError("injected late validation failure")),
    )

    with pytest.raises(ValueError, match="injected late validation failure"):
        _build_variable(paths)

    assert _release_bytes(paths.release / "music-variable") == before
    assert (paths.receipts / "music-variable.json").read_bytes() == receipt_before
    assert not list(paths.release.glob(".music-variable-*.staging"))


def test_build_music_variable_promotion_failure_rolls_back_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = ArtifactPaths.from_root(tmp_path, repository_root=tmp_path)
    _standalone_releases(paths)
    _build_variable(paths)
    before = _release_bytes(paths.release / "music-variable")
    receipt_before = (paths.receipts / "music-variable.json").read_bytes()
    replace = Path.replace

    def fail_new_generation(source: Path, target: Path) -> Path:
        if source.name.startswith(".music-variable-") and source.name.endswith(".staging"):
            raise OSError("injected promotion failure")
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_new_generation)

    with pytest.raises(OSError, match="injected promotion failure"):
        _build_variable(paths)

    assert _release_bytes(paths.release / "music-variable") == before
    assert (paths.receipts / "music-variable.json").read_bytes() == receipt_before
    assert not list(paths.release.glob(".music-variable-*.staging"))


def test_build_music_variable_receipt_failure_rolls_back_the_promoted_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = ArtifactPaths.from_root(tmp_path, repository_root=tmp_path)
    _standalone_releases(paths)
    _build_variable(paths)
    before = _release_bytes(paths.release / "music-variable")
    receipt = paths.receipts / "music-variable.json"
    receipt_before = receipt.read_bytes()
    from minimax_music3_webgpu import acoustic_manifest as acoustic_module

    atomic_json = acoustic_module._atomic_json

    def fail_receipt(path: Path, payload: dict) -> None:
        if path == receipt:
            raise OSError("injected receipt failure")
        atomic_json(path, payload)

    monkeypatch.setattr(acoustic_module, "_atomic_json", fail_receipt)

    with pytest.raises(OSError, match="injected receipt failure"):
        _build_variable(paths)

    assert _release_bytes(paths.release / "music-variable") == before
    assert receipt.read_bytes() == receipt_before
    assert not list(paths.release.glob(".music-variable-*.staging"))
    archives = [path for path in (paths.root / "archive" / "music-variable").iterdir() if path.is_dir()]
    assert len(archives) == 1
    assert _release_bytes(archives[0] / "release") == before
    assert (archives[0] / "receipt.json").read_bytes() == receipt_before


@pytest.mark.parametrize("corruption", ["length", "overlap"])
def test_build_music_variable_rejects_invalid_external_initializer_ranges(
    tmp_path: Path,
    corruption: str,
) -> None:
    paths = ArtifactPaths.from_root(tmp_path, repository_root=tmp_path)
    _standalone_releases(paths)
    _build_variable(paths)
    before = _release_bytes(paths.release / "music-variable")
    receipt_before = (paths.receipts / "music-variable.json").read_bytes()

    with pytest.raises(ValueError, match="external initializer"):
        _build_variable(paths, corrupt_range=corruption)

    assert _release_bytes(paths.release / "music-variable") == before
    assert (paths.receipts / "music-variable.json").read_bytes() == receipt_before
    assert not list(paths.release.glob(".music-variable-*.staging"))


def test_build_music_variable_cli_runs_the_pinned_source_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[ArtifactPaths] = []
    monkeypatch.setattr(
        "minimax_music3_webgpu.cli.build_music_variable_release",
        lambda paths: received.append(paths),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "minimax-music3-webgpu",
            "build-music-variable",
            "--artifacts-dir",
            "artifacts",
        ],
    )

    main()

    assert received == [ArtifactPaths.from_root(Path("artifacts"))]


def test_music_variable_preflight_verifies_pinned_sources_disk_and_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from minimax_music3_webgpu import acoustic_manifest as acoustic_module

    paths = ArtifactPaths.from_root(tmp_path, repository_root=tmp_path)
    _standalone_releases(paths)
    _acoustic_source_receipt(paths)
    checked: list[str] = []
    monkeypatch.setattr(
        acoustic_module.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=100 * 1024**3),
    )
    monkeypatch.setattr(
        acoustic_module,
        "_reject_conflicting_processes",
        lambda *_, **__: checked.append("processes"),
    )

    acoustic_module._preflight_large_build(paths)

    assert checked == ["processes"]

    profile = paths.root / "browser-profiles" / "variable-duration" / "local"
    profile.mkdir(parents=True)
    (profile / "SingletonLock").write_text("locked")
    with pytest.raises(RuntimeError, match="target profile is locked"):
        acoustic_module._preflight_large_build(paths)


def test_process_preflight_scopes_chrome_and_rejects_other_converter(tmp_path: Path) -> None:
    from minimax_music3_webgpu import acoustic_manifest as acoustic_module

    own_tree = [
        (100, 0, "powershell.exe", "codex wrapper music3-convert build-music-variable"),
        (200, 100, "uv.exe", "uv run minimax_music3_webgpu.cli build-music-variable"),
        (300, 200, "python.exe", "python minimax_music3_webgpu.cli build-music-variable"),
    ]
    repository = tmp_path / "repo"
    profile = repository / "artifacts" / "browser-profiles" / "variable-duration" / "local"

    acoustic_module._reject_conflicting_processes(
        [*own_tree, (500, 1, "chrome.exe", "chrome.exe --user-data-dir=C:/unrelated/profile")],
        current_pid=300,
        repository_root=repository,
        target_profiles=(profile,),
    )

    with pytest.raises(RuntimeError, match="another converter process"):
        acoustic_module._reject_conflicting_processes(
            [*own_tree, (400, 1, "python.exe", "music3-convert build-music-variable")],
            current_pid=300,
            repository_root=repository,
            target_profiles=(profile,),
        )
    with pytest.raises(RuntimeError, match="Chrome is using the converter target"):
        acoustic_module._reject_conflicting_processes(
            [*own_tree, (501, 1, "chrome.exe", f"chrome.exe --app={repository / 'index.html'}")],
            current_pid=300,
            repository_root=repository,
            target_profiles=(profile,),
        )
    with pytest.raises(RuntimeError, match="Chrome is using the converter target"):
        acoustic_module._reject_conflicting_processes(
            [*own_tree, (502, 1, "chrome.exe", f"chrome.exe --user-data-dir={profile}")],
            current_pid=300,
            repository_root=repository,
            target_profiles=(profile,),
        )


def test_build_music_5s_copies_and_rehashes_every_exact_release_artifact(tmp_path: Path) -> None:
    paths = ArtifactPaths.from_root(tmp_path, repository_root=tmp_path)
    _standalone_releases(paths)

    manifest_path = build_music_5s_release(paths)
    manifest = json.loads(manifest_path.read_text())

    assert manifest["model"] == _base()["model"]
    assert manifest["slice"] == {
        "semanticFrames": 125,
        "latentLength": 430,
        "outputSamples": 220160,
        "sampleRate": 44100,
        "channels": 2,
        "flowSteps": 30,
        "globalGuidance": 1.5,
        "flowGuidance": 1.7,
    }
    assert manifest["graph"]["path"] == "global/global.onnx"
    assert manifest["rvqDepth"]["path"] == "rvq/rvq/rvq.onnx"
    assert manifest["conditionEncoder"]["path"] == "condition/condition/condition.onnx"
    assert manifest["flow"]["path"] == "flow/flow/flow.onnx"
    assert manifest["vocoder"]["path"] == "vocoder/vocoder/vocoder.onnx"
    assert manifest["tokenizerFiles"][0]["path"] == "global/tokenizer/tokenizer.json"
    assert manifest["licenseFile"]["path"] == "global/LICENSE"
    assert manifest["flow"]["externalData"][0]["onnxLocation"] == "weights.bin"

    artifacts = [
        manifest["graph"], manifest["reducedHead"], manifest["rvqDepth"],
        manifest["feedback"], manifest["conditionEncoder"], manifest["flow"], manifest["vocoder"],
        *manifest["graph"]["externalData"], *manifest["reducedHead"]["externalData"],
        *manifest["rvqDepth"]["externalData"], *manifest["feedback"]["externalData"],
        *manifest["conditionEncoder"]["externalData"], *manifest["flow"]["externalData"],
        *manifest["vocoder"]["externalData"], *manifest["embedding"]["shards"],
        *manifest["rvqEmbedding"]["shards"], *manifest["tokenizerFiles"], manifest["licenseFile"],
    ]
    for artifact in artifacts:
        path = manifest_path.parent / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert _sha(path) == artifact["sha256"]
        assert artifact["bytes"] <= 134217728


def test_build_music_5s_rejects_tampered_standalone_artifact_without_replacing_release(tmp_path: Path) -> None:
    paths = ArtifactPaths.from_root(tmp_path, repository_root=tmp_path)
    _standalone_releases(paths)
    existing = paths.release / "music-5s"
    existing.mkdir(parents=True)
    (existing / "sentinel.txt").write_text("keep")
    (paths.release / "flow" / "flow" / "weights.bin").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="integrity"):
        build_music_5s_release(paths)

    assert (existing / "sentinel.txt").read_text() == "keep"


def test_build_music_5s_cli_uses_requested_artifacts_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[ArtifactPaths] = []
    monkeypatch.setattr(
        "minimax_music3_webgpu.cli.build_music_5s_release",
        lambda paths: received.append(paths),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["minimax-music3-webgpu", "build-music-5s", "--artifacts-dir", "artifacts"],
    )

    main()

    assert received == [ArtifactPaths.from_root(Path("artifacts"))]
