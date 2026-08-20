from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from safetensors.torch import save_file
import torch

from minimax_music3_webgpu.manifest import emit_vocoder_release
from minimax_music3_webgpu.paths import ArtifactPaths
from minimax_music3_webgpu.vocoder import (
    EXACT_VOCODER_CONFIG,
    MiniMaxMusic3VocoderExport,
    VocoderConfig,
    build_vocoder_reference_module,
    expected_source_shapes,
    export_vocoder_module,
    fold_vocoder_state_dict,
    fold_weight_norm,
    graph_expectations,
    load_vocoder_state_dict,
    prepare_vocoder_state_dict,
    publish_vocoder_module,
    run_vocoder_cpu_oracle,
    select_fp32_snakes,
    validate_source_state_dict,
    validate_vocoder_graph,
)


def test_exact_source_metadata_matches_pinned_vocoder() -> None:
    shapes = expected_source_shapes(EXACT_VOCODER_CONFIG)

    assert len(shapes) == 121
    assert shapes["dec_in_proj.weight"] == (1024, 64, 1)
    assert shapes["conv_in.weight_v"] == (1536, 1024, 7)
    assert shapes["blocks.0.conv_t1.weight_g"] == (1536, 1, 1)
    assert shapes["blocks.0.conv_t1.weight_v"] == (1536, 768, 16)
    assert shapes["blocks.3.conv_t1.weight_v"] == (192, 96, 4)
    assert shapes["blocks.0.res_unit3.conv1.weight_v"] == (768, 768, 7)
    assert shapes["conv_out.weight_v"] == (1, 96, 7)


def test_source_validation_rejects_wrong_shape_and_dtype() -> None:
    state = {
        name: torch.empty(shape, dtype=torch.float32, device="meta")
        for name, shape in expected_source_shapes(EXACT_VOCODER_CONFIG).items()
    }
    state["conv_out.weight_v"] = torch.empty((1, 95, 7), dtype=torch.float32, device="meta")

    with pytest.raises(ValueError, match="conv_out.weight_v shape"):
        validate_source_state_dict(state, EXACT_VOCODER_CONFIG)

    state["conv_out.weight_v"] = torch.empty((1, 96, 7), dtype=torch.float16, device="meta")
    with pytest.raises(ValueError, match="conv_out.weight_v dtype"):
        validate_source_state_dict(state, EXACT_VOCODER_CONFIG)


def test_loader_reads_and_validates_a_complete_safetensors_state(tmp_path: Path) -> None:
    config = _tiny_config()
    expected = _source_state(config)
    path = tmp_path / "vocoder.safetensors"
    save_file(expected, path)

    actual = load_vocoder_state_dict(path, config)

    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])


def test_weight_norm_folds_dimension_zero_for_conv_and_conv_transpose() -> None:
    conv_v = torch.tensor(
        [
            [[3.0, 4.0], [0.0, 0.0]],
            [[0.0, 0.0], [5.0, 12.0]],
        ]
    )
    conv_g = torch.tensor([[[10.0]], [[26.0]]])

    folded = fold_weight_norm(conv_g, conv_v)

    torch.testing.assert_close(folded[0], conv_v[0] * 2.0)
    torch.testing.assert_close(folded[1], conv_v[1] * 2.0)

    conv_transpose_v = torch.tensor(
        [
            [[3.0, 4.0], [0.0, 0.0]],
            [[0.0, 0.0], [5.0, 12.0]],
        ]
    )
    conv_transpose_g = torch.tensor([[[5.0]], [[39.0]]])
    folded_transpose = fold_weight_norm(conv_transpose_g, conv_transpose_v)

    torch.testing.assert_close(folded_transpose[0], conv_transpose_v[0])
    torch.testing.assert_close(folded_transpose[1], conv_transpose_v[1] * 3.0)


def test_fp32_snake_selection_uses_float16_normal_threshold() -> None:
    state = {
        "blocks.0.snake1.alpha": torch.tensor([[[4.748142e-6, 1.0]]]),
        "blocks.1.snake1.alpha": torch.tensor([[[2.455492e-5, 1.0]]]),
        "blocks.2.snake1.alpha": torch.tensor([[[torch.finfo(torch.float16).tiny, 1.0]]]),
    }

    assert select_fp32_snakes(state) == ("blocks.0.snake1", "blocks.1.snake1")


def test_prepare_state_folds_in_float32_then_uses_mixed_precision() -> None:
    config = _tiny_config()
    source = _source_state(config)
    source["blocks.0.snake1.alpha"].flatten()[0] = 4.748142e-6

    prepared, fp32_snakes = prepare_vocoder_state_dict(source, config)

    assert fp32_snakes == ("blocks.0.snake1",)
    assert "conv_in.weight_g" not in prepared
    assert "conv_in.weight_v" not in prepared
    assert prepared["conv_in.weight"].dtype == torch.float16
    assert prepared["blocks.0.snake1.alpha"].dtype == torch.float32
    assert prepared["blocks.1.snake1.alpha"].dtype == torch.float16


def test_folded_state_matches_weight_norm_reference_on_tiny_model() -> None:
    config = _tiny_config()
    source = _source_state(config)
    reference = build_vocoder_reference_module(source, config)
    folded = fold_vocoder_state_dict(source, config)
    exported = MiniMaxMusic3VocoderExport.from_prepared_state(config, folded, ())
    values = torch.arange(config.latent_channels * config.latent_length, dtype=torch.float32)
    latents = (((values % 257) - 128) / 1024).reshape(1, config.latent_channels, config.latent_length)

    with torch.no_grad():
        expected = reference(latents)
        actual = exported(latents)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_fixed_export_has_exact_operator_and_shape_contract(tmp_path: Path) -> None:
    config = _tiny_config()
    source = _source_state(config)
    source["blocks.0.snake1.alpha"].flatten()[0] = 4.748142e-6
    prepared, fp32_snakes = prepare_vocoder_state_dict(source, config)
    module = MiniMaxMusic3VocoderExport.from_prepared_state(config, prepared, fp32_snakes)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        graph_path = export_vocoder_module(module, tmp_path / "vocoder.onnx")

    report = validate_vocoder_graph(graph_path, config)
    model = onnx.load_model(graph_path, load_external_data=False)
    counts = Counter(node.op_type for node in model.graph.node)

    assert report.input_shape == (1, config.latent_channels, config.latent_length)
    assert report.output_shape == (1, 2, config.latent_length * 16)
    assert report.input_dtype == onnx.TensorProto.FLOAT16
    assert report.output_dtype == onnx.TensorProto.FLOAT
    assert counts["Conv"] == 15
    assert counts["ConvTranspose"] == 2
    assert counts["Sin"] == 15
    assert counts["Pow"] == 15
    assert counts["Reciprocal"] == 15
    assert counts["Split"] == 2
    assert counts["Concat"] == 2
    assert not ({"Reshape", "Shape", "ReduceL2"} & counts.keys())
    assert [str(item.message) for item in caught] == []


def test_exact_graph_expectations_are_fixed_to_five_second_contract() -> None:
    expected = graph_expectations(EXACT_VOCODER_CONFIG)

    assert expected.input_shape == (1, 128, 430)
    assert expected.output_shape == (1, 2, 220160)
    assert expected.node_counts == {
        "Conv": 27,
        "ConvTranspose": 4,
        "Sin": 29,
        "Pow": 29,
        "Reciprocal": 29,
        "Split": 2,
        "Concat": 2,
    }


def test_publish_exports_external_data_atomically(tmp_path: Path) -> None:
    config = _tiny_config()
    source = _source_state(config)
    prepared, fp32_snakes = prepare_vocoder_state_dict(source, config)
    module = MiniMaxMusic3VocoderExport.from_prepared_state(config, prepared, fp32_snakes)
    destination = tmp_path / "published"

    artifact = publish_vocoder_module(module, destination, max_file_bytes=128 * 1024)

    assert artifact.model_path == destination / "vocoder.onnx"
    assert artifact.model_path.is_file()
    assert artifact.shards
    assert all(shard.is_file() and shard.stat().st_size <= 128 * 1024 for shard in artifact.shards)
    model = onnx.load_model(artifact.model_path, load_external_data=False)
    locations = {
        field.value
        for tensor in model.graph.initializer
        for field in tensor.external_data
        if field.key == "location"
    }
    assert locations == {shard.name for shard in artifact.shards}
    validate_vocoder_graph(artifact.model_path, config)


def test_publish_failure_restores_existing_release_and_cleans_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_config()
    source = _source_state(config)
    prepared, fp32_snakes = prepare_vocoder_state_dict(source, config)
    module = MiniMaxMusic3VocoderExport.from_prepared_state(config, prepared, fp32_snakes)
    destination = tmp_path / "published"
    destination.mkdir()
    old_files = {
        "vocoder.onnx": b"existing graph",
        "weights-000.bin": b"existing shard",
    }
    for name, contents in old_files.items():
        (destination / name).write_bytes(contents)

    original_replace = Path.replace

    def fail_staging_promotion(path: Path, target: Path) -> Path:
        if path.name == "package" and Path(target) == destination:
            raise OSError("injected staging promotion failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_staging_promotion)

    with pytest.raises(OSError, match="injected staging promotion failure"):
        publish_vocoder_module(module, destination, max_file_bytes=128 * 1024)

    assert {path.name: path.read_bytes() for path in destination.iterdir()} == old_files
    assert list(tmp_path.glob(".published-*.backup")) == []
    assert list(tmp_path.glob(".published-staging-*")) == []


def test_publish_restore_failure_preserves_existing_release_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_config()
    source = _source_state(config)
    prepared, fp32_snakes = prepare_vocoder_state_dict(source, config)
    module = MiniMaxMusic3VocoderExport.from_prepared_state(config, prepared, fp32_snakes)
    destination = tmp_path / "published"
    destination.mkdir()
    old_files = {
        "vocoder.onnx": b"existing graph",
        "weights-000.bin": b"existing shard",
    }
    for name, contents in old_files.items():
        (destination / name).write_bytes(contents)

    original_replace = Path.replace

    def fail_promotion_and_restore(path: Path, target: Path) -> Path:
        if path.name == "package" and Path(target) == destination:
            raise OSError("injected staging promotion failure")
        if path.name.endswith(".backup") and Path(target) == destination:
            raise OSError("injected backup restore failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_promotion_and_restore)

    with pytest.raises(OSError, match="injected backup restore failure"):
        publish_vocoder_module(module, destination, max_file_bytes=128 * 1024)

    backups = list(tmp_path.glob(".published-*.backup"))
    assert len(backups) == 1
    assert {path.name: path.read_bytes() for path in backups[0].iterdir()} == old_files
    assert not destination.exists()
    assert list(tmp_path.glob(".published-staging-*")) == []


def test_vocoder_release_is_atomic_hashed_and_exact(tmp_path: Path) -> None:
    source_graph = _exact_vocoder_manifest_fixture(tmp_path / "source")
    paths = ArtifactPaths.from_root(tmp_path / "artifacts", repository_root=tmp_path)

    manifest = emit_vocoder_release(paths, source_graph)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["slice"] == {
        "latentChannels": 128,
        "latentLength": 430,
        "outputSamples": 220_160,
        "sampleRate": 44_100,
        "channels": 2,
    }
    assert payload["precision"] == {
        "convolution": "float16",
        "fp32Snakes": ["blocks.0.snake1", "blocks.1.snake1"],
    }
    graph = payload["vocoder"]
    assert graph["gpuOutputs"] == []
    assert graph["path"] == "vocoder/vocoder.onnx"
    for entry in [graph, *graph["externalData"]]:
        artifact = manifest.parent / entry["path"]
        assert artifact.stat().st_size == entry["bytes"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == entry["sha256"]

    original = manifest.read_bytes()
    (source_graph.parent / "weights.bin").unlink()
    with pytest.raises(ValueError, match="external initializer file is missing"):
        emit_vocoder_release(paths, source_graph)
    assert manifest.read_bytes() == original


def test_vocoder_release_restore_failure_preserves_existing_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_graph = _exact_vocoder_manifest_fixture(tmp_path / "source")
    paths = ArtifactPaths.from_root(tmp_path / "artifacts", repository_root=tmp_path)
    manifest = emit_vocoder_release(paths, source_graph)
    release = manifest.parent
    original_manifest = manifest.read_bytes()
    original_replace = Path.replace

    def fail_promotion_and_restore(path: Path, target: Path) -> Path:
        if path.name.endswith(".staging") and Path(target) == release:
            raise OSError("injected vocoder promotion failure")
        if path.name.endswith(".backup") and Path(target) == release:
            raise OSError("injected vocoder restore failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_promotion_and_restore)

    with pytest.raises(OSError, match="injected vocoder restore failure"):
        emit_vocoder_release(paths, source_graph)

    backups = list(paths.release.glob(".vocoder-*.backup"))
    assert len(backups) == 1
    assert (backups[0] / "manifest.json").read_bytes() == original_manifest
    assert not release.exists()
    assert list(paths.release.glob(".vocoder-*.staging")) == []


@pytest.mark.skipif(
    os.environ.get("MINIMAX_RUN_REAL_VOCODER") != "1",
    reason="real vocoder gate is opt-in",
)
def test_real_checkpoint_short_oracle_and_full_fixed_export() -> None:
    source_path = Path("artifacts/source/vocoder/diffusion_pytorch_model.safetensors")
    state = load_vocoder_state_dict(source_path)
    short_config = replace(EXACT_VOCODER_CONFIG, latent_length=1)
    reference = build_vocoder_reference_module(state, short_config)
    folded = fold_vocoder_state_dict(state, short_config)
    folded_module = MiniMaxMusic3VocoderExport.from_prepared_state(short_config, folded, ())
    values = torch.arange(short_config.latent_channels, dtype=torch.float32)
    short_latents = (((values % 257) - 128) / 1024).reshape(1, 128, 1)
    with torch.no_grad():
        torch.testing.assert_close(
            folded_module(short_latents), reference(short_latents), rtol=1e-5, atol=1e-5
        )

    del reference, folded_module, folded
    prepared, fp32_snakes = prepare_vocoder_state_dict(state)
    del state
    module = MiniMaxMusic3VocoderExport.from_prepared_state(
        EXACT_VOCODER_CONFIG, prepared, fp32_snakes
    )
    artifact = publish_vocoder_module(
        module,
        Path("artifacts/work/vocoder-product-test"),
        max_file_bytes=128 * 1024 * 1024,
    )
    report = validate_vocoder_graph(artifact.model_path)
    assert report.output_shape == (1, 2, 220_160)
    assert all(shard.stat().st_size <= 128 * 1024 * 1024 for shard in artifact.shards)

    session = ort.InferenceSession(str(artifact.model_path), providers=["CPUExecutionProvider"])
    count = 128 * 430
    full_values = np.arange(count, dtype=np.float32)
    latents = (((full_values % 257) - 128) / 1024).astype(np.float16).reshape(1, 128, 430)
    waveform = session.run(["waveform"], {"latents": latents})[0]
    assert waveform.shape == (1, 2, 220_160)
    assert np.isfinite(waveform).all()


def test_tiny_cpu_oracle_matches_exported_mixed_precision_graph(tmp_path: Path) -> None:
    config = _tiny_config()
    source = _source_state(config)
    source["blocks.0.snake1.alpha"].flatten()[0] = 4.748142e-6
    prepared, fp32_snakes = prepare_vocoder_state_dict(source, config)
    module = MiniMaxMusic3VocoderExport.from_prepared_state(config, prepared, fp32_snakes)
    graph_path = export_vocoder_module(module, tmp_path / "vocoder.onnx")
    values = torch.arange(config.latent_channels * config.latent_length, dtype=torch.float32)
    latents = (((values % 257) - 128) / 1024).reshape(1, config.latent_channels, config.latent_length)

    report = run_vocoder_cpu_oracle(module, graph_path, latents.to(torch.float16))

    assert report.shape == (1, 2, config.output_samples)
    assert report.finite
    assert report.max_absolute_error <= 0.002


def _tiny_config() -> VocoderConfig:
    return VocoderConfig(
        latent_channels=8,
        decoder_input_dim=8,
        decoder_hidden_dim=16,
        upsampling_ratios=(4, 4),
        latent_length=3,
    )


def _source_state(config: VocoderConfig) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    state = {}
    for name, shape in expected_source_shapes(config).items():
        if name.endswith("weight_g"):
            tensor = torch.rand(shape, generator=generator, dtype=torch.float32) + 0.5
        elif name.endswith("alpha"):
            tensor = torch.ones(shape, dtype=torch.float32)
        else:
            tensor = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.02
        state[name] = tensor
    return state


def _exact_vocoder_manifest_fixture(directory: Path) -> Path:
    directory.mkdir(parents=True)
    nodes = []
    for operator, count in graph_expectations().node_counts.items():
        if operator == "ConvTranspose":
            continue
        nodes.extend(
            onnx.helper.make_node(operator, [], [f"{operator}_{index}"])
            for index in range(count)
        )
    for index, stride in enumerate(EXACT_VOCODER_CONFIG.upsampling_ratios):
        padding = (stride + 1) // 2
        nodes.append(
            onnx.helper.make_node(
                "ConvTranspose",
                [],
                [f"ConvTranspose_{index}"],
                kernel_shape=[2 * stride],
                strides=[stride],
                pads=[padding, padding],
            )
        )
    weight = onnx.TensorProto(
        name="weight",
        data_type=onnx.TensorProto.FLOAT16,
        dims=[2],
        data_location=onnx.TensorProto.EXTERNAL,
        external_data=[
            onnx.StringStringEntryProto(key="location", value="weights.bin"),
            onnx.StringStringEntryProto(key="offset", value="0"),
            onnx.StringStringEntryProto(key="length", value="4"),
        ],
    )
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes,
            "vocoder-release-fixture",
            [
                onnx.helper.make_tensor_value_info(
                    "latents", onnx.TensorProto.FLOAT16, [1, 128, 430]
                )
            ],
            [
                onnx.helper.make_tensor_value_info(
                    "waveform", onnx.TensorProto.FLOAT, [1, 2, 220_160]
                )
            ],
            [weight],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 18)],
    )
    graph = directory / "vocoder.onnx"
    onnx.save_model(model, graph)
    (directory / "weights.bin").write_bytes(b"\x00\x00\x00\x00")
    return graph
