import json
import hashlib
import os
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from safetensors.torch import save_file
import torch

import minimax_music3_webgpu.flow_transformer as flow_transformer_module
from minimax_music3_webgpu.flow_transformer import (
    FlowTransformerConfig,
    exact_flow_schedule,
    expected_flow_shapes,
    export_flow_step,
    export_maximum_flow_step,
    flow_step_reference,
    fixed_rope_tables,
    inspect_flow_source,
    maximum_flow_binding_bytes,
    open_flow_state,
    validate_flow_graph,
    validate_maximum_flow_graph,
    validate_flow_metadata,
)
from minimax_music3_webgpu.manifest import emit_flow_release
from minimax_music3_webgpu.paths import ArtifactPaths


FIXTURES = Path(__file__).parents[1] / "fixtures"


def tiny_config() -> FlowTransformerConfig:
    return FlowTransformerConfig(
        in_channels=4,
        condition_dim=8,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=4,
        ff_inner_dim=16,
        rotary_dim=4,
        fourier_embedding_dim=8,
    )


def state_for(config: FlowTransformerConfig) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return {
        name: torch.randn(shape, generator=generator, dtype=torch.float32) * 0.03
        for name, shape in expected_flow_shapes(config).items()
    }


def test_full_source_contract_has_all_441_exact_keys() -> None:
    config = FlowTransformerConfig()
    shapes = expected_flow_shapes(config)
    q4_shapes = {
        key: shape
        for key, shape in shapes.items()
        if len(shape) == 2 and key != "time_proj.weight"
    }
    q4_bytes = sum(
        output_dim * ((input_dim + 127) // 128) * (64 + 2)
        for output_dim, input_dim in q4_shapes.values()
    )
    fp16_bytes = sum(
        int(np.prod(shape, dtype=np.int64)) * 2
        for key, shape in shapes.items()
        if key not in q4_shapes
    )

    validate_flow_metadata(config, shapes)

    assert len(shapes) == 441
    assert config.num_attention_heads == 32
    assert config.hidden_size == 2048
    assert config.ff_inner_dim == 8192
    assert config.in_channels == 128
    assert config.rotary_dim == 32
    assert sum(np.prod(shape, dtype=np.int64) for shape in shapes.values()) == 2_431_905_920
    assert len(q4_shapes) == 220
    assert q4_bytes == 1_250_709_504
    assert fp16_bytes == 12_574_976
    assert q4_bytes + fp16_bytes == 1_263_284_480
    assert shapes["preprocess_conv.weight"] == (2304, 2304, 1)
    assert shapes["transformer_blocks.35.ff_in.weight"] == (16384, 2048)
    assert shapes["postprocess_conv.weight"] == (128, 128, 1)


def test_source_inspection_reads_shard_metadata_without_combining_tensors(tmp_path: Path) -> None:
    config = tiny_config()
    state = state_for(config)
    names = sorted(state)
    shards = {
        "flow-00001.safetensors": names[::2],
        "flow-00002.safetensors": names[1::2],
    }
    weight_map = {}
    for filename, keys in shards.items():
        save_file({key: state[key] for key in keys}, tmp_path / filename)
        weight_map.update({key: filename for key in keys})
    (tmp_path / "config.json").write_text(json.dumps(config.__dict__), encoding="utf-8")
    (tmp_path / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )

    report = inspect_flow_source(tmp_path)

    assert report.config == config
    assert report.tensor_count == len(state)
    assert report.total_parameters == sum(tensor.numel() for tensor in state.values())
    assert report.shards == tuple(sorted(shards))
    with open_flow_state(tmp_path) as lazy_state:
        assert lazy_state.shapes == expected_flow_shapes(config)
        assert lazy_state["proj_out.weight"].shape == state["proj_out.weight"].shape


def test_source_contract_rejects_missing_or_wrong_tensor() -> None:
    config = tiny_config()
    shapes = expected_flow_shapes(config)
    shapes.pop("transformer_blocks.0.attn.to_q.weight")
    shapes["proj_out.weight"] = (1, 1)

    with pytest.raises(ValueError, match="missing=.*to_q.*wrong_shapes=.*proj_out"):
        validate_flow_metadata(config, shapes)


def test_exact_schedule_matches_frozen_float32_bits() -> None:
    fixture = json.loads((FIXTURES / "flow-schedule.json").read_text(encoding="utf-8"))
    schedule = exact_flow_schedule()

    assert schedule.timesteps.dtype == np.float32
    assert schedule.dts.dtype == np.float32
    assert schedule.timesteps.view(np.uint32).tolist() == fixture["timestep_f32_bits"]
    assert schedule.dts.view(np.uint32).tolist() == fixture["dt_f32_bits"]
    assert len(set(fixture["dt_f32_bits"])) == 2


def test_exact_schedule_supports_any_positive_step_count() -> None:
    schedule = exact_flow_schedule(7)

    assert schedule.timesteps.view(np.uint32).tolist() == [
        0,
        1041385764,
        1049774372,
        1054567862,
        1058162980,
        1060559726,
        1062956471,
    ]
    assert schedule.dts.view(np.uint32).tolist() == [
        1041385764,
        1041385764,
        1041385764,
        1041385764,
        1041385768,
        1041385764,
        1041385764,
    ]
    assert exact_flow_schedule(1).timesteps.tolist() == [0.0]
    assert exact_flow_schedule(1).dts.tolist() == [1.0]


@pytest.mark.parametrize("num_steps", [0, -1, 1.5, True, 2**53])
def test_exact_schedule_rejects_invalid_step_counts(num_steps) -> None:
    with pytest.raises(ValueError, match="positive safe integer"):
        exact_flow_schedule(num_steps)


def test_fixed_production_rope_tables_match_the_pinned_431_token_contract() -> None:
    cosine, sine = fixed_rope_tables(431, 32)

    assert cosine.shape == sine.shape == (1, 431, 1, 32)
    assert cosine.dtype == sine.dtype == np.float16
    assert hashlib.sha256(cosine.tobytes()).hexdigest() == "ad452ef725ffc7e7029cc2b5aa01e66949070cbee56c395367b3ace6cae0824f"
    assert hashlib.sha256(sine.tobytes()).hexdigest() == "1c212c16fd1288ff63e95b87bd5c6448c7b15081222bf07fc2f4bf5f18f44b08"


def test_one_block_fixed_export_matches_reference_and_has_no_dynamic_shape_ops(tmp_path: Path) -> None:
    config = tiny_config()
    state = state_for(config)
    graph = export_flow_step(
        state,
        tmp_path / "flow-step.onnx",
        config=config,
        latent_length=6,
        quantize=False,
    )
    report = validate_flow_graph(graph, config=config, latent_length=6, quantized=False)
    session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    generator = np.random.default_rng(5)
    latents = generator.standard_normal((1, 4, 6), dtype=np.float32).astype(np.float16)
    condition = generator.standard_normal((1, 6, 8), dtype=np.float32).astype(np.float16)
    timestep = np.array([0.25], dtype=np.float16)
    dt = np.array([0.125], dtype=np.float32)

    guidance = np.array([1.25], dtype=np.float16)
    actual = session.run(
        None,
        {
            "latents": latents,
            "condition": condition,
            "timestep": timestep,
            "dt": dt,
            "guidance": guidance,
        },
    )[0]
    expected = flow_step_reference(
        state, latents, condition, timestep, dt, guidance, config=config
    )

    assert report.matmul_nbits_nodes == 0
    assert report.dynamic_shape_ops == ()
    assert session.get_inputs()[-1].name == "guidance"
    assert session.get_inputs()[-1].type == "tensor(float16)"
    np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-3)


def test_one_block_maximum_graph_matches_legacy_and_162_frame_558_latent_tail(
    tmp_path: Path,
) -> None:
    config = tiny_config()
    state = state_for(config)
    maximum_graph = export_maximum_flow_step(
        state,
        tmp_path / "flow-step-maximum.onnx",
        config=config,
        quantize=False,
    )
    report = validate_maximum_flow_graph(maximum_graph, config=config, quantized=False)
    maximum_session = ort.InferenceSession(str(maximum_graph), providers=["CPUExecutionProvider"])
    generator = np.random.default_rng(17)
    latents = generator.standard_normal((1, config.in_channels, 689), dtype=np.float32).astype(np.float16)
    condition = generator.standard_normal((1, 689, config.condition_dim), dtype=np.float32).astype(np.float16)
    timestep = np.array([0.25], dtype=np.float16)
    dt = np.array([0.125], dtype=np.float32)
    disabled_overlap = {
        "noise_prompt": np.zeros((1, config.in_channels, 172), dtype=np.float16),
        "previous_latent": np.zeros((1, config.in_channels, 172), dtype=np.float16),
        "overlap_enabled": np.zeros((1,), dtype=np.float16),
    }

    assert report.dynamic_shape_ops == ()
    for latent_length in (430, 516, 558, 602, 689):
        fixed_graph = export_flow_step(
            state,
            tmp_path / f"flow-step-{latent_length}.onnx",
            config=config,
            latent_length=latent_length,
            quantize=False,
        )
        fixed_session = ort.InferenceSession(str(fixed_graph), providers=["CPUExecutionProvider"])
        (fixed_output,) = fixed_session.run(
            None,
            {
                "latents": latents[..., :latent_length],
                "condition": condition[:, :latent_length],
                "timestep": timestep,
                "dt": dt,
                "guidance": np.array([1.7], dtype=np.float16),
            },
        )
        reference_output = flow_step_reference(
            state,
            latents[..., :latent_length],
            condition[:, :latent_length],
            timestep,
            dt,
            np.array([1.7], dtype=np.float16),
            config=config,
        )
        active_latent_mask = np.zeros((1, 689, 1), dtype=np.float16)
        active_latent_mask[:, :latent_length] = 1
        key_attention_bias = np.full((1, 1, 1, 690), np.finfo(np.float16).min, dtype=np.float16)
        key_attention_bias[..., : latent_length + 1] = 0
        (actual,) = maximum_session.run(
            None,
            {
                "latents": latents,
                "condition": condition,
                "timestep": timestep,
                "dt": dt,
                "guidance": np.array([1.7], dtype=np.float16),
                "active_latent_mask": active_latent_mask,
                "key_attention_bias": key_attention_bias,
                **disabled_overlap,
            },
        )

        np.testing.assert_allclose(fixed_output, reference_output, atol=2e-3, rtol=2e-3)
        np.testing.assert_allclose(
            actual[..., :latent_length], reference_output, atol=2e-3, rtol=2e-3
        )
        assert np.count_nonzero(actual[..., latent_length:]) == 0


def test_maximum_graph_applies_pinned_overlap_before_transformer_and_euler(tmp_path: Path) -> None:
    config = tiny_config()
    state = state_for(config)
    maximum_graph = export_maximum_flow_step(
        state,
        tmp_path / "flow-step-maximum.onnx",
        config=config,
        quantize=False,
    )
    fixed_graph = export_flow_step(
        state,
        tmp_path / "flow-step-430.onnx",
        config=config,
        latent_length=430,
        quantize=False,
    )
    maximum_session = ort.InferenceSession(str(maximum_graph), providers=["CPUExecutionProvider"])
    fixed_session = ort.InferenceSession(str(fixed_graph), providers=["CPUExecutionProvider"])
    generator = np.random.default_rng(19)
    latents = generator.standard_normal((1, config.in_channels, 689), dtype=np.float32).astype(np.float16)
    condition = generator.standard_normal((1, 689, config.condition_dim), dtype=np.float32).astype(np.float16)
    noise_prompt = generator.standard_normal(
        (1, config.in_channels, 172), dtype=np.float32
    ).astype(np.float16)
    previous_latent = generator.standard_normal(
        (1, config.in_channels, 172), dtype=np.float32
    ).astype(np.float16)
    timestep = np.array([0.375], dtype=np.float16)
    dt = np.array([1 / 30], dtype=np.float32)
    blended = latents[..., :430].copy()
    time_value = torch.from_numpy(timestep)
    blended[..., :172] = (
        (1.0 - (1.0 - 1e-6) * time_value) * torch.from_numpy(noise_prompt)
        + time_value * torch.from_numpy(previous_latent)
    ).numpy()
    (expected,) = fixed_session.run(
        None,
        {
            "latents": blended,
            "condition": condition[:, :430],
            "timestep": timestep,
            "dt": dt,
            "guidance": np.array([1.7], dtype=np.float16),
        },
    )
    active_latent_mask = np.zeros((1, 689, 1), dtype=np.float16)
    active_latent_mask[:, :430] = 1
    key_attention_bias = np.full((1, 1, 1, 690), np.finfo(np.float16).min, dtype=np.float16)
    key_attention_bias[..., :431] = 0

    (actual,) = maximum_session.run(
        None,
        {
            "latents": latents,
            "condition": condition,
            "timestep": timestep,
            "dt": dt,
            "guidance": np.array([1.7], dtype=np.float16),
            "active_latent_mask": active_latent_mask,
            "key_attention_bias": key_attention_bias,
            "noise_prompt": noise_prompt,
            "previous_latent": previous_latent,
            "overlap_enabled": np.ones((1,), dtype=np.float16),
        },
    )

    np.testing.assert_allclose(actual[..., :430], expected, atol=2e-3, rtol=2e-3)
    assert np.count_nonzero(actual[..., 430:]) == 0


def test_maximum_q4_graph_preserves_weights_and_has_bounded_static_contract(tmp_path: Path) -> None:
    config = tiny_config()
    state = state_for(config)
    fixed_graph = export_flow_step(
        state,
        tmp_path / "fixed" / "flow-step-q4.onnx",
        config=config,
        latent_length=6,
        quantize=True,
        external_data=True,
        max_file_bytes=16 * 1024,
    )
    maximum_graph = export_maximum_flow_step(
        state,
        tmp_path / "maximum" / "flow-step-q4.onnx",
        config=config,
        quantize=True,
        external_data=True,
        max_file_bytes=16 * 1024,
    )

    report = validate_maximum_flow_graph(maximum_graph, config=config, quantized=True)
    fixed_model = onnx.load_model(fixed_graph, load_external_data=True)
    maximum_model = onnx.load_model(maximum_graph, load_external_data=True)
    fixed_q4 = {
        tensor.name: onnx.numpy_helper.to_array(tensor)
        for tensor in fixed_model.graph.initializer
        if tensor.name.endswith((".q4", ".scales"))
    }
    maximum_q4 = {
        tensor.name: onnx.numpy_helper.to_array(tensor)
        for tensor in maximum_model.graph.initializer
        if tensor.name.endswith((".q4", ".scales"))
    }

    assert report.matmul_nbits_nodes == 9
    assert report.external_locations
    assert all(
        (maximum_graph.parent / location).stat().st_size <= 16 * 1024
        for location in report.external_locations
    )
    assert report.max_initializer_bytes <= 128 * 1024 * 1024
    assert report.max_activation_bytes <= 128 * 1024 * 1024
    assert set(report.operator_types) <= {
        "Add",
        "Cast",
        "Concat",
        "Conv",
        "Cos",
        "LayerNormalization",
        "MatMul",
        "MatMulNBits",
        "Mul",
        "Neg",
        "Reshape",
        "Sigmoid",
        "Sin",
        "Slice",
        "Softmax",
        "Split",
        "Sub",
        "Transpose",
        "Unsqueeze",
    }
    assert fixed_q4.keys() == maximum_q4.keys()
    for name, fixed_value in fixed_q4.items():
        np.testing.assert_array_equal(maximum_q4[name], fixed_value)
    assert maximum_flow_binding_bytes(FlowTransformerConfig(), quantized=True) == (
        16_777_216,
        60_940_800,
    )


def test_one_block_q4_graph_has_nine_b32_matmuls_and_two_fp16_linear_weights(tmp_path: Path) -> None:
    config = tiny_config()
    state = state_for(config)
    graph = export_flow_step(
        state,
        tmp_path / "flow-step-q4.onnx",
        config=config,
        latent_length=6,
        quantize=True,
        external_data=True,
        max_file_bytes=16 * 1024,
    )
    report = validate_flow_graph(graph, config=config, latent_length=6, quantized=True)
    model = onnx.load_model(graph, load_external_data=False)

    assert report.matmul_nbits_nodes == 9
    assert report.external_locations
    assert all((graph.parent / name).stat().st_size <= 16 * 1024 for name in report.external_locations)
    for node in model.graph.node:
        if node.op_type != "MatMulNBits":
            continue
        attributes = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
        weight_key = node.name.removesuffix(".MatMulNBits")
        output_dim, input_dim = expected_flow_shapes(config)[weight_key]
        assert attributes == {
            "K": input_dim,
            "N": output_dim,
            "accuracy_level": 4,
            "bits": 4,
            "block_size": 32,
        }
        assert len(node.input) == 3

    nodes = {node.name: node for node in model.graph.node}
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    for weight_key in ("time_proj.weight", "proj_out.weight"):
        assert nodes[f"{weight_key}.MatMul"].op_type == "MatMul"
        initializer = initializers[weight_key]
        assert initializer.data_type == onnx.TensorProto.FLOAT16
        assert tuple(initializer.dims) == tuple(reversed(expected_flow_shapes(config)[weight_key]))

    session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    generator = np.random.default_rng(11)
    result = session.run(
        None,
        {
            "latents": generator.standard_normal((1, 4, 6), dtype=np.float32).astype(np.float16),
            "condition": generator.standard_normal((1, 6, 8), dtype=np.float32).astype(np.float16),
            "timestep": np.array([0.5], dtype=np.float16),
            "dt": np.array([1 / 30], dtype=np.float32),
            "guidance": np.array([1.7], dtype=np.float16),
        },
    )[0]
    assert result.shape == (1, 4, 6)
    assert np.isfinite(result).all()


def test_q4_validator_rejects_wrong_matmul_nbits_dimensions(tmp_path: Path) -> None:
    config = tiny_config()
    graph = export_flow_step(
        state_for(config),
        tmp_path / "flow-step-q4.onnx",
        config=config,
        latent_length=6,
        quantize=True,
    )
    model = onnx.load_model(graph, load_external_data=False)
    q4_node = next(node for node in model.graph.node if node.op_type == "MatMulNBits")
    k_attribute = next(item for item in q4_node.attribute if item.name == "K")
    k_attribute.i += 1
    onnx.save_model(model, graph)

    with pytest.raises(ValueError, match="MatMulNBits.*dimensions"):
        validate_flow_graph(graph, config=config, latent_length=6, quantized=True)


def test_failed_external_export_preserves_published_generation_and_removes_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tiny_config()
    state = state_for(config)
    graph = export_flow_step(
        state,
        tmp_path / "flow-step.onnx",
        config=config,
        latent_length=6,
        quantize=True,
        external_data=True,
        max_file_bytes=16 * 1024,
    )
    model = onnx.load_model(graph, load_external_data=False)
    locations = {
        item.value
        for tensor in model.graph.initializer
        for item in tensor.external_data
        if item.key == "location"
    }
    published = {graph: graph.read_bytes()}
    published.update({graph.parent / name: (graph.parent / name).read_bytes() for name in locations})
    original_add = flow_transformer_module._InitializerWriter.add
    calls = 0

    def fail_during_write(writer, name, value):
        nonlocal calls
        calls += 1
        if calls == 12:
            raise RuntimeError("injected external writer failure")
        return original_add(writer, name, value)

    monkeypatch.setattr(flow_transformer_module._InitializerWriter, "add", fail_during_write)

    with pytest.raises(RuntimeError, match="injected external writer failure"):
        export_flow_step(
            state,
            graph,
            config=config,
            latent_length=6,
            quantize=True,
            external_data=True,
            max_file_bytes=16 * 1024,
        )

    assert {path: path.read_bytes() for path in published} == published
    assert not list(tmp_path.glob(".flow-step.onnx.staging-*"))


def test_flow_release_is_hashed_exact_and_failed_rebuild_preserves_it(tmp_path: Path) -> None:
    config = tiny_config()
    source_graph = export_flow_step(
        state_for(config),
        tmp_path / "product" / "flow-step.onnx",
        config=config,
        latent_length=6,
        quantize=True,
        external_data=True,
        max_file_bytes=16 * 1024,
    )
    paths = ArtifactPaths.from_root(tmp_path / "artifacts", repository_root=tmp_path)

    manifest = emit_flow_release(paths, source_graph)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["slice"] == {
        "semanticFrames": 125,
        "latentLength": 430,
        "flowSteps": 30,
        "flowGuidance": 1.7,
    }
    assert payload["quantization"] == {
        "bits": 4,
        "blockSize": 32,
        "accuracyLevel": 4,
        "symmetric": True,
    }
    assert payload["precision"] == {
        "float16Weights": ["time_proj.weight", "proj_out.weight"],
    }
    assert payload["webgpu"]["requiredLimits"] == {
        "maxStorageBufferBindingSize": 128 * 1024 * 1024,
        "maxStorageBuffersPerShaderStage": 9,
    }
    graph = payload["flow"]
    assert graph["gpuOutputs"] == ["next_latents"]
    for entry in [graph, *graph["externalData"]]:
        artifact = manifest.parent / entry["path"]
        assert artifact.stat().st_size == entry["bytes"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == entry["sha256"]

    original = manifest.read_bytes()
    source_model = onnx.load_model(source_graph, load_external_data=False)
    source_location = next(
        item.value
        for tensor in source_model.graph.initializer
        for item in tensor.external_data
        if item.key == "location"
    )
    (source_graph.parent / source_location).unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        emit_flow_release(paths, source_graph)
    assert manifest.read_bytes() == original
    assert not list(paths.release.glob(".flow-*.staging"))


def pinned_real_layer_zero():
    from diffusers import MiniMaxMusic3Transformer1DModel

    source = Path("artifacts/source/transformer")
    receipt = json.loads(Path("artifacts/receipts/source-acoustic.json").read_text(encoding="utf-8"))
    assert receipt["revision"] == "fbdf52fbaaca799592917417eb05f1899f1255ec"
    config = FlowTransformerConfig(num_layers=1)
    reference = MiniMaxMusic3Transformer1DModel(**config.__dict__).eval()
    with open_flow_state(source) as source_state, torch.no_grad():
        for key, target in reference.state_dict().items():
            target.copy_(source_state[key])
    reference.to(torch.float16)
    return config, reference


@pytest.mark.converter_smoke
@pytest.mark.skipif(
    os.environ.get("MINIMAX_RUN_REAL_FLOW") != "1",
    reason="set MINIMAX_RUN_REAL_FLOW=1 for the pinned checkpoint layer-0 oracle",
)
def test_real_layer_zero_boundary_matches_pinned_diffusers_oracle(tmp_path: Path) -> None:
    config, reference = pinned_real_layer_zero()
    graph = export_flow_step(
        reference.state_dict(),
        tmp_path / "flow-step-layer-zero.onnx",
        config=config,
        latent_length=2,
        quantize=False,
        external_data=True,
    )
    validate_flow_graph(graph, config=config, latent_length=2, quantized=False)
    generator = torch.Generator().manual_seed(23)
    latents = (torch.randn((1, 128, 2), generator=generator) * 0.05).to(torch.float16)
    condition = (torch.randn((1, 2, 2048), generator=generator) * 0.05).to(torch.float16)
    timestep = torch.tensor([0.25], dtype=torch.float16)
    dt = exact_flow_schedule().dts[:1]

    with torch.no_grad():
        velocity = reference(
            torch.cat((latents, latents), dim=0),
            torch.cat((timestep, timestep), dim=0),
            torch.cat((condition, torch.zeros_like(condition)), dim=0),
            return_dict=False,
        )[0]
        guided = velocity[1:2] + 1.7 * (velocity[0:1] - velocity[1:2])
        expected = (latents.float() + torch.from_numpy(dt).reshape(1, 1, 1) * guided.float()).half()

    session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    (actual,) = session.run(
        None,
        {
            "latents": latents.numpy(),
            "condition": condition.numpy(),
            "timestep": timestep.numpy(),
            "dt": dt,
            "guidance": np.array([1.7], dtype=np.float16),
        },
    )

    assert actual.shape == (1, 128, 2)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected.numpy(), atol=3e-3, rtol=3e-3)


@pytest.mark.converter_smoke
@pytest.mark.skipif(
    os.environ.get("MINIMAX_RUN_REAL_FLOW_MAXIMUM") != "1",
    reason="set MINIMAX_RUN_REAL_FLOW_MAXIMUM=1 for the pinned layer-0 maximum-window oracle",
)
def test_real_layer_zero_maximum_graph_matches_pinned_diffusers_at_all_active_lengths(
    tmp_path: Path,
) -> None:
    config, reference = pinned_real_layer_zero()
    state = reference.state_dict()
    maximum_graph = export_maximum_flow_step(
        state,
        tmp_path / "maximum" / "flow-step-layer-zero.onnx",
        config=config,
        quantize=False,
        external_data=True,
    )
    maximum_session = ort.InferenceSession(
        str(maximum_graph), providers=["CPUExecutionProvider"]
    )
    generator = torch.Generator().manual_seed(29)
    latents = (torch.randn((1, 128, 689), generator=generator) * 0.05).to(torch.float16)
    condition = (torch.randn((1, 689, 2048), generator=generator) * 0.05).to(
        torch.float16
    )
    timestep = torch.tensor([0.25], dtype=torch.float16)
    dt = exact_flow_schedule().dts[:1]
    disabled_overlap = {
        "noise_prompt": np.zeros((1, 128, 172), dtype=np.float16),
        "previous_latent": np.zeros((1, 128, 172), dtype=np.float16),
        "overlap_enabled": np.zeros((1,), dtype=np.float16),
    }

    for latent_length in (430, 516, 602, 689):
        active_latents = latents[..., :latent_length]
        active_condition = condition[:, :latent_length]
        fixed_graph = export_flow_step(
            state,
            tmp_path / f"fixed-{latent_length}" / "flow-step-layer-zero.onnx",
            config=config,
            latent_length=latent_length,
            quantize=False,
            external_data=True,
        )
        fixed_session = ort.InferenceSession(
            str(fixed_graph), providers=["CPUExecutionProvider"]
        )
        (fixed_output,) = fixed_session.run(
            None,
            {
                "latents": active_latents.numpy(),
                "condition": active_condition.numpy(),
                "timestep": timestep.numpy(),
                "dt": dt,
                "guidance": np.array([1.7], dtype=np.float16),
            },
        )
        with torch.no_grad():
            velocity = reference(
                torch.cat((active_latents, active_latents), dim=0),
                torch.cat((timestep, timestep), dim=0),
                torch.cat((active_condition, torch.zeros_like(active_condition)), dim=0),
                return_dict=False,
            )[0]
            guided = velocity[1:2] + 1.7 * (velocity[0:1] - velocity[1:2])
            reference_output = (
                active_latents.float()
                + torch.from_numpy(dt).reshape(1, 1, 1) * guided.float()
            ).half()
        active_latent_mask = np.zeros((1, 689, 1), dtype=np.float16)
        active_latent_mask[:, :latent_length] = 1
        key_attention_bias = np.full(
            (1, 1, 1, 690), np.finfo(np.float16).min, dtype=np.float16
        )
        key_attention_bias[..., : latent_length + 1] = 0
        (maximum_output,) = maximum_session.run(
            None,
            {
                "latents": latents.numpy(),
                "condition": condition.numpy(),
                "timestep": timestep.numpy(),
                "dt": dt,
                "guidance": np.array([1.7], dtype=np.float16),
                "active_latent_mask": active_latent_mask,
                "key_attention_bias": key_attention_bias,
                **disabled_overlap,
            },
        )

        np.testing.assert_allclose(
            fixed_output, reference_output.numpy(), atol=3e-3, rtol=3e-3
        )
        np.testing.assert_allclose(
            maximum_output[..., :latent_length],
            reference_output.numpy(),
            atol=3e-3,
            rtol=3e-3,
        )
        np.testing.assert_allclose(
            maximum_output[..., :latent_length], fixed_output, atol=3e-3, rtol=3e-3
        )
        assert np.count_nonzero(maximum_output[..., latent_length:]) == 0


@pytest.mark.converter_smoke
@pytest.mark.skipif(
    os.environ.get("MINIMAX_RUN_REAL_FLOW_PRODUCT") != "1",
    reason="set MINIMAX_RUN_REAL_FLOW_PRODUCT=1 for the full pinned q4 product gate",
)
def test_full_pinned_q4_product_gate() -> None:
    source = Path("artifacts/source/transformer")
    receipt = json.loads(Path("artifacts/receipts/source-acoustic.json").read_text(encoding="utf-8"))
    assert receipt["revision"] == "fbdf52fbaaca799592917417eb05f1899f1255ec"
    source_report = inspect_flow_source(source)
    assert source_report.tensor_count == 441
    assert source_report.total_parameters == 2_431_905_920
    output = Path("artifacts/work/flow-product-test/flow-step.onnx")
    started = time.perf_counter()
    cpu_started = time.process_time()

    def validate_product():
        report = validate_flow_graph(output)
        onnx.checker.check_model(output)
        model = onnx.load_model(output, load_external_data=False)
        q4_nodes = [node for node in model.graph.node if node.op_type == "MatMulNBits"]
        assert len(q4_nodes) == report.matmul_nbits_nodes == 219
        assert report.dynamic_shape_ops == ()
        assert report.external_locations
        for node in q4_nodes:
            attributes = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
            assert len(node.input) == 3
            assert attributes["bits"] == 4
            assert attributes["block_size"] == 32
            assert attributes["accuracy_level"] == 4
        nodes = {node.name: node for node in model.graph.node}
        initializers = {tensor.name: tensor for tensor in model.graph.initializer}
        for weight_key in ("time_proj.weight", "proj_out.weight"):
            assert nodes[f"{weight_key}.MatMul"].op_type == "MatMul"
            assert initializers[weight_key].data_type == onnx.TensorProto.FLOAT16
        external_files = [output.parent / location for location in report.external_locations]
        assert all(path.stat().st_size <= 128 * 1024 * 1024 for path in external_files)
        return external_files

    try:
        external_files = validate_product()
    except (AssertionError, FileNotFoundError, OSError, ValueError, onnx.checker.ValidationError):
        with open_flow_state(source) as source_state:
            export_flow_step(
                source_state,
                output,
                latent_length=430,
                quantize=True,
                external_data=True,
                max_file_bytes=128 * 1024 * 1024,
            )
        external_files = validate_product()
    total_bytes = output.stat().st_size + sum(path.stat().st_size for path in external_files)
    print(
        "flow product gate: "
        f"elapsed={time.perf_counter() - started:.3f}s "
        f"cpu={time.process_time() - cpu_started:.3f}s "
        f"graph={output.stat().st_size} "
        f"shards={len(external_files)} "
        f"total={total_bytes}"
    )
