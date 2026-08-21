import hashlib
import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from safetensors.numpy import save_file
import torch
import torch.nn.functional as F

from minimax_music3_webgpu.condition_encoder import (
    ConditionEncoderWeights,
    condition_encoder_model,
    export_condition_encoder,
    load_condition_encoder_weights,
    nearest_indices,
)
from minimax_music3_webgpu.manifest import emit_condition_release
from minimax_music3_webgpu.paths import ArtifactPaths


def test_nearest_indices_match_diffusers_interpolate_contract() -> None:
    for frame_count, latent_length in ((125, 430), (150, 516), (175, 602), (200, 689)):
        source = torch.arange(frame_count, dtype=torch.float32).reshape(1, 1, frame_count)
        expected = F.interpolate(source, size=latent_length, mode="nearest").numpy().reshape(-1)

        indices = nearest_indices(frame_count, latent_length)

        np.testing.assert_array_equal(indices, expected)
    assert nearest_indices(125, 430)[:8].tolist() == [0, 0, 0, 0, 1, 1, 1, 2]
    assert nearest_indices(150, 516)[430] == 124


def test_load_weights_requires_exact_keys_and_shapes(tmp_path) -> None:
    tensors = _tiny_weights()
    valid = tmp_path / "valid.safetensors"
    save_file(tensors, valid)

    weights = load_condition_encoder_weights(valid, condition_hidden_dim=4, out_dim=3)

    assert weights.layer_weight_logits.shape == (8,)
    assert weights.layer_scale.shape == (1,)
    assert weights.proj_weight.shape == (3, 4, 3)
    assert weights.proj_bias.shape == (3,)

    invalid = tmp_path / "invalid.safetensors"
    save_file({**tensors, "unexpected": np.zeros(1, dtype=np.float32)}, invalid)
    with pytest.raises(ValueError, match="exactly the four condition encoder tensors"):
        load_condition_encoder_weights(invalid, condition_hidden_dim=4, out_dim=3)


def test_fixed_graph_matches_tiny_diffusers_math_without_runtime_resize(tmp_path) -> None:
    source = tmp_path / "tiny.safetensors"
    tensors = _tiny_weights()
    save_file(tensors, source)
    weights = load_condition_encoder_weights(source, condition_hidden_dim=4, out_dim=3)
    model = condition_encoder_model(weights, frame_count=5, latent_length=17)
    model_path = tmp_path / "condition-5.onnx"
    onnx.save_model(model, model_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    frame_hiddens = np.linspace(-0.5, 0.5, 1 * 5 * 8 * 4, dtype=np.float32).reshape(1, 5, 8 * 4)

    (actual,) = session.run(None, {"frame_hiddens": frame_hiddens.astype(np.float16)})
    expected = _condition_encoder_oracle(frame_hiddens, tensors, latent_length=17)

    assert session.get_inputs()[0].shape == [1, 5, 32]
    assert session.get_outputs()[0].shape == [1, 17, 3]
    assert actual.dtype == np.float16
    np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-3)
    op_types = {node.op_type for node in model.graph.node}
    assert {"Mul", "ReduceSum", "Conv", "Gather"} <= op_types
    assert not {"Shape", "Resize", "Einsum", "Softmax"} & op_types


def test_maximum_window_matches_source_math_and_zeros_inactive_tail(tmp_path) -> None:
    source = tmp_path / "tiny.safetensors"
    tensors = _tiny_weights()
    save_file(tensors, source)
    weights = load_condition_encoder_weights(source, condition_hidden_dim=4, out_dim=3)
    maximum_model = condition_encoder_model(
        weights,
        frame_count=200,
        latent_length=689,
        maximum_window=True,
    )
    maximum_path = tmp_path / "condition-maximum.onnx"
    onnx.save_model(maximum_model, maximum_path)
    maximum_session = ort.InferenceSession(str(maximum_path), providers=["CPUExecutionProvider"])

    frame_hiddens = np.linspace(-0.5, 0.5, 1 * 200 * 8 * 4, dtype=np.float32).reshape(
        1, 200, 8 * 4
    )
    for frame_count, latent_length in ((1, 3), (125, 430), (162, 558), (200, 689)):
        padded_frames = frame_hiddens.copy()
        padded_frames[:, frame_count:] = 0
        nearest_index = np.zeros(689, dtype=np.int64)
        nearest_index[:latent_length] = nearest_indices(frame_count, latent_length)
        active_latent_mask = np.zeros((1, 689, 1), dtype=np.float16)
        active_latent_mask[:, :latent_length] = 1
        (actual,) = maximum_session.run(
            None,
            {
                "frame_hiddens": padded_frames.astype(np.float16),
                "nearest_index": nearest_index,
                "active_latent_mask": active_latent_mask,
            },
        )
        expected = _condition_encoder_oracle(
            padded_frames[:, :frame_count],
            tensors,
            latent_length=latent_length,
        )

        assert actual.shape == (1, 689, 3)
        np.testing.assert_allclose(actual[:, :latent_length], expected, atol=2e-3, rtol=2e-3)
        np.testing.assert_array_equal(actual[:, latent_length:], 0)


def test_maximum_window_uses_fixed_webgpu_safe_bindings() -> None:
    model = condition_encoder_model(
        _weights_from_tensors(_tiny_weights()),
        frame_count=200,
        latent_length=689,
        maximum_window=True,
    )

    input_shapes = {
        value.name: [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        for value in model.graph.input
    }
    assert input_shapes == {
        "frame_hiddens": [1, 200, 32],
        "nearest_index": [689],
        "active_latent_mask": [1, 689, 1],
    }
    input_types = {
        value.name: value.type.tensor_type.elem_type for value in model.graph.input
    }
    assert input_types == {
        "frame_hiddens": onnx.TensorProto.FLOAT16,
        "nearest_index": onnx.TensorProto.INT64,
        "active_latent_mask": onnx.TensorProto.FLOAT16,
    }
    assert [
        dimension.dim_value for dimension in model.graph.output[0].type.tensor_type.shape.dim
    ] == [1, 689, 3]
    assert {node.op_type for node in model.graph.node} <= {
        "Conv",
        "Gather",
        "Mul",
        "ReduceSum",
        "Reshape",
        "Transpose",
    }
    binding_sizes = {
        "frame_hiddens": 1 * 200 * 32768 * 2,
        "grouped_hiddens": 1 * 200 * 8 * 4096 * 2,
        "projected_hiddens": 1 * 2048 * 200 * 2,
        "condition": 1 * 689 * 2048 * 2,
        "proj_weight": 2048 * 4096 * 3 * 2,
    }
    assert max(binding_sizes.values()) <= 128 * 1024 * 1024


def test_maximum_window_export_bounds_external_ranges(tmp_path) -> None:
    source = tmp_path / "tiny.safetensors"
    save_file(_tiny_weights(), source)

    exported = export_condition_encoder(
        source,
        tmp_path / "packed",
        condition_hidden_dim=4,
        out_dim=3,
        frame_count=200,
        latent_length=689,
        maximum_window=True,
        max_file_bytes=256,
    )

    model = onnx.load_model(exported.model_path, load_external_data=False)
    for initializer in model.graph.initializer:
        fields = {entry.key: entry.value for entry in initializer.external_data}
        if fields:
            assert int(fields["offset"]) + int(fields["length"]) <= 256
    session = ort.InferenceSession(str(exported.model_path), providers=["CPUExecutionProvider"])
    assert [item.name for item in session.get_inputs()] == [
        "frame_hiddens",
        "nearest_index",
        "active_latent_mask",
    ]
    assert session.get_outputs()[0].shape == [1, 689, 3]


def test_export_packs_every_initializer_below_artifact_limit(tmp_path) -> None:
    source = tmp_path / "tiny.safetensors"
    save_file(_tiny_weights(), source)

    exported = export_condition_encoder(
        source,
        tmp_path / "packed",
        condition_hidden_dim=4,
        out_dim=3,
        frame_count=5,
        latent_length=17,
        max_file_bytes=256,
    )

    assert exported.model_path.name == "condition-125.onnx"
    assert exported.model_path.stat().st_size < 1024 * 1024
    assert exported.shards
    assert all(shard.size <= 256 for shard in exported.shards)
    model = onnx.load_model(exported.model_path, load_external_data=False)
    assert all(
        initializer.data_location == onnx.TensorProto.EXTERNAL
        for initializer in model.graph.initializer
        if initializer.name == "proj_weight"
    )
    session = ort.InferenceSession(str(exported.model_path), providers=["CPUExecutionProvider"])
    assert session.get_outputs()[0].shape == [1, 17, 3]


def test_condition_release_is_complete_and_failed_rebuild_preserves_it(tmp_path) -> None:
    source = tmp_path / "tiny.safetensors"
    save_file(_tiny_weights(), source)
    exported = export_condition_encoder(
        source,
        tmp_path / "packed",
        condition_hidden_dim=4,
        out_dim=3,
        frame_count=5,
        latent_length=17,
        max_file_bytes=256,
    )
    paths = ArtifactPaths.from_root(tmp_path / "artifacts", repository_root=tmp_path)

    manifest = emit_condition_release(paths, exported)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    graph = payload["conditionEncoder"]
    assert graph["gpuOutputs"] == ["condition"]
    for entry in [graph, *graph["externalData"]]:
        artifact = manifest.parent / entry["path"]
        assert artifact.stat().st_size == entry["bytes"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == entry["sha256"]

    original = manifest.read_bytes()
    exported.shards[0].path.unlink()
    with pytest.raises(FileNotFoundError):
        emit_condition_release(paths, exported)
    assert manifest.read_bytes() == original


@pytest.mark.converter_smoke
@pytest.mark.skipif(
    os.environ.get("MINIMAX_RUN_REAL_CONDITION") != "1",
    reason="set MINIMAX_RUN_REAL_CONDITION=1 for the pinned checkpoint oracle",
)
def test_real_checkpoint_matches_pinned_diffusers_oracle(tmp_path) -> None:
    from diffusers import MiniMaxMusic3ConditionEncoder

    source_dir = Path("artifacts/source/condition_encoder")
    source_weights = source_dir / "diffusion_pytorch_model.safetensors"
    reference = MiniMaxMusic3ConditionEncoder.from_pretrained(source_dir).eval().half()
    weights = load_condition_encoder_weights(source_weights)
    values = torch.linspace(-0.05, 0.05, 200 * 32768, dtype=torch.float32).reshape(1, 200, 32768)
    maximum_model = condition_encoder_model(
        weights,
        frame_count=200,
        latent_length=689,
        maximum_window=True,
    )
    maximum_path = tmp_path / "condition-maximum.onnx"
    onnx.save_model(maximum_model, maximum_path)
    maximum_session = ort.InferenceSession(str(maximum_path), providers=["CPUExecutionProvider"])

    for frame_count, latent_length in ((125, 430), (150, 516), (175, 602), (200, 689)):
        padded_frames = values.clone()
        padded_frames[:, frame_count:] = 0
        nearest_index = np.zeros(689, dtype=np.int64)
        nearest_index[:latent_length] = nearest_indices(frame_count, latent_length)
        active_latent_mask = np.zeros((1, 689, 1), dtype=np.float16)
        active_latent_mask[:, :latent_length] = 1
        (actual,) = maximum_session.run(
            None,
            {
                "frame_hiddens": padded_frames.numpy().astype(np.float16),
                "nearest_index": nearest_index,
                "active_latent_mask": active_latent_mask,
            },
        )
        with torch.no_grad():
            expected = reference(padded_frames[:, :frame_count].half()).numpy()

        assert actual.shape == (1, 689, 2048)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(
            actual[:, :latent_length].astype(np.float32), expected, atol=5e-4, rtol=5e-3
        )
        np.testing.assert_array_equal(actual[:, latent_length:], 0)


def _tiny_weights() -> dict[str, np.ndarray]:
    return {
        "layer_weight_logits": np.linspace(-1, 1, 8, dtype=np.float32),
        "layer_scale": np.array([0.75], dtype=np.float32),
        "proj.weight": np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3) / 100,
        "proj.bias": np.linspace(-0.2, 0.2, 3, dtype=np.float32),
    }


def _weights_from_tensors(tensors: dict[str, np.ndarray]) -> ConditionEncoderWeights:
    return ConditionEncoderWeights(
        layer_weight_logits=tensors["layer_weight_logits"],
        layer_scale=tensors["layer_scale"],
        proj_weight=tensors["proj.weight"],
        proj_bias=tensors["proj.bias"],
    )


def _condition_encoder_oracle(
    frame_hiddens: np.ndarray,
    weights: dict[str, np.ndarray],
    *,
    latent_length: int,
) -> np.ndarray:
    hidden = torch.from_numpy(frame_hiddens).reshape(1, frame_hiddens.shape[1], 8, 4)
    layer_weights = torch.softmax(torch.from_numpy(weights["layer_weight_logits"]), dim=0)
    hidden = (hidden * layer_weights.reshape(1, 1, 8, 1)).sum(dim=2)
    hidden = hidden * torch.from_numpy(weights["layer_scale"])
    hidden = F.conv1d(
        hidden.transpose(1, 2),
        torch.from_numpy(weights["proj.weight"]),
        torch.from_numpy(weights["proj.bias"]),
        padding=1,
    )
    hidden = F.interpolate(hidden, size=latent_length, mode="nearest")
    return hidden.transpose(1, 2).numpy().astype(np.float16)
