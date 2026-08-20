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
    condition_encoder_model,
    export_condition_encoder,
    load_condition_encoder_weights,
    nearest_indices,
)


def test_nearest_indices_match_diffusers_interpolate_contract() -> None:
    source = torch.arange(125, dtype=torch.float32).reshape(1, 1, 125)
    expected = F.interpolate(source, size=430, mode="nearest").numpy().reshape(-1)

    indices = nearest_indices(125, 430)

    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(indices[:8], np.array([0, 0, 0, 0, 1, 1, 1, 2]))
    assert indices[-1] == 124


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


@pytest.mark.converter_smoke
@pytest.mark.skipif(
    os.environ.get("MINIMAX_RUN_REAL_CONDITION") != "1",
    reason="set MINIMAX_RUN_REAL_CONDITION=1 for the pinned checkpoint oracle",
)
def test_real_checkpoint_matches_pinned_diffusers_oracle(tmp_path) -> None:
    from diffusers import MiniMaxMusic3ConditionEncoder

    source_dir = Path("artifacts/source/condition_encoder")
    source_weights = source_dir / "diffusion_pytorch_model.safetensors"
    reference = MiniMaxMusic3ConditionEncoder.from_pretrained(source_dir).eval()
    values = torch.linspace(-0.05, 0.05, 125 * 32768, dtype=torch.float32).reshape(1, 125, 32768)
    with torch.no_grad():
        expected = reference(values).numpy()

    exported = export_condition_encoder(source_weights, tmp_path / "condition-125")
    session = ort.InferenceSession(str(exported.model_path), providers=["CPUExecutionProvider"])
    (actual,) = session.run(None, {"frame_hiddens": values.numpy().astype(np.float16)})

    assert actual.shape == (1, 430, 2048)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual.astype(np.float32), expected, atol=5e-4, rtol=5e-3)


def _tiny_weights() -> dict[str, np.ndarray]:
    return {
        "layer_weight_logits": np.linspace(-1, 1, 8, dtype=np.float32),
        "layer_scale": np.array([0.75], dtype=np.float32),
        "proj.weight": np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3) / 100,
        "proj.bias": np.linspace(-0.2, 0.2, 3, dtype=np.float32),
    }


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
