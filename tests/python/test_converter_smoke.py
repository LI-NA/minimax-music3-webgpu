import subprocess
import sys

import numpy as np
import onnxruntime as ort
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from minimax_music3_webgpu.global_decoder import builder_arguments, rewrite_attention_mask_for_gqa, validate_global_decoder


@pytest.mark.converter_smoke
def test_tiny_qwen3_builder_prefill_and_cached_decode_are_finite(tmp_path) -> None:
    torch.manual_seed(0)
    source = tmp_path / "language_model"
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        head_dim=16,
        max_position_embeddings=128,
    )
    Qwen3ForCausalLM(config).save_pretrained(source, safe_serialization=True)
    fused = _build(source, tmp_path, "fused", True)
    unfused = _build(source, tmp_path, "unfused", False)
    fused_report = validate_global_decoder(fused, expected_layers=2)
    unfused_report = validate_global_decoder(unfused, expected_layers=2)
    assert fused_report.attention_nodes == unfused_report.attention_nodes == 2
    assert fused_report.matmul_nbits_nodes == unfused_report.matmul_nbits_nodes

    fused_session = ort.InferenceSession(str(fused), providers=["CPUExecutionProvider"])
    unfused_session = ort.InferenceSession(str(unfused), providers=["CPUExecutionProvider"])
    assert [(item.name, item.shape, item.type) for item in fused_session.get_inputs()] == [(item.name, item.shape, item.type) for item in unfused_session.get_inputs()]
    assert [(item.name, item.shape, item.type) for item in fused_session.get_outputs()] == [(item.name, item.shape, item.type) for item in unfused_session.get_outputs()]

    session = unfused_session
    inputs = {
        "inputs_embeds": np.zeros((2, 2, 64), dtype=np.float16),
        "seqlens_k": np.array([1, 1], dtype=np.int32),
        "total_seq_len": np.array(2, dtype=np.int32),
    }
    for item in session.get_inputs():
        if "past" in item.name:
            inputs[item.name] = np.zeros((2, 2, 0, 16), dtype=np.float16)
    outputs = session.run(None, inputs)
    assert all(np.isfinite(output).all() for output in outputs)
    first_cache = {item.name: value for item, value in zip(session.get_outputs(), outputs, strict=True) if "present" in item.name}

    for expected_length in (3, 4):
        step_inputs = {
            "inputs_embeds": np.zeros((2, 1, 64), dtype=np.float16),
            "seqlens_k": np.array([expected_length - 1, expected_length - 1], dtype=np.int32),
            "total_seq_len": np.array(expected_length, dtype=np.int32),
        }
        for item in session.get_inputs():
            if "past" in item.name:
                step_inputs[item.name] = first_cache[item.name.replace("past_key_values", "present")]
        outputs = session.run(None, step_inputs)
        assert all(np.isfinite(output).all() for output in outputs)
        first_cache = {item.name: value for item, value in zip(session.get_outputs(), outputs, strict=True) if "present" in item.name}
        assert {value.shape[2] for value in first_cache.values()} == {expected_length}


def _build(source, root, name: str, fused: bool):
    output = root / name
    process = subprocess.run(
        [sys.executable, "-m", "onnxruntime_genai.models.builder", *builder_arguments(source, output, root / f"{name}-cache", num_hidden_layers=2, fuse_qk_norm_gqa=fused)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr or process.stdout
    model = output / "global_decoder.onnx"
    rewrite_attention_mask_for_gqa(model, batch_size=2)
    return model
