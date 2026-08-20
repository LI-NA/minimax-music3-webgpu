"""Reduced semantic-token and end-token output head export."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from safetensors import safe_open
import torch

from .constants import (
    ARTIFACT_FILE_LIMIT,
    AUDIO_END_TOKEN_ID,
    HIDDEN_SIZE,
    SEMANTIC_TOKEN_COUNT,
    SEMANTIC_TOKEN_START,
)
from .external_data import RepackedModel, repack_external_data


def select_reduced_head_rows(
    weights: np.ndarray,
    *,
    semantic_start: int,
    semantic_count: int,
    end_token_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.ascontiguousarray(weights[semantic_start : semantic_start + semantic_count]),
        np.ascontiguousarray(weights[end_token_id : end_token_id + 1]),
    )


def export_reduced_head(source_shard: Path, output_dir: Path) -> RepackedModel:
    with safe_open(source_shard, framework="pt", device="cpu") as source:
        tensor = source.get_slice("lm_head.weight")
        shape = tensor.get_shape()
        if len(shape) != 2 or shape[1] != HIDDEN_SIZE:
            raise ValueError("lm_head.weight has an unexpected shape")
        semantic = tensor[
            SEMANTIC_TOKEN_START : SEMANTIC_TOKEN_START + SEMANTIC_TOKEN_COUNT
        ].to(dtype=torch.float16).numpy()
        end = tensor[AUDIO_END_TOKEN_ID : AUDIO_END_TOKEN_ID + 1].to(
            dtype=torch.float16
        ).numpy()

    with TemporaryDirectory() as temporary_directory:
        model_path = Path(temporary_directory) / "reduced-head.onnx"
        onnx.save_model(
            _reduced_head_model(semantic, end),
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="reduced-head-source.bin",
            size_threshold=0,
        )
        return repack_external_data(model_path, output_dir, ARTIFACT_FILE_LIMIT)


def _reduced_head_model(semantic: np.ndarray, end: np.ndarray) -> onnx.ModelProto:
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Gather", ["hidden_states", "last_index"], ["last_state"], axis=1),
                helper.make_node("Flatten", ["last_state"], ["final_state"], axis=1),
                helper.make_node("MatMul", ["final_state", "semantic_weight"], ["semantic_logits"]),
                helper.make_node("MatMul", ["final_state", "end_weight"], ["end_logit"]),
            ],
            "reduced_head",
            [helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT16, [None, None, HIDDEN_SIZE])],
            [
                helper.make_tensor_value_info("semantic_logits", TensorProto.FLOAT16, [None, SEMANTIC_TOKEN_COUNT]),
                helper.make_tensor_value_info("end_logit", TensorProto.FLOAT16, [None, 1]),
            ],
            initializer=[
                numpy_helper.from_array(np.array([-1], dtype=np.int64), "last_index"),
                numpy_helper.from_array(np.ascontiguousarray(semantic.T), "semantic_weight"),
                numpy_helper.from_array(np.ascontiguousarray(end.T), "end_weight"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
