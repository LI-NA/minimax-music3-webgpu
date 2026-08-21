"""Fixed-shape MiniMax Music 3 condition-encoder export."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from safetensors import safe_open

from .constants import ARTIFACT_FILE_LIMIT
from .external_data import RepackedModel, repack_external_data


_WEIGHT_KEYS = {
    "layer_scale",
    "layer_weight_logits",
    "proj.bias",
    "proj.weight",
}


@dataclass(frozen=True)
class ConditionEncoderWeights:
    layer_weight_logits: np.ndarray
    layer_scale: np.ndarray
    proj_weight: np.ndarray
    proj_bias: np.ndarray


def load_condition_encoder_weights(
    source: Path,
    *,
    condition_hidden_dim: int = 4096,
    num_condition_layers: int = 8,
    out_dim: int = 2048,
) -> ConditionEncoderWeights:
    with safe_open(source, framework="np") as checkpoint:
        if set(checkpoint.keys()) != _WEIGHT_KEYS:
            raise ValueError("checkpoint must contain exactly the four condition encoder tensors")
        tensors = {name: checkpoint.get_tensor(name) for name in _WEIGHT_KEYS}
    expected_shapes = {
        "layer_weight_logits": (num_condition_layers,),
        "layer_scale": (1,),
        "proj.weight": (out_dim, condition_hidden_dim, 3),
        "proj.bias": (out_dim,),
    }
    for name, shape in expected_shapes.items():
        if tensors[name].shape != shape or tensors[name].dtype != np.float32:
            raise ValueError(f"{name} must be F32 with shape {shape}")
    return ConditionEncoderWeights(
        layer_weight_logits=tensors["layer_weight_logits"],
        layer_scale=tensors["layer_scale"],
        proj_weight=tensors["proj.weight"],
        proj_bias=tensors["proj.bias"],
    )


def nearest_indices(input_length: int, output_length: int) -> np.ndarray:
    if input_length <= 0 or output_length <= 0:
        raise ValueError("nearest-neighbor lengths must be positive")
    scale = np.float32(input_length / output_length)
    return np.floor(np.arange(output_length, dtype=np.float32) * scale).astype(np.int64)


def condition_encoder_model(
    weights: ConditionEncoderWeights,
    *,
    frame_count: int = 125,
    latent_length: int = 430,
    maximum_window: bool = False,
) -> onnx.ModelProto:
    if weights.proj_weight.ndim != 3 or weights.proj_weight.shape[2] != 3:
        raise ValueError("condition projection must be a width-three Conv1D kernel")
    num_layers = weights.layer_weight_logits.shape[0]
    output_dim, hidden_dim, _ = weights.proj_weight.shape
    if (
        weights.layer_scale.shape != (1,)
        or weights.proj_bias.shape != (output_dim,)
        or frame_count <= 0
        or latent_length <= 0
    ):
        raise ValueError("condition encoder weights or fixed lengths are invalid")
    if maximum_window and (frame_count, latent_length) != (200, 689):
        raise ValueError("maximum condition window must be 200 frames and 689 latents")

    logits = weights.layer_weight_logits.astype(np.float32)
    exponents = np.exp(logits - logits.max())
    layer_weights = (exponents / exponents.sum()).astype(np.float16).reshape(1, 1, num_layers, 1)
    initializers = [
        numpy_helper.from_array(
            np.array([1, frame_count, num_layers, hidden_dim], dtype=np.int64),
            "grouped_shape",
        ),
        numpy_helper.from_array(layer_weights, "layer_weights"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "layer_axis"),
        numpy_helper.from_array(weights.layer_scale.astype(np.float16), "layer_scale"),
        numpy_helper.from_array(weights.proj_weight.astype(np.float16), "proj_weight"),
        numpy_helper.from_array(weights.proj_bias.astype(np.float16), "proj_bias"),
    ]
    if not maximum_window:
        initializers.append(
            numpy_helper.from_array(nearest_indices(frame_count, latent_length), "nearest_index")
        )
    resampled_output = "resampled_condition" if maximum_window else "condition"
    nodes = [
        helper.make_node("Reshape", ["frame_hiddens", "grouped_shape"], ["grouped_hiddens"]),
        helper.make_node("Mul", ["grouped_hiddens", "layer_weights"], ["weighted_hiddens"]),
        helper.make_node(
            "ReduceSum",
            ["weighted_hiddens", "layer_axis"],
            ["mixed_hiddens"],
            keepdims=0,
        ),
        helper.make_node("Mul", ["mixed_hiddens", "layer_scale"], ["scaled_hiddens"]),
        helper.make_node("Transpose", ["scaled_hiddens"], ["channel_hiddens"], perm=[0, 2, 1]),
        helper.make_node(
            "Conv",
            ["channel_hiddens", "proj_weight", "proj_bias"],
            ["projected_hiddens"],
            kernel_shape=[3],
            pads=[1, 1],
        ),
        helper.make_node("Gather", ["projected_hiddens", "nearest_index"], ["resampled_hiddens"], axis=2),
        helper.make_node("Transpose", ["resampled_hiddens"], [resampled_output], perm=[0, 2, 1]),
    ]
    if maximum_window:
        nodes.append(
            helper.make_node(
                "Mul",
                ["resampled_condition", "active_latent_mask"],
                ["condition"],
            )
        )
    inputs = [
        helper.make_tensor_value_info(
            "frame_hiddens",
            TensorProto.FLOAT16,
            [1, frame_count, num_layers * hidden_dim],
        )
    ]
    if maximum_window:
        inputs.extend(
            [
                helper.make_tensor_value_info(
                    "nearest_index", TensorProto.INT64, [latent_length]
                ),
                helper.make_tensor_value_info(
                    "active_latent_mask", TensorProto.FLOAT16, [1, latent_length, 1]
                ),
            ]
        )
    model = helper.make_model(
        helper.make_graph(
            nodes,
            f"minimax_music3_condition_{frame_count}",
            inputs,
            [
                helper.make_tensor_value_info(
                    "condition",
                    TensorProto.FLOAT16,
                    [1, latent_length, output_dim],
                )
            ],
            initializer=initializers,
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    onnx.checker.check_model(model)
    return model


def export_condition_encoder(
    source: Path,
    output_dir: Path,
    *,
    condition_hidden_dim: int = 4096,
    num_condition_layers: int = 8,
    out_dim: int = 2048,
    frame_count: int = 125,
    latent_length: int = 430,
    maximum_window: bool = False,
    max_file_bytes: int = ARTIFACT_FILE_LIMIT,
) -> RepackedModel:
    weights = load_condition_encoder_weights(
        source,
        condition_hidden_dim=condition_hidden_dim,
        num_condition_layers=num_condition_layers,
        out_dim=out_dim,
    )
    model = condition_encoder_model(
        weights,
        frame_count=frame_count,
        latent_length=latent_length,
        maximum_window=maximum_window,
    )
    with TemporaryDirectory() as temporary_directory:
        model_path = Path(temporary_directory) / "condition-125.onnx"
        onnx.save_model(model, model_path)
        return repack_external_data(
            model_path,
            output_dir,
            max_file_bytes,
            inline_threshold=32,
        )
