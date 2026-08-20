"""Fixed-shape MiniMax Music 3 flow transformer conversion contracts."""

from dataclasses import dataclass
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping
import uuid

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.capi._pybind_state import quantize_matmul_4bits
from safetensors import safe_open
import torch
import torch.nn.functional as torch_functional


@dataclass(frozen=True)
class FlowTransformerConfig:
    in_channels: int = 128
    condition_dim: int = 2048
    num_layers: int = 36
    num_attention_heads: int = 32
    attention_head_dim: int = 64
    ff_inner_dim: int = 8192
    rotary_dim: int = 32
    fourier_embedding_dim: int = 256

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if any(value < 1 for value in values):
            raise ValueError("flow transformer dimensions must be positive")
        if self.rotary_dim % 2 or self.rotary_dim > self.attention_head_dim:
            raise ValueError("rotary_dim must be even and no larger than attention_head_dim")
        if self.fourier_embedding_dim % 2:
            raise ValueError("fourier_embedding_dim must be even")

    @property
    def hidden_size(self) -> int:
        return self.num_attention_heads * self.attention_head_dim


@dataclass(frozen=True)
class FlowSchedule:
    timesteps: np.ndarray
    dts: np.ndarray


@dataclass(frozen=True)
class FlowGraphReport:
    matmul_nbits_nodes: int
    dynamic_shape_ops: tuple[str, ...]
    external_locations: tuple[str, ...]


@dataclass(frozen=True)
class FlowSourceReport:
    config: FlowTransformerConfig
    tensor_count: int
    total_parameters: int
    shards: tuple[str, ...]


class FlowSafetensorState(Mapping[str, torch.Tensor]):
    """Lazy checkpoint mapping that materializes at most one requested tensor."""

    def __init__(self, source_dir: str | Path):
        self.source = Path(source_dir).resolve()
        raw_config = json.loads((self.source / "config.json").read_text(encoding="utf-8"))
        fields = FlowTransformerConfig.__dataclass_fields__
        self.config = FlowTransformerConfig(**{name: raw_config[name] for name in fields})
        index = json.loads(
            (self.source / "diffusion_pytorch_model.safetensors.index.json").read_text(encoding="utf-8")
        )
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("flow safetensor index has no weight_map")
        self.weight_map: dict[str, str] = weight_map
        self._stack: ExitStack | None = None
        self._handles = {}
        self.shapes: dict[str, tuple[int, ...]] = {}

    def __enter__(self):
        self._stack = ExitStack()
        for shard_name in sorted(set(self.weight_map.values())):
            shard = (self.source / shard_name).resolve()
            if not shard.is_relative_to(self.source) or not shard.is_file():
                raise ValueError(f"flow shard path is invalid: {shard_name}")
            self._handles[shard_name] = self._stack.enter_context(
                safe_open(shard, framework="pt", device="cpu")
            )
        self.shapes = {
            key: tuple(self._handles[shard].get_slice(key).get_shape())
            for key, shard in self.weight_map.items()
        }
        validate_flow_metadata(self.config, self.shapes)
        return self

    def __exit__(self, exception_type, exception, traceback):
        if self._stack is not None:
            self._stack.close()
        self._stack = None
        self._handles = {}

    def __getitem__(self, key: str) -> torch.Tensor:
        if not self._handles:
            raise RuntimeError("flow safetensor state is not open")
        return self._handles[self.weight_map[key]].get_tensor(key)

    def __iter__(self):
        return iter(self.weight_map)

    def __len__(self) -> int:
        return len(self.weight_map)


def open_flow_state(source_dir: str | Path) -> FlowSafetensorState:
    return FlowSafetensorState(source_dir)


def exact_flow_schedule(num_steps: int = 30) -> FlowSchedule:
    if num_steps != 30:
        raise ValueError("the fixed flow graph requires exactly 30 steps")
    source_sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps).astype(np.float32)
    timesteps = (np.float32(1.0) - source_sigmas).astype(np.float32)
    sigmas = np.concatenate((timesteps, np.ones(1, dtype=np.float32)))
    return FlowSchedule(timesteps, (sigmas[1:] - sigmas[:-1]).astype(np.float32))


def fixed_rope_tables(sequence_length: int, rotary_dim: int) -> tuple[np.ndarray, np.ndarray]:
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if rotary_dim < 2 or rotary_dim % 2:
        raise ValueError("rotary_dim must be positive and even")
    steps = torch.arange(sequence_length, dtype=torch.float32)
    inverse = 1.0 / (
        10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    frequencies = torch.outer(steps, inverse)
    frequencies = torch.cat((frequencies, frequencies), dim=-1)
    cosine = frequencies.cos().to(torch.float16).numpy()[None, :, None, :]
    sine = frequencies.sin().to(torch.float16).numpy()[None, :, None, :]
    return np.ascontiguousarray(cosine), np.ascontiguousarray(sine)


def expected_flow_shapes(config: FlowTransformerConfig) -> dict[str, tuple[int, ...]]:
    hidden = config.hidden_size
    concat = 2 * config.in_channels + config.condition_dim
    shapes: dict[str, tuple[int, ...]] = {
        "time_proj.weight": (config.fourier_embedding_dim // 2, 1),
        "time_embed.linear_1.weight": (hidden, config.fourier_embedding_dim),
        "time_embed.linear_1.bias": (hidden,),
        "time_embed.linear_2.weight": (hidden, hidden),
        "time_embed.linear_2.bias": (hidden,),
        "preprocess_conv.weight": (concat, concat, 1),
        "proj_in.weight": (hidden, concat),
        "proj_out.weight": (config.in_channels, hidden),
        "postprocess_conv.weight": (config.in_channels, config.in_channels, 1),
    }
    for index in range(config.num_layers):
        prefix = f"transformer_blocks.{index}"
        shapes.update(
            {
                f"{prefix}.norm1.weight": (hidden,),
                f"{prefix}.norm1.bias": (hidden,),
                f"{prefix}.attn.to_q.weight": (hidden, hidden),
                f"{prefix}.attn.to_k.weight": (hidden, hidden),
                f"{prefix}.attn.to_v.weight": (hidden, hidden),
                f"{prefix}.attn.to_out.0.weight": (hidden, hidden),
                f"{prefix}.norm2.weight": (hidden,),
                f"{prefix}.norm2.bias": (hidden,),
                f"{prefix}.ff_in.weight": (2 * config.ff_inner_dim, hidden),
                f"{prefix}.ff_in.bias": (2 * config.ff_inner_dim,),
                f"{prefix}.ff_out.weight": (hidden, config.ff_inner_dim),
                f"{prefix}.ff_out.bias": (hidden,),
            }
        )
    return shapes


def validate_flow_metadata(
    config: FlowTransformerConfig,
    shapes: Mapping[str, tuple[int, ...]],
) -> None:
    expected = expected_flow_shapes(config)
    actual = dict(shapes)
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    wrong = sorted(key for key in expected.keys() & actual.keys() if expected[key] != actual[key])
    if missing or unexpected or wrong:
        raise ValueError(
            f"flow metadata mismatch: missing={missing}, unexpected={unexpected}, wrong_shapes={wrong}"
        )


def inspect_flow_source(source_dir: str | Path) -> FlowSourceReport:
    source = Path(source_dir).resolve()
    raw_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    fields = FlowTransformerConfig.__dataclass_fields__
    config = FlowTransformerConfig(**{name: raw_config[name] for name in fields})
    index = json.loads(
        (source / "diffusion_pytorch_model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("flow safetensor index has no weight_map")
    shard_names = tuple(sorted(set(weight_map.values())))
    shapes: dict[str, tuple[int, ...]] = {}
    for shard_name in shard_names:
        shard = (source / shard_name).resolve()
        if not shard.is_relative_to(source) or not shard.is_file():
            raise ValueError(f"flow shard path is invalid: {shard_name}")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in shapes or weight_map.get(key) != shard_name:
                    raise ValueError(f"flow shard index mismatch for {key}")
                shapes[key] = tuple(handle.get_slice(key).get_shape())
    if set(shapes) != set(weight_map):
        raise ValueError("flow shard index keys do not match shard metadata")
    validate_flow_metadata(config, shapes)
    return FlowSourceReport(
        config,
        len(shapes),
        sum(int(np.prod(shape, dtype=np.int64)) for shape in shapes.values()),
        shard_names,
    )


class _InitializerWriter:
    def __init__(self, model_path: Path, external: bool, max_file_bytes: int):
        self.model_path = model_path
        self.external = external
        self.max_file_bytes = max_file_bytes
        self.initializers: list[TensorProto] = []
        self.locations: list[str] = []
        self.inline_threshold = 64
        self._file = None
        self._size = 0

    def add(self, name: str, value: np.ndarray) -> str:
        array = np.ascontiguousarray(value)
        if not self.external or array.nbytes <= self.inline_threshold:
            self.initializers.append(numpy_helper.from_array(array, name))
            return name
        if array.nbytes > self.max_file_bytes:
            raise ValueError(f"initializer {name} exceeds the artifact file limit")
        if self._file is None or self._size + array.nbytes > self.max_file_bytes:
            self._open_shard()
        offset = self._size
        self._file.write(array.tobytes(order="C"))
        self._size += array.nbytes
        tensor = TensorProto(name=name, data_type=helper.np_dtype_to_tensor_dtype(array.dtype), dims=array.shape)
        tensor.data_location = TensorProto.EXTERNAL
        for key, raw in (("location", self.locations[-1]), ("offset", offset), ("length", array.nbytes)):
            entry = tensor.external_data.add()
            entry.key = key
            entry.value = str(raw)
        self.initializers.append(tensor)
        return name

    def _open_shard(self) -> None:
        if self._file is not None:
            self._file.close()
        location = f"{self.model_path.stem}.data-{len(self.locations):03d}.bin"
        self.locations.append(location)
        self._file = (self.model_path.parent / location).open("wb")
        self._size = 0

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class _FlowGraphBuilder:
    def __init__(
        self,
        state: Mapping[str, torch.Tensor],
        output: Path,
        config: FlowTransformerConfig,
        latent_length: int,
        quantize: bool,
        external_data: bool,
        max_file_bytes: int,
    ):
        self.state = state
        self.output = output
        self.config = config
        self.latent_length = latent_length
        self.quantize = quantize
        self.nodes: list[onnx.NodeProto] = []
        self.writer = _InitializerWriter(output, external_data, max_file_bytes)
        self._constant_index = 0

    def constant(self, value, dtype=None, prefix="constant") -> str:
        array = np.asarray(value, dtype=dtype)
        name = f"{prefix}.{self._constant_index}"
        self._constant_index += 1
        return self.writer.add(name, array)

    def tensor(self, key: str, *, transpose: bool = False) -> np.ndarray:
        value = self.state[key].detach().cpu().to(torch.float16).numpy()
        return np.ascontiguousarray(value.T if transpose else value)

    def linear(self, source: str, key: str, output: str, *, use_q4: bool = True) -> str:
        weight = self.tensor(key, transpose=True)
        if self.quantize and use_q4:
            rows, columns = weight.shape
            blocks = (rows + 127) // 128
            packed = np.zeros((columns, blocks, 64), dtype=np.uint8)
            scales = np.zeros((columns, blocks), dtype=np.float16)
            zero_points = np.zeros((columns, (blocks + 1) // 2), dtype=np.uint8)
            quantize_matmul_4bits(packed, weight, scales, zero_points, 128, columns, rows, True)
            packed_name = self.writer.add(f"{key}.q4", packed)
            scale_name = self.writer.add(f"{key}.scales", scales)
            self.nodes.append(
                helper.make_node(
                    "MatMulNBits",
                    [source, packed_name, scale_name],
                    [output],
                    name=f"{key}.MatMulNBits",
                    domain="com.microsoft",
                    K=rows,
                    N=columns,
                    bits=4,
                    block_size=128,
                    accuracy_level=4,
                )
            )
        else:
            weight_name = self.writer.add(key, weight)
            self.nodes.append(helper.make_node("MatMul", [source, weight_name], [output], name=f"{key}.MatMul"))
        return output

    def add_bias(self, source: str, key: str, output: str) -> str:
        self.nodes.append(helper.make_node("Add", [source, self.writer.add(key, self.tensor(key))], [output]))
        return output

    def layer_norm(self, source: str, prefix: str, output: str) -> str:
        scale = self.writer.add(f"{prefix}.weight", self.tensor(f"{prefix}.weight"))
        bias = self.writer.add(f"{prefix}.bias", self.tensor(f"{prefix}.bias"))
        self.nodes.append(helper.make_node("LayerNormalization", [source, scale, bias], [output], axis=-1, epsilon=1e-5))
        return output

    def apply_rope(self, source: str, prefix: str, cos: str, sin: str) -> str:
        axes = self.constant([-1], np.int64, "rope.axes")
        steps = self.constant([1], np.int64, "rope.steps")
        leading = f"{prefix}.leading"
        trailing = f"{prefix}.trailing"
        self.nodes.extend(
            [
                helper.make_node(
                    "Slice",
                    [source, self.constant([0], np.int64), self.constant([self.config.rotary_dim], np.int64), axes, steps],
                    [leading],
                ),
                helper.make_node(
                    "Slice",
                    [source, self.constant([self.config.rotary_dim], np.int64), self.constant([self.config.attention_head_dim], np.int64), axes, steps],
                    [trailing],
                ),
                helper.make_node(
                    "Split",
                    [leading, self.constant([self.config.rotary_dim // 2] * 2, np.int64)],
                    [f"{prefix}.first", f"{prefix}.second"],
                    axis=-1,
                ),
                helper.make_node("Neg", [f"{prefix}.second"], [f"{prefix}.negative_second"]),
                helper.make_node(
                    "Concat",
                    [f"{prefix}.negative_second", f"{prefix}.first"],
                    [f"{prefix}.rotate_half"],
                    axis=-1,
                ),
                helper.make_node("Mul", [leading, cos], [f"{prefix}.cosine"]),
                helper.make_node("Mul", [f"{prefix}.rotate_half", sin], [f"{prefix}.sine"]),
                helper.make_node("Add", [f"{prefix}.cosine", f"{prefix}.sine"], [f"{prefix}.rotated"]),
                helper.make_node("Concat", [f"{prefix}.rotated", trailing], [f"{prefix}.output"], axis=-1),
            ]
        )
        return f"{prefix}.output"

    def build(self) -> Path:
        if self.latent_length < 1:
            raise ValueError("latent_length must be positive")
        source_shapes = getattr(self.state, "shapes", None)
        validate_flow_metadata(
            self.config,
            source_shapes or {key: tuple(value.shape) for key, value in self.state.items()},
        )
        batch = 2
        sequence = self.latent_length + 1
        hidden = self.config.hidden_size
        concat_channels = 2 * self.config.in_channels + self.config.condition_dim
        axes_one = self.constant([1], np.int64, "axes.one")
        zero = self.constant(np.array(0, dtype=np.float16), prefix="zero")

        self.nodes.extend(
            [
                helper.make_node("Concat", ["latents", "latents"], ["latent_batch"], axis=0),
                helper.make_node("Mul", ["condition", zero], ["zero_condition"]),
                helper.make_node("Concat", ["condition", "zero_condition"], ["condition_batch"], axis=0),
                helper.make_node("Concat", ["timestep", "timestep"], ["timestep_batch"], axis=0),
                helper.make_node("Mul", ["latent_batch", zero], ["zero_latents"]),
                helper.make_node("Transpose", ["condition_batch"], ["condition_channels"], perm=[0, 2, 1]),
                helper.make_node(
                    "Concat", ["latent_batch", "zero_latents", "condition_channels"], ["transformer_input"], axis=1
                ),
                helper.make_node(
                    "Conv",
                    ["transformer_input", self.writer.add("preprocess_conv.weight", self.tensor("preprocess_conv.weight"))],
                    ["preprocessed"],
                    kernel_shape=[1],
                    pads=[0, 0],
                    strides=[1],
                ),
                helper.make_node("Add", ["preprocessed", "transformer_input"], ["preprocess_residual"]),
                helper.make_node("Transpose", ["preprocess_residual"], ["project_input"], perm=[0, 2, 1]),
            ]
        )
        projected = self.linear("project_input", "proj_in.weight", "projected")

        self.nodes.append(helper.make_node("Unsqueeze", ["timestep_batch", axes_one], ["timestep_column"]))
        angles = self.linear("timestep_column", "time_proj.weight", "time_angles_unscaled", use_q4=False)
        two_pi = self.constant(np.array(2 * np.pi, dtype=np.float16), prefix="two_pi")
        self.nodes.extend(
            [
                helper.make_node("Mul", [angles, two_pi], ["time_angles"]),
                helper.make_node("Cos", ["time_angles"], ["time_cos"]),
                helper.make_node("Sin", ["time_angles"], ["time_sin"]),
                helper.make_node("Concat", ["time_cos", "time_sin"], ["fourier_time"], axis=-1),
            ]
        )
        time_hidden = self.linear("fourier_time", "time_embed.linear_1.weight", "time_hidden_linear")
        time_hidden = self.add_bias(time_hidden, "time_embed.linear_1.bias", "time_hidden_bias")
        self.nodes.extend(
            [
                helper.make_node("Sigmoid", [time_hidden], ["time_hidden_sigmoid"]),
                helper.make_node("Mul", [time_hidden, "time_hidden_sigmoid"], ["time_hidden_silu"]),
            ]
        )
        time_output = self.linear("time_hidden_silu", "time_embed.linear_2.weight", "time_output_linear")
        time_output = self.add_bias(time_output, "time_embed.linear_2.bias", "time_output")
        self.nodes.extend(
            [
                helper.make_node("Unsqueeze", [time_output, axes_one], ["time_token"]),
                helper.make_node("Concat", ["time_token", projected], ["block.0.input"], axis=1),
            ]
        )

        cosine, sine = fixed_rope_tables(sequence, self.config.rotary_dim)
        cos = self.writer.add("rotary_cos", cosine)
        sin = self.writer.add("rotary_sin", sine)
        attention_scale = self.constant(np.array(self.config.attention_head_dim**-0.5, dtype=np.float16), prefix="attention_scale")
        qkv_shape = self.constant([batch, sequence, self.config.num_attention_heads, self.config.attention_head_dim], np.int64, "qkv_shape")
        flat_shape = self.constant([batch, sequence, hidden], np.int64, "flat_shape")
        split_ff = self.constant([self.config.ff_inner_dim, self.config.ff_inner_dim], np.int64, "ff_split")
        block_input = "block.0.input"
        for index in range(self.config.num_layers):
            prefix = f"transformer_blocks.{index}"
            normalized = self.layer_norm(block_input, f"{prefix}.norm1", f"{prefix}.norm1.output")
            projections = {}
            for part in ("q", "k", "v"):
                flat = self.linear(normalized, f"{prefix}.attn.to_{part}.weight", f"{prefix}.{part}.flat")
                shaped = f"{prefix}.{part}.shaped"
                self.nodes.append(helper.make_node("Reshape", [flat, qkv_shape], [shaped]))
                projections[part] = self.apply_rope(shaped, f"{prefix}.{part}.rope", cos, sin) if part != "v" else shaped
            for part in projections:
                self.nodes.append(
                    helper.make_node("Transpose", [projections[part]], [f"{prefix}.{part}.heads"], perm=[0, 2, 1, 3])
                )
            self.nodes.extend(
                [
                    helper.make_node("Transpose", [f"{prefix}.k.heads"], [f"{prefix}.k.transposed"], perm=[0, 1, 3, 2]),
                    helper.make_node("MatMul", [f"{prefix}.q.heads", f"{prefix}.k.transposed"], [f"{prefix}.scores"]),
                    helper.make_node("Mul", [f"{prefix}.scores", attention_scale], [f"{prefix}.scaled_scores"]),
                    helper.make_node("Softmax", [f"{prefix}.scaled_scores"], [f"{prefix}.probabilities"], axis=-1),
                    helper.make_node("MatMul", [f"{prefix}.probabilities", f"{prefix}.v.heads"], [f"{prefix}.context"]),
                    helper.make_node("Transpose", [f"{prefix}.context"], [f"{prefix}.context_transposed"], perm=[0, 2, 1, 3]),
                    helper.make_node("Reshape", [f"{prefix}.context_transposed", flat_shape], [f"{prefix}.context_flat"]),
                ]
            )
            attention = self.linear(
                f"{prefix}.context_flat", f"{prefix}.attn.to_out.0.weight", f"{prefix}.attention"
            )
            self.nodes.append(helper.make_node("Add", [block_input, attention], [f"{prefix}.attention_residual"]))
            normalized = self.layer_norm(
                f"{prefix}.attention_residual", f"{prefix}.norm2", f"{prefix}.norm2.output"
            )
            ff = self.linear(normalized, f"{prefix}.ff_in.weight", f"{prefix}.ff_in.linear")
            ff = self.add_bias(ff, f"{prefix}.ff_in.bias", f"{prefix}.ff_in.output")
            self.nodes.extend(
                [
                    helper.make_node("Split", [ff, split_ff], [f"{prefix}.gate_states", f"{prefix}.gate"], axis=-1),
                    helper.make_node("Sigmoid", [f"{prefix}.gate"], [f"{prefix}.gate_sigmoid"]),
                    helper.make_node("Mul", [f"{prefix}.gate", f"{prefix}.gate_sigmoid"], [f"{prefix}.gate_silu"]),
                    helper.make_node("Mul", [f"{prefix}.gate_states", f"{prefix}.gate_silu"], [f"{prefix}.gated"]),
                ]
            )
            ff_output = self.linear(f"{prefix}.gated", f"{prefix}.ff_out.weight", f"{prefix}.ff_out.linear")
            ff_output = self.add_bias(ff_output, f"{prefix}.ff_out.bias", f"{prefix}.ff_out.output")
            block_input = f"block.{index + 1}.input"
            self.nodes.append(helper.make_node("Add", [f"{prefix}.attention_residual", ff_output], [block_input]))

        self.nodes.append(
            helper.make_node(
                "Slice",
                [block_input, self.constant([1], np.int64), self.constant([sequence], np.int64), axes_one, self.constant([1], np.int64)],
                ["without_time_token"],
            )
        )
        projected_output = self.linear("without_time_token", "proj_out.weight", "projected_output")
        self.nodes.extend(
            [
                helper.make_node("Transpose", [projected_output], ["output_channels"], perm=[0, 2, 1]),
                helper.make_node(
                    "Conv",
                    ["output_channels", self.writer.add("postprocess_conv.weight", self.tensor("postprocess_conv.weight"))],
                    ["postprocessed"],
                    kernel_shape=[1],
                    pads=[0, 0],
                    strides=[1],
                ),
                helper.make_node("Add", ["postprocessed", "output_channels"], ["velocity"]),
            ]
        )
        batch_axis = self.constant([0], np.int64, "batch_axis")
        batch_step = self.constant([1], np.int64, "batch_step")
        self.nodes.extend(
            [
                helper.make_node("Slice", ["velocity", self.constant([0], np.int64), self.constant([1], np.int64), batch_axis, batch_step], ["conditional_velocity"]),
                helper.make_node("Slice", ["velocity", self.constant([1], np.int64), self.constant([2], np.int64), batch_axis, batch_step], ["unconditional_velocity"]),
                helper.make_node("Sub", ["conditional_velocity", "unconditional_velocity"], ["guidance_delta"]),
                helper.make_node("Mul", ["guidance_delta", self.constant(np.array(1.7, dtype=np.float16), prefix="guidance")], ["scaled_guidance"]),
                helper.make_node("Add", ["unconditional_velocity", "scaled_guidance"], ["guided_velocity"]),
                helper.make_node("Cast", ["latents"], ["latents_fp32"], to=TensorProto.FLOAT),
                helper.make_node("Cast", ["guided_velocity"], ["guided_velocity_fp32"], to=TensorProto.FLOAT),
                helper.make_node("Unsqueeze", ["dt", self.constant([1, 2], np.int64, "dt_axes")], ["dt_broadcast"]),
                helper.make_node("Mul", ["guided_velocity_fp32", "dt_broadcast"], ["euler_delta"]),
                helper.make_node("Add", ["latents_fp32", "euler_delta"], ["next_latents_fp32"]),
                helper.make_node("Cast", ["next_latents_fp32"], ["next_latents"], to=TensorProto.FLOAT16),
            ]
        )
        graph = helper.make_graph(
            self.nodes,
            f"minimax_music3_flow_step_{self.latent_length}",
            [
                helper.make_tensor_value_info("latents", TensorProto.FLOAT16, [1, self.config.in_channels, self.latent_length]),
                helper.make_tensor_value_info("condition", TensorProto.FLOAT16, [1, self.latent_length, self.config.condition_dim]),
                helper.make_tensor_value_info("timestep", TensorProto.FLOAT16, [1]),
                helper.make_tensor_value_info("dt", TensorProto.FLOAT, [1]),
            ],
            [helper.make_tensor_value_info("next_latents", TensorProto.FLOAT16, [1, self.config.in_channels, self.latent_length])],
            self.writer.initializers,
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
        )
        model.ir_version = 10
        self.writer.close()
        if not self.writer.external:
            onnx.checker.check_model(model)
        onnx.save_model(model, self.output)
        onnx.checker.check_model(self.output)
        return self.output


def export_flow_step(
    state_dict: Mapping[str, torch.Tensor],
    output_path: str | Path,
    *,
    config: FlowTransformerConfig = FlowTransformerConfig(),
    latent_length: int = 430,
    quantize: bool = True,
    external_data: bool = False,
    max_file_bytes: int = 128 * 1024 * 1024,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    old_external = _referenced_external_files(output)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    generation = uuid.uuid4().hex
    staged_output = staging / f"{output.stem}.{generation}{output.suffix}"
    builder = _FlowGraphBuilder(
        state_dict,
        staged_output,
        config,
        latent_length,
        quantize,
        external_data,
        max_file_bytes,
    )
    moved_external: list[Path] = []
    published = False
    try:
        builder.build()
        report = validate_flow_graph(
            staged_output,
            config=config,
            latent_length=latent_length,
            quantized=quantize,
        )
        for location in report.external_locations:
            source = (staging / location).resolve()
            if not source.is_relative_to(staging.resolve()) or not source.is_file():
                raise ValueError(f"flow staged external path is invalid: {location}")
            destination = output.parent / location
            os.replace(source, destination)
            moved_external.append(destination)
        os.replace(staged_output, output)
        published = True
        for old_file in old_external:
            if old_file not in moved_external:
                try:
                    old_file.unlink(missing_ok=True)
                except OSError:
                    pass
        return output
    except Exception:
        if not published:
            for moved in moved_external:
                moved.unlink(missing_ok=True)
        raise
    finally:
        builder.writer.close()
        shutil.rmtree(staging, ignore_errors=True)


def _referenced_external_files(model_path: Path) -> tuple[Path, ...]:
    if not model_path.is_file():
        return ()
    try:
        model = onnx.load_model(model_path, load_external_data=False)
    except Exception:
        return ()
    parent = model_path.parent.resolve()
    files = set()
    for tensor in model.graph.initializer:
        fields = {item.key: item.value for item in tensor.external_data}
        location = fields.get("location")
        if not location:
            continue
        candidate = (parent / location).resolve()
        if candidate.is_relative_to(parent) and candidate.is_file():
            files.add(candidate)
    return tuple(sorted(files))


def validate_flow_graph(
    model_path: str | Path,
    *,
    config: FlowTransformerConfig = FlowTransformerConfig(),
    latent_length: int = 430,
    quantized: bool = True,
) -> FlowGraphReport:
    path = Path(model_path)
    model = onnx.load_model(path, load_external_data=False)
    inputs = {item.name: item.type.tensor_type for item in model.graph.input}
    outputs = {item.name: item.type.tensor_type for item in model.graph.output}
    contracts = {
        "latents": (TensorProto.FLOAT16, [1, config.in_channels, latent_length]),
        "condition": (TensorProto.FLOAT16, [1, latent_length, config.condition_dim]),
        "timestep": (TensorProto.FLOAT16, [1]),
        "dt": (TensorProto.FLOAT, [1]),
        "next_latents": (TensorProto.FLOAT16, [1, config.in_channels, latent_length]),
    }
    for name, (dtype, shape) in contracts.items():
        value = inputs.get(name) or outputs.get(name)
        if value is None or value.elem_type != dtype or [dim.dim_value for dim in value.shape.dim] != shape:
            raise ValueError(f"flow graph value {name} has an invalid type or shape")
    dynamic = tuple(
        sorted({node.op_type for node in model.graph.node if node.op_type in {"Shape", "Range", "ConstantOfShape"}})
    )
    q4_nodes = [node for node in model.graph.node if node.op_type == "MatMulNBits"]
    expected_q4 = config.num_layers * 6 + 4 if quantized else 0
    if dynamic or len(q4_nodes) != expected_q4:
        raise ValueError("flow graph topology does not match the fixed contract")
    for node in q4_nodes:
        attributes = {item.name: helper.get_attribute_value(item) for item in node.attribute}
        if (
            len(node.input) != 3
            or attributes.get("bits") != 4
            or attributes.get("block_size") != 128
            or attributes.get("accuracy_level") != 4
        ):
            raise ValueError("flow MatMulNBits node has an invalid q4 contract")
    locations = set()
    for tensor in model.graph.initializer:
        fields = {item.key: item.value for item in tensor.external_data}
        if not fields:
            continue
        location = fields.get("location")
        if not location:
            raise ValueError("flow external initializer has no location")
        external = path.parent / location
        offset = int(fields.get("offset", -1))
        length = int(fields.get("length", -1))
        if offset < 0 or length < 0 or not external.is_file() or offset + length > external.stat().st_size:
            raise ValueError("flow external initializer has an invalid range")
        locations.add(location)
    return FlowGraphReport(len(q4_nodes), dynamic, tuple(sorted(locations)))


def flow_step_reference(
    state_dict: Mapping[str, torch.Tensor],
    latents: np.ndarray,
    condition: np.ndarray,
    timestep: np.ndarray,
    dt: np.ndarray,
    *,
    config: FlowTransformerConfig,
) -> np.ndarray:
    def weight(name: str) -> torch.Tensor:
        return state_dict[name].detach().to(torch.float16)

    latent = torch.from_numpy(latents).to(torch.float16)
    cond = torch.from_numpy(condition).to(torch.float16)
    time = torch.from_numpy(timestep).to(torch.float16)
    hidden = torch.cat((latent, latent), dim=0)
    cond = torch.cat((cond, torch.zeros_like(cond)), dim=0)
    time = torch.cat((time, time), dim=0)
    joined = torch.cat((hidden, torch.zeros_like(hidden), cond.transpose(1, 2)), dim=1)
    joined = torch_functional.conv1d(joined, weight("preprocess_conv.weight")) + joined
    hidden = torch_functional.linear(joined.transpose(1, 2), weight("proj_in.weight"))
    angles = 2.0 * np.pi * torch_functional.linear(time.unsqueeze(-1), weight("time_proj.weight"))
    time_hidden = torch.cat((angles.cos(), angles.sin()), dim=-1)
    time_hidden = torch_functional.linear(
        time_hidden,
        weight("time_embed.linear_1.weight"),
        weight("time_embed.linear_1.bias"),
    )
    time_hidden = torch_functional.silu(time_hidden)
    time_hidden = torch_functional.linear(
        time_hidden,
        weight("time_embed.linear_2.weight"),
        weight("time_embed.linear_2.bias"),
    )
    hidden = torch.cat((time_hidden.unsqueeze(1), hidden), dim=1)
    cosine, sine = fixed_rope_tables(hidden.shape[1], config.rotary_dim)
    cos = torch.from_numpy(cosine)
    sin = torch.from_numpy(sine)

    def rope(value: torch.Tensor) -> torch.Tensor:
        leading = value[..., : config.rotary_dim]
        first, second = leading.chunk(2, dim=-1)
        rotated = leading * cos + torch.cat((-second, first), dim=-1) * sin
        return torch.cat((rotated, value[..., config.rotary_dim :]), dim=-1)

    for index in range(config.num_layers):
        prefix = f"transformer_blocks.{index}"
        normalized = torch_functional.layer_norm(
            hidden,
            (config.hidden_size,),
            weight(f"{prefix}.norm1.weight"),
            weight(f"{prefix}.norm1.bias"),
        )
        projections = [
            torch_functional.linear(normalized, weight(f"{prefix}.attn.to_{part}.weight")).reshape(
                2, hidden.shape[1], config.num_attention_heads, config.attention_head_dim
            )
            for part in ("q", "k", "v")
        ]
        query, key, value = rope(projections[0]), rope(projections[1]), projections[2]
        query, key, value = (item.transpose(1, 2) for item in (query, key, value))
        probabilities = torch.softmax(query @ key.transpose(-1, -2) * config.attention_head_dim**-0.5, dim=-1)
        context = (probabilities @ value).transpose(1, 2).reshape(2, hidden.shape[1], config.hidden_size)
        hidden = hidden + torch_functional.linear(context, weight(f"{prefix}.attn.to_out.0.weight"))
        normalized = torch_functional.layer_norm(
            hidden,
            (config.hidden_size,),
            weight(f"{prefix}.norm2.weight"),
            weight(f"{prefix}.norm2.bias"),
        )
        gate_states, gate = torch_functional.linear(
            normalized,
            weight(f"{prefix}.ff_in.weight"),
            weight(f"{prefix}.ff_in.bias"),
        ).chunk(2, dim=-1)
        feed_forward = torch_functional.linear(
            gate_states * torch_functional.silu(gate),
            weight(f"{prefix}.ff_out.weight"),
            weight(f"{prefix}.ff_out.bias"),
        )
        hidden = hidden + feed_forward
    output = torch_functional.linear(hidden[:, 1:], weight("proj_out.weight")).transpose(1, 2)
    velocity = torch_functional.conv1d(output, weight("postprocess_conv.weight")) + output
    guided = velocity[1:2] + np.float16(1.7) * (velocity[0:1] - velocity[1:2])
    updated = latent.float() + torch.from_numpy(dt).float().reshape(1, 1, 1) * guided.float()
    return updated.to(torch.float16).numpy()
