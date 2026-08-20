"""Fixed-length MiniMax Music 3 vocoder export."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from uuid import uuid4
import warnings

import onnx
import numpy as np
import onnxruntime as ort
from safetensors.torch import load_file
import torch
from torch import nn
from torch.nn.utils import weight_norm

from .constants import ARTIFACT_FILE_LIMIT
from .external_data import repack_external_data


@dataclass(frozen=True)
class VocoderConfig:
    latent_channels: int = 128
    decoder_input_dim: int = 1024
    decoder_hidden_dim: int = 1536
    upsampling_ratios: tuple[int, ...] = (8, 8, 4, 2)
    latent_length: int = 430
    sampling_rate: int = 44_100

    @property
    def output_samples(self) -> int:
        return self.latent_length * math.prod(self.upsampling_ratios)


EXACT_VOCODER_CONFIG = VocoderConfig()
EXACT_FP32_SNAKES = ("blocks.0.snake1", "blocks.1.snake1")


@dataclass(frozen=True)
class GraphExpectations:
    input_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    node_counts: dict[str, int]


@dataclass(frozen=True)
class VocoderGraphReport:
    input_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    input_dtype: int
    output_dtype: int
    node_counts: dict[str, int]


@dataclass(frozen=True)
class CpuOracleReport:
    shape: tuple[int, int, int]
    finite: bool
    max_absolute_error: float


@dataclass(frozen=True)
class VocoderArtifact:
    model_path: Path
    shards: tuple[Path, ...]


def expected_source_shapes(config: VocoderConfig = EXACT_VOCODER_CONFIG) -> dict[str, tuple[int, ...]]:
    half_latent = config.latent_channels // 2
    shapes: dict[str, tuple[int, ...]] = {
        "dec_in_proj.weight": (config.decoder_input_dim, half_latent, 1),
        "dec_in_proj.bias": (config.decoder_input_dim,),
        "conv_in.weight_g": (config.decoder_hidden_dim, 1, 1),
        "conv_in.weight_v": (config.decoder_hidden_dim, config.decoder_input_dim, 7),
        "conv_in.bias": (config.decoder_hidden_dim,),
    }
    final_dim = config.decoder_hidden_dim
    for block_index, stride in enumerate(config.upsampling_ratios):
        input_dim = config.decoder_hidden_dim // (2**block_index)
        output_dim = config.decoder_hidden_dim // (2 ** (block_index + 1))
        final_dim = output_dim
        block = f"blocks.{block_index}"
        shapes[f"{block}.snake1.alpha"] = (1, input_dim, 1)
        shapes[f"{block}.conv_t1.weight_g"] = (input_dim, 1, 1)
        shapes[f"{block}.conv_t1.weight_v"] = (input_dim, output_dim, 2 * stride)
        shapes[f"{block}.conv_t1.bias"] = (output_dim,)
        for unit_index in range(1, 4):
            unit = f"{block}.res_unit{unit_index}"
            shapes[f"{unit}.snake1.alpha"] = (1, output_dim, 1)
            shapes[f"{unit}.conv1.weight_g"] = (output_dim, 1, 1)
            shapes[f"{unit}.conv1.weight_v"] = (output_dim, output_dim, 7)
            shapes[f"{unit}.conv1.bias"] = (output_dim,)
            shapes[f"{unit}.snake2.alpha"] = (1, output_dim, 1)
            shapes[f"{unit}.conv2.weight_g"] = (output_dim, 1, 1)
            shapes[f"{unit}.conv2.weight_v"] = (output_dim, output_dim, 1)
            shapes[f"{unit}.conv2.bias"] = (output_dim,)
    shapes.update(
        {
            "snake_out.alpha": (1, final_dim, 1),
            "conv_out.weight_g": (1, 1, 1),
            "conv_out.weight_v": (1, final_dim, 7),
            "conv_out.bias": (1,),
        }
    )
    return shapes


def validate_source_state_dict(
    state_dict: Mapping[str, torch.Tensor], config: VocoderConfig = EXACT_VOCODER_CONFIG
) -> None:
    expected = expected_source_shapes(config)
    missing = sorted(expected.keys() - state_dict.keys())
    extra = sorted(state_dict.keys() - expected.keys())
    if missing or extra:
        raise ValueError(f"vocoder state keys differ: missing={missing}, extra={extra}")
    for name, shape in expected.items():
        tensor = state_dict[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} shape must be {shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} dtype must be float32, got {tensor.dtype}")


def load_vocoder_state_dict(
    path: Path, config: VocoderConfig = EXACT_VOCODER_CONFIG
) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"missing vocoder safetensors: {path}")
    state_dict = load_file(path, device="cpu")
    validate_source_state_dict(state_dict, config)
    return state_dict


def fold_weight_norm(weight_g: torch.Tensor, weight_v: torch.Tensor) -> torch.Tensor:
    if weight_v.ndim < 2 or weight_g.shape != (weight_v.shape[0],) + (1,) * (weight_v.ndim - 1):
        raise ValueError("weight_g does not match dimension-zero weight norm")
    value = weight_v.to(torch.float32)
    scale = weight_g.to(torch.float32)
    norm = torch.linalg.vector_norm(value, dim=tuple(range(1, value.ndim)), keepdim=True)
    if torch.any(norm == 0):
        raise ValueError("weight_v has a zero dimension-zero norm")
    return value * (scale / norm)


def select_fp32_snakes(state_dict: Mapping[str, torch.Tensor]) -> tuple[str, ...]:
    threshold = torch.finfo(torch.float16).tiny
    selected = []
    for name in sorted(state_dict):
        if name.endswith(".alpha") and torch.any(state_dict[name].abs() < threshold):
            selected.append(name.removesuffix(".alpha"))
    return tuple(selected)


def prepare_vocoder_state_dict(
    source: Mapping[str, torch.Tensor], config: VocoderConfig = EXACT_VOCODER_CONFIG
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    fp32_snakes = select_fp32_snakes(source)
    if config == EXACT_VOCODER_CONFIG and fp32_snakes != EXACT_FP32_SNAKES:
        raise ValueError(f"unexpected FP32 Snake set for pinned checkpoint: {fp32_snakes}")
    folded = fold_vocoder_state_dict(source, config)
    prepared: dict[str, torch.Tensor] = {}
    for name, tensor in folded.items():
        snake = name.removesuffix(".alpha") if name.endswith(".alpha") else None
        dtype = torch.float32 if snake in fp32_snakes else torch.float16
        prepared[name] = tensor.to(dtype).clone()
    return prepared, fp32_snakes


def fold_vocoder_state_dict(
    source: Mapping[str, torch.Tensor], config: VocoderConfig = EXACT_VOCODER_CONFIG
) -> dict[str, torch.Tensor]:
    validate_source_state_dict(source, config)
    folded: dict[str, torch.Tensor] = {}
    for name in expected_source_shapes(config):
        if name.endswith(".weight_g"):
            continue
        if name.endswith(".weight_v"):
            prefix = name.removesuffix(".weight_v")
            folded[f"{prefix}.weight"] = fold_weight_norm(
                source[f"{prefix}.weight_g"], source[name]
            )
        else:
            folded[name] = source[name].to(torch.float32).clone()
    return folded


class MiniMaxMusic3Snake1d(nn.Module):
    def __init__(self, channels: int, force_float32: bool = False):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))
        self.force_float32 = force_float32

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        values = hidden_states.float() if self.force_float32 else hidden_states
        alpha = self.alpha.float() if self.force_float32 else self.alpha
        result = values + torch.reciprocal(alpha + 1e-9) * torch.sin(alpha * values).pow(2)
        return result.to(dtype) if self.force_float32 else result


class MiniMaxMusic3VocoderResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int, prefix: str, fp32_snakes: frozenset[str]):
        super().__init__()
        self.snake1 = MiniMaxMusic3Snake1d(dim, f"{prefix}.snake1" in fp32_snakes)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=3 * dilation)
        self.snake2 = MiniMaxMusic3Snake1d(dim, f"{prefix}.snake2" in fp32_snakes)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.snake2(self.conv1(self.snake1(hidden_states))))
        return hidden_states + residual


class MiniMaxMusic3VocoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stride: int,
        prefix: str,
        fp32_snakes: frozenset[str],
    ):
        super().__init__()
        self.snake1 = MiniMaxMusic3Snake1d(input_dim, f"{prefix}.snake1" in fp32_snakes)
        self.conv_t1 = nn.ConvTranspose1d(
            input_dim,
            output_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
        )
        self.res_unit1 = MiniMaxMusic3VocoderResidualUnit(
            output_dim, 1, f"{prefix}.res_unit1", fp32_snakes
        )
        self.res_unit2 = MiniMaxMusic3VocoderResidualUnit(
            output_dim, 3, f"{prefix}.res_unit2", fp32_snakes
        )
        self.res_unit3 = MiniMaxMusic3VocoderResidualUnit(
            output_dim, 9, f"{prefix}.res_unit3", fp32_snakes
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_t1(self.snake1(hidden_states))
        hidden_states = self.res_unit1(hidden_states)
        hidden_states = self.res_unit2(hidden_states)
        return self.res_unit3(hidden_states)


class MiniMaxMusic3VocoderExport(nn.Module):
    def __init__(self, config: VocoderConfig, fp32_snakes: tuple[str, ...] = ()):
        super().__init__()
        self.config = config
        selected = frozenset(fp32_snakes)
        self.dec_in_proj = nn.Conv1d(config.latent_channels // 2, config.decoder_input_dim, 1)
        self.conv_in = nn.Conv1d(config.decoder_input_dim, config.decoder_hidden_dim, 7, padding=3)
        blocks = []
        final_dim = config.decoder_hidden_dim
        for index, stride in enumerate(config.upsampling_ratios):
            input_dim = config.decoder_hidden_dim // (2**index)
            output_dim = config.decoder_hidden_dim // (2 ** (index + 1))
            final_dim = output_dim
            blocks.append(
                MiniMaxMusic3VocoderBlock(
                    input_dim, output_dim, stride, f"blocks.{index}", selected
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.snake_out = MiniMaxMusic3Snake1d(final_dim, "snake_out" in selected)
        self.conv_out = nn.Conv1d(final_dim, 1, 7, padding=3)

    @classmethod
    def from_prepared_state(
        cls,
        config: VocoderConfig,
        state_dict: Mapping[str, torch.Tensor],
        fp32_snakes: tuple[str, ...],
    ) -> "MiniMaxMusic3VocoderExport":
        with torch.device("meta"):
            module = cls(config, fp32_snakes)
        module.load_state_dict(dict(state_dict), strict=True, assign=True)
        return module.eval()

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        expected = (1, self.config.latent_channels, self.config.latent_length)
        if not torch.jit.is_tracing() and tuple(latents.shape) != expected:
            raise ValueError(f"latents shape must be {expected}")
        half = self.config.latent_channels // 2
        left, right = torch.split(latents, [half, half], dim=1)
        hidden_states = torch.cat((left, right), dim=0)
        hidden_states = self.conv_in(self.dec_in_proj(hidden_states))
        for block in self.blocks:
            hidden_states = block(hidden_states)
        waveform = torch.tanh(self.conv_out(self.snake_out(hidden_states)))
        left_waveform, right_waveform = torch.split(waveform, [1, 1], dim=0)
        return torch.cat((left_waveform, right_waveform), dim=1).float()


def build_vocoder_reference_module(
    source: Mapping[str, torch.Tensor], config: VocoderConfig = EXACT_VOCODER_CONFIG
) -> MiniMaxMusic3VocoderExport:
    validate_source_state_dict(source, config)
    with torch.device("meta"):
        module = MiniMaxMusic3VocoderExport(config)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="`torch.nn.utils.weight_norm` is deprecated.*",
            category=FutureWarning,
        )
        for name, child in list(module.named_modules()):
            if name != "dec_in_proj" and isinstance(child, (nn.Conv1d, nn.ConvTranspose1d)):
                weight_norm(child)
    module.load_state_dict(dict(source), strict=True, assign=True)
    return module.eval()


def graph_expectations(config: VocoderConfig = EXACT_VOCODER_CONFIG) -> GraphExpectations:
    blocks = len(config.upsampling_ratios)
    snakes = 1 + 7 * blocks
    return GraphExpectations(
        input_shape=(1, config.latent_channels, config.latent_length),
        output_shape=(1, 2, config.output_samples),
        node_counts={
            "Conv": 3 + 6 * blocks,
            "ConvTranspose": blocks,
            "Sin": snakes,
            "Pow": snakes,
            "Reciprocal": snakes,
            "Split": 2,
            "Concat": 2,
        },
    )


def export_vocoder_module(module: MiniMaxMusic3VocoderExport, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(
        graph_expectations(module.config).input_shape,
        dtype=torch.float16,
        device=next(module.parameters()).device,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="You are using the legacy TorchScript-based ONNX export.*"
        )
        warnings.filterwarnings(
            "ignore", message="The feature will be removed. Please remove usage of this function"
        )
        torch.onnx.export(
            module,
            (example,),
            output_path,
            input_names=["latents"],
            output_names=["waveform"],
            opset_version=18,
            dynamo=False,
            do_constant_folding=True,
        )
    validate_vocoder_graph(output_path, module.config)
    return output_path


def publish_vocoder_module(
    module: MiniMaxMusic3VocoderExport,
    output_dir: Path,
    max_file_bytes: int = ARTIFACT_FILE_LIMIT,
) -> VocoderArtifact:
    if max_file_bytes <= 0:
        raise ValueError("max_file_bytes must be positive")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{output_dir.name}-staging-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        raw_model = export_vocoder_module(module, staging / "raw" / "vocoder.onnx")
        packed = repack_external_data(raw_model, staging / "package", max_file_bytes)
        validate_vocoder_graph(packed.model_path, module.config)
        if packed.model_path.stat().st_size > max_file_bytes:
            raise ValueError("vocoder graph exceeds artifact limit")
        shard_names = tuple(shard.path.name for shard in packed.shards)
        _promote_vocoder_directory(staging / "package", output_dir)
    return VocoderArtifact(
        model_path=output_dir / "vocoder.onnx",
        shards=tuple(output_dir / name for name in shard_names),
    )


def _promote_vocoder_directory(staging: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}-{uuid4().hex}.backup")
    moved_existing = False
    cleanup_backup = True
    try:
        if destination.exists():
            destination.replace(backup)
            moved_existing = True
        staging.replace(destination)
    except Exception:
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        if moved_existing and backup.exists():
            try:
                backup.replace(destination)
            except Exception:
                cleanup_backup = False
                raise
        raise
    finally:
        if cleanup_backup and backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def validate_vocoder_graph(
    model_path: Path, config: VocoderConfig = EXACT_VOCODER_CONFIG
) -> VocoderGraphReport:
    model = onnx.load_model(model_path, load_external_data=False)
    if len(model.graph.input) != 1 or model.graph.input[0].name != "latents":
        raise ValueError("vocoder graph must have one latents input")
    if len(model.graph.output) != 1 or model.graph.output[0].name != "waveform":
        raise ValueError("vocoder graph must have one waveform output")
    input_shape, input_dtype = _value_contract(model.graph.input[0])
    output_shape, output_dtype = _value_contract(model.graph.output[0])
    expected = graph_expectations(config)
    if input_shape != expected.input_shape or input_dtype != onnx.TensorProto.FLOAT16:
        raise ValueError("vocoder input contract is invalid")
    if output_shape != expected.output_shape or output_dtype != onnx.TensorProto.FLOAT:
        raise ValueError("vocoder output contract is invalid")
    counts = Counter(node.op_type for node in model.graph.node)
    for operator, count in expected.node_counts.items():
        if counts[operator] != count:
            raise ValueError(f"vocoder graph must contain {count} {operator} nodes")
    forbidden = {"Reshape", "Shape", "ReduceL2"}
    if forbidden & counts.keys():
        raise ValueError("vocoder graph contains forbidden bookkeeping or weight-norm nodes")
    transpose_nodes = [node for node in model.graph.node if node.op_type == "ConvTranspose"]
    for node, stride in zip(transpose_nodes, config.upsampling_ratios, strict=True):
        attributes = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
        if attributes.get("kernel_shape") != [2 * stride]:
            raise ValueError("ConvTranspose kernel shape is invalid")
        if attributes.get("strides") != [stride]:
            raise ValueError("ConvTranspose stride is invalid")
        padding = math.ceil(stride / 2)
        if attributes.get("pads") != [padding, padding]:
            raise ValueError("ConvTranspose padding is invalid")
    return VocoderGraphReport(
        input_shape=input_shape,
        output_shape=output_shape,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        node_counts=dict(counts),
    )


def run_vocoder_cpu_oracle(
    module: MiniMaxMusic3VocoderExport,
    model_path: Path,
    latents: torch.Tensor,
    *,
    absolute_tolerance: float = 0.002,
    relative_tolerance: float = 0.002,
) -> CpuOracleReport:
    with torch.no_grad():
        expected = module.eval()(latents).detach().cpu().numpy()
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    actual = session.run(["waveform"], {"latents": latents.cpu().numpy()})[0]
    if actual.shape != expected.shape:
        raise ValueError(f"CPU oracle shape differs: {actual.shape} != {expected.shape}")
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    return CpuOracleReport(
        shape=tuple(actual.shape),
        finite=bool(np.isfinite(actual).all()),
        max_absolute_error=float(np.max(np.abs(actual - expected))),
    )


def _value_contract(value: onnx.ValueInfoProto) -> tuple[tuple[int, int, int], int]:
    tensor_type = value.type.tensor_type
    dimensions = tensor_type.shape.dim
    if len(dimensions) != 3 or any(dimension.dim_param or not dimension.HasField("dim_value") for dimension in dimensions):
        raise ValueError("vocoder tensor shape must be static rank three")
    return tuple(dimension.dim_value for dimension in dimensions), tensor_type.elem_type
