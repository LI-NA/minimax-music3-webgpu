"""Fixed-shape RVQ depth and feedback conversion."""

from dataclasses import dataclass
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from safetensors import safe_open
import torch

from .constants import ARTIFACT_FILE_LIMIT
from .embedding import EmbeddingShard, EmbeddingTableReceipt
from .external_data import RepackedModel, repack_external_data
from .paths import ArtifactPaths


@dataclass(frozen=True)
class RvqSourceReport:
    config: dict[str, int]
    tensor_count: int
    audio_embedding_shape: tuple[int, int]
    head_shapes: tuple[tuple[int, int], ...]
    source_sha256: str


@dataclass(frozen=True)
class RvqGraphReport:
    softmax_nodes: int
    rms_norms: int
    head_matmuls: int
    dynamic_shape_ops: tuple[str, ...]


_CONFIG_FIELDS = (
    "hidden_size",
    "num_layers",
    "num_attention_heads",
    "intermediate_size",
    "audio_vocab_size",
    "num_codebooks",
    "max_position_embeddings",
)


def inspect_rvq_source(config_path: str | Path, weights_path: str | Path) -> RvqSourceReport:
    config_file = Path(config_path)
    weights_file = Path(weights_path)
    raw_config = json.loads(config_file.read_text(encoding="utf-8"))
    config = {field: raw_config[field] for field in _CONFIG_FIELDS}
    with safe_open(weights_file, framework="pt", device="cpu") as source:
        shapes = {key: tuple(source.get_slice(key).get_shape()) for key in source.keys()}
        validate_rvq_metadata(config, shapes)
        embedding_shape = tuple(source.get_slice("audio_embeddings.weight").get_shape())
        head_shapes = tuple(
            tuple(source.get_slice(f"audio_heads.{index}.weight").get_shape())
            for index in range(config["num_codebooks"] - 1)
        )
    return RvqSourceReport(
        config,
        len(shapes),
        embedding_shape,
        head_shapes,
        _sha256(weights_file),
    )


def validate_rvq_metadata(config: dict[str, int], shapes: dict[str, tuple[int, ...]]) -> None:
    expected_config = {
        "hidden_size": 4096,
        "num_layers": 4,
        "num_attention_heads": 16,
        "intermediate_size": 6144,
        "audio_vocab_size": 1024,
        "num_codebooks": 8,
        "max_position_embeddings": 16,
    }
    if config != expected_config:
        raise ValueError(f"RVQ config mismatch: {config}")
    expected = _expected_shapes(config)
    if shapes != expected:
        missing = sorted(expected.keys() - shapes.keys())
        unexpected = sorted(shapes.keys() - expected.keys())
        wrong = sorted(key for key in expected.keys() & shapes.keys() if expected[key] != shapes[key])
        raise ValueError(
            f"RVQ safetensor metadata mismatch: missing={missing}, unexpected={unexpected}, wrong_shapes={wrong}"
        )


def export_rvq_depth_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    output_path: str | Path,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    num_layers: int,
    vocab_size: int,
    external_data: bool = False,
) -> Path:
    output = Path(output_path)
    positions = 8
    batch = 2
    head_dim = hidden_size // num_heads
    if hidden_size % num_heads:
        raise ValueError("hidden size must be divisible by attention heads")
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    def constant(name: str, value: np.ndarray) -> str:
        initializers.append(numpy_helper.from_array(value, name))
        return name

    axes_one = constant("axes_one", np.array([1], dtype=np.int64))
    axes_last = constant("axes_last", np.array([-1], dtype=np.int64))
    one = constant("one", np.array(1, dtype=np.int32))
    residual_shape = constant("residual_shape", np.array([batch, 6, hidden_size], dtype=np.int64))
    sequence_shape = constant("sequence_shape", np.array([batch, positions, hidden_size], dtype=np.int64))
    qkv_shape = constant("qkv_shape", np.array([batch, positions, num_heads, head_dim], dtype=np.int64))
    epsilon = constant("rms_epsilon", np.array(1e-6, dtype=np.float32))
    attention_scale = constant("attention_scale", np.array(head_dim**-0.5, dtype=np.float16))
    causal = np.full((1, 1, positions, positions), -65504, dtype=np.float16)
    causal = np.triu(causal, k=1)
    constant("causal_mask", causal)

    def weight(name: str) -> str:
        value = state_dict[name].detach().cpu().to(torch.float16).numpy()
        return constant(name, np.ascontiguousarray(value.T if value.ndim == 2 else value))

    def rms_norm(source: str, source_weight: str, prefix: str) -> str:
        nodes.extend(
            [
                helper.make_node("Cast", [source], [f"{prefix}.fp32"], to=TensorProto.FLOAT),
                helper.make_node("Mul", [f"{prefix}.fp32", f"{prefix}.fp32"], [f"{prefix}.square"]),
                helper.make_node("ReduceMean", [f"{prefix}.square", axes_last], [f"{prefix}.mean"], keepdims=1),
                helper.make_node("Add", [f"{prefix}.mean", epsilon], [f"{prefix}.variance"]),
                helper.make_node("Sqrt", [f"{prefix}.variance"], [f"{prefix}.sqrt"]),
                helper.make_node("Reciprocal", [f"{prefix}.sqrt"], [f"{prefix}.scale"]),
                helper.make_node("Mul", [f"{prefix}.fp32", f"{prefix}.scale"], [f"{prefix}.normalized_fp32"]),
                helper.make_node("Cast", [f"{prefix}.normalized_fp32"], [f"{prefix}.normalized"], to=TensorProto.FLOAT16),
                helper.make_node("Mul", [f"{prefix}.normalized", weight(source_weight)], [f"{prefix}.output"]),
            ]
        )
        return f"{prefix}.output"

    nodes.extend(
        [
            helper.make_node("Unsqueeze", ["global_last_hidden", axes_one], ["global_row"]),
            helper.make_node("Unsqueeze", ["semantic_embedding", axes_one], ["semantic_row"]),
            helper.make_node("Reshape", ["residual_embeddings", residual_shape], ["residual_rows"]),
            helper.make_node("Concat", ["global_row", "semantic_row", "residual_rows"], ["raw_sequence"], axis=1),
            helper.make_node("MatMul", ["raw_sequence", weight("projection.weight")], ["projected_sequence"]),
        ]
    )
    pos = state_dict["pos_embedding.weight"][:positions].detach().cpu().to(torch.float16).numpy()
    constant("position_embeddings", np.ascontiguousarray(pos))
    nodes.append(helper.make_node("Add", ["projected_sequence", "position_embeddings"], ["layer.0.input"]))
    hidden = "layer.0.input"
    for index in range(num_layers):
        prefix = f"layers.{index}"
        normed = rms_norm(hidden, f"{prefix}.input_layernorm.weight", f"{prefix}.input_norm")
        projections = {}
        for part in ("q", "k", "v"):
            matmul = f"{prefix}.{part}.flat"
            shaped = f"{prefix}.{part}.shaped"
            transposed = f"{prefix}.{part}"
            nodes.extend(
                [
                    helper.make_node("MatMul", [normed, weight(f"{prefix}.attn.to_{part}.weight")], [matmul]),
                    helper.make_node("Reshape", [matmul, qkv_shape], [shaped]),
                    helper.make_node("Transpose", [shaped], [transposed], perm=[0, 2, 1, 3]),
                ]
            )
            projections[part] = transposed
        nodes.extend(
            [
                helper.make_node("Transpose", [projections["k"]], [f"{prefix}.k_transposed"], perm=[0, 1, 3, 2]),
                helper.make_node("MatMul", [projections["q"], f"{prefix}.k_transposed"], [f"{prefix}.scores"]),
                helper.make_node("Mul", [f"{prefix}.scores", attention_scale], [f"{prefix}.scaled_scores"]),
                helper.make_node("Add", [f"{prefix}.scaled_scores", "causal_mask"], [f"{prefix}.masked_scores"]),
                helper.make_node("Softmax", [f"{prefix}.masked_scores"], [f"{prefix}.probabilities"], axis=-1),
                helper.make_node("MatMul", [f"{prefix}.probabilities", projections["v"]], [f"{prefix}.context"]),
                helper.make_node("Transpose", [f"{prefix}.context"], [f"{prefix}.context_transposed"], perm=[0, 2, 1, 3]),
                helper.make_node("Reshape", [f"{prefix}.context_transposed", sequence_shape], [f"{prefix}.context_flat"]),
                helper.make_node("MatMul", [f"{prefix}.context_flat", weight(f"{prefix}.attn.to_out.weight")], [f"{prefix}.attention"]),
                helper.make_node("Add", [hidden, f"{prefix}.attention"], [f"{prefix}.attention_residual"]),
            ]
        )
        post_norm = rms_norm(
            f"{prefix}.attention_residual",
            f"{prefix}.post_attention_layernorm.weight",
            f"{prefix}.post_norm",
        )
        nodes.extend(
            [
                helper.make_node("MatMul", [post_norm, weight(f"{prefix}.gate_proj.weight")], [f"{prefix}.gate"]),
                helper.make_node("Sigmoid", [f"{prefix}.gate"], [f"{prefix}.gate_sigmoid"]),
                helper.make_node("Mul", [f"{prefix}.gate", f"{prefix}.gate_sigmoid"], [f"{prefix}.silu"]),
                helper.make_node("MatMul", [post_norm, weight(f"{prefix}.up_proj.weight")], [f"{prefix}.up"]),
                helper.make_node("Mul", [f"{prefix}.silu", f"{prefix}.up"], [f"{prefix}.gated"]),
                helper.make_node("MatMul", [f"{prefix}.gated", weight(f"{prefix}.down_proj.weight")], [f"{prefix}.mlp"]),
                helper.make_node("Add", [f"{prefix}.attention_residual", f"{prefix}.mlp"], [f"layer.{index + 1}.input"]),
            ]
        )
        hidden = f"layer.{index + 1}.input"
    normalized = rms_norm(hidden, "norm.weight", "final_norm")
    nodes.extend(
        [
            helper.make_node("Add", ["depth_index", one], ["current_position"]),
            helper.make_node("Gather", [normalized, "current_position"], ["depth_hidden"], axis=1),
        ]
    )
    logits = []
    for index in range(7):
        name = f"head.{index}"
        expanded = f"{name}.expanded"
        nodes.extend(
            [
                helper.make_node("MatMul", ["depth_hidden", weight(f"audio_heads.{index}.weight")], [name]),
                helper.make_node("Unsqueeze", [name, axes_one], [expanded]),
            ]
        )
        logits.append(expanded)
    nodes.extend(
        [
            helper.make_node("Concat", logits, ["depth_logits_fp16"], axis=1),
            helper.make_node("Cast", ["depth_logits_fp16"], ["depth_logits"], to=TensorProto.FLOAT),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "minimax_music3_rvq_depth_fixed_8",
        [
            helper.make_tensor_value_info("global_last_hidden", TensorProto.FLOAT16, [batch, hidden_size]),
            helper.make_tensor_value_info("semantic_embedding", TensorProto.FLOAT16, [batch, hidden_size]),
            helper.make_tensor_value_info("residual_embeddings", TensorProto.FLOAT16, [batch, 6, hidden_size]),
            helper.make_tensor_value_info("depth_index", TensorProto.INT32, []),
        ],
        [
            helper.make_tensor_value_info("depth_hidden", TensorProto.FLOAT16, [batch, hidden_size]),
            helper.make_tensor_value_info("depth_logits", TensorProto.FLOAT, [batch, 7, vocab_size]),
        ],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    output.parent.mkdir(parents=True, exist_ok=True)
    if external_data:
        onnx.save_model(
            model,
            output,
            save_as_external_data=True,
            all_tensors_to_one_file=False,
            size_threshold=0,
            convert_attribute=False,
        )
        onnx.checker.check_model(output)
    else:
        onnx.checker.check_model(model)
        onnx.save_model(model, output)
    return output


def validate_rvq_graph(model_path: str | Path, *, hidden_size: int = 4096, vocab_size: int = 1024) -> RvqGraphReport:
    model = onnx.load_model(model_path, load_external_data=False)
    onnx.checker.check_model(model_path)
    inputs = {item.name: item.type.tensor_type for item in model.graph.input}
    outputs = {item.name: item.type.tensor_type for item in model.graph.output}
    expected_inputs = {
        "global_last_hidden": (TensorProto.FLOAT16, [2, hidden_size]),
        "semantic_embedding": (TensorProto.FLOAT16, [2, hidden_size]),
        "residual_embeddings": (TensorProto.FLOAT16, [2, 6, hidden_size]),
        "depth_index": (TensorProto.INT32, []),
    }
    expected_outputs = {
        "depth_hidden": (TensorProto.FLOAT16, [2, hidden_size]),
        "depth_logits": (TensorProto.FLOAT, [2, 7, vocab_size]),
    }
    for name, (element_type, shape) in {**expected_inputs, **expected_outputs}.items():
        tensor = inputs.get(name) or outputs.get(name)
        if tensor is None or tensor.elem_type != element_type or [item.dim_value for item in tensor.shape.dim] != shape:
            raise ValueError(f"RVQ graph value {name} has an invalid type or shape")
    dynamic = tuple(sorted({node.op_type for node in model.graph.node if node.op_type in {"Shape", "Range", "Trilu", "ConstantOfShape"}}))
    softmax = sum(node.op_type == "Softmax" for node in model.graph.node)
    rms_norms = sum(node.op_type == "ReduceMean" for node in model.graph.node)
    initializer_names = {item.name for item in model.graph.initializer}
    heads = sum(
        node.op_type == "MatMul" and any(name.startswith("audio_heads.") for name in node.input)
        for node in model.graph.node
    )
    if dynamic or softmax != 4 or rms_norms != 9 or heads != 7:
        raise ValueError("RVQ graph topology does not match the fixed four-layer contract")
    if "audio_embeddings.weight" in initializer_names:
        raise ValueError("RVQ graph must keep the audio embedding table external to ONNX")
    return RvqGraphReport(softmax, rms_norms, heads, dynamic)


def export_feedback_graph(output_path: str | Path, *, hidden_size: int = 4096) -> Path:
    output = Path(output_path)
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), "feedback_axes")
    scale = numpy_helper.from_array(np.array(8**-0.5, dtype=np.float16), "feedback_scale")
    graph = helper.make_graph(
        [
            helper.make_node("ReduceSum", ["residual_rows", "feedback_axes"], ["residual_sum"], keepdims=1),
            helper.make_node("Add", ["semantic_rows", "residual_sum"], ["frame_sum"]),
            helper.make_node("Mul", ["frame_sum", "feedback_scale"], ["inputs_embeds"]),
        ],
        "minimax_music3_feedback",
        [
            helper.make_tensor_value_info("semantic_rows", TensorProto.FLOAT16, [2, 1, hidden_size]),
            helper.make_tensor_value_info("residual_rows", TensorProto.FLOAT16, [2, 7, hidden_size]),
        ],
        [helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT16, [2, 1, hidden_size])],
        [axes, scale],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, output)
    return output


def export_rvq_embedding_table(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    rows: int = 7168,
    columns: int = 4096,
    max_file_bytes: int = ARTIFACT_FILE_LIMIT,
) -> EmbeddingTableReceipt:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    row_bytes = columns * np.dtype(np.float16).itemsize
    rows_per_shard = max_file_bytes // row_bytes
    if rows_per_shard < 1:
        raise ValueError("max_file_bytes cannot hold one RVQ embedding row")
    shards = []
    with safe_open(source_path, framework="pt", device="cpu") as source:
        tensor = source.get_slice("audio_embeddings.weight")
        if tuple(tensor.get_shape()) != (rows, columns):
            raise ValueError("audio_embeddings.weight has an unexpected shape")
        for start in range(0, rows, rows_per_shard):
            count = min(rows_per_shard, rows - start)
            values = tensor[start : start + count].to(dtype=torch.float16).numpy()
            path = output / f"embedding-{len(shards):03d}.fp16"
            np.ascontiguousarray(values).tofile(path)
            shards.append(
                EmbeddingShard(
                    path,
                    start,
                    count,
                    columns,
                    row_bytes,
                    path.stat().st_size,
                    _sha256(path),
                )
            )
    return EmbeddingTableReceipt(tuple(shards))


@dataclass(frozen=True)
class RvqStageReceipt:
    graph: RepackedModel
    embedding: EmbeddingTableReceipt
    feedback: Path
    source_sha256: str


def build_rvq_stage(paths: ArtifactPaths) -> RvqStageReceipt:
    source_dir = paths.source_path("rvq_depth_decoder")
    config_path = source_dir / "config.json"
    weights_path = source_dir / "diffusion_pytorch_model.safetensors"
    report = inspect_rvq_source(config_path, weights_path)
    raw_dir = paths.work / "rvq-depth-raw"
    packed_dir = paths.work / "rvq-depth-packed"
    embedding_dir = paths.work / "rvq-embedding"
    feedback_path = paths.work / "feedback.onnx"
    paths.validate_write_targets(raw_dir, packed_dir, embedding_dir, feedback_path, paths.receipts)
    raw_dir.mkdir(parents=True, exist_ok=True)
    with safe_open(weights_path, framework="pt", device="cpu") as source:
        state = {
            key: source.get_tensor(key)
            for key in source.keys()
            if key != "audio_embeddings.weight"
        }
    raw_graph = export_rvq_depth_from_state_dict(
        state,
        raw_dir / "rvq-depth.onnx",
        hidden_size=4096,
        intermediate_size=6144,
        num_heads=16,
        num_layers=4,
        vocab_size=1024,
        external_data=True,
    )
    del state
    packed = repack_external_data(raw_graph, packed_dir, ARTIFACT_FILE_LIMIT, inline_threshold=64)
    shutil.rmtree(raw_dir)
    validate_rvq_graph(packed.model_path)
    embedding = export_rvq_embedding_table(weights_path, embedding_dir)
    if sum(item.size for item in embedding.shards) != 56 * 1024 * 1024:
        raise ValueError("RVQ embedding table must be exactly 56 MiB")
    export_feedback_graph(feedback_path)
    _validate_feedback_graph(feedback_path)
    receipt = RvqStageReceipt(packed, embedding, feedback_path, report.source_sha256)
    payload = asdict(receipt)
    payload["graph"]["model_path"] = str(receipt.graph.model_path)
    for shard in payload["graph"]["shards"]:
        shard["path"] = str(shard["path"])
    for shard in payload["embedding"]["shards"]:
        shard["path"] = str(shard["path"])
    payload["feedback"] = str(receipt.feedback)
    receipt_path = paths.receipts / "rvq-stage.json"
    temporary = receipt_path.with_name(f".{receipt_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(receipt_path)
    return receipt


def _validate_feedback_graph(path: Path) -> None:
    model = onnx.load_model(path)
    onnx.checker.check_model(model)
    if [item.name for item in model.graph.output] != ["inputs_embeds"]:
        raise ValueError("feedback graph output contract is invalid")


def _expected_keys(num_layers: int, num_codebooks: int) -> set[str]:
    keys = {
        "audio_embeddings.weight",
        "norm.weight",
        "pos_embedding.weight",
        "projection.weight",
        *(f"audio_heads.{index}.weight" for index in range(num_codebooks - 1)),
    }
    for index in range(num_layers):
        prefix = f"layers.{index}"
        keys.update(
            {
                f"{prefix}.attn.to_q.weight",
                f"{prefix}.attn.to_k.weight",
                f"{prefix}.attn.to_v.weight",
                f"{prefix}.attn.to_out.weight",
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.gate_proj.weight",
                f"{prefix}.up_proj.weight",
                f"{prefix}.down_proj.weight",
            }
        )
    return keys


def _expected_shapes(config: dict[str, int]) -> dict[str, tuple[int, ...]]:
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    vocab = config["audio_vocab_size"]
    codebooks = config["num_codebooks"]
    shapes = {
        "audio_embeddings.weight": (vocab * (codebooks - 1), hidden),
        "norm.weight": (hidden,),
        "pos_embedding.weight": (config["max_position_embeddings"], hidden),
        "projection.weight": (hidden, hidden),
        **{f"audio_heads.{index}.weight": (vocab, hidden) for index in range(codebooks - 1)},
    }
    for index in range(config["num_layers"]):
        prefix = f"layers.{index}"
        shapes.update(
            {
                f"{prefix}.attn.to_q.weight": (hidden, hidden),
                f"{prefix}.attn.to_k.weight": (hidden, hidden),
                f"{prefix}.attn.to_v.weight": (hidden, hidden),
                f"{prefix}.attn.to_out.weight": (hidden, hidden),
                f"{prefix}.input_layernorm.weight": (hidden,),
                f"{prefix}.post_attention_layernorm.weight": (hidden,),
                f"{prefix}.gate_proj.weight": (intermediate, hidden),
                f"{prefix}.up_proj.weight": (intermediate, hidden),
                f"{prefix}.down_proj.weight": (hidden, intermediate),
            }
        )
    return shapes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
