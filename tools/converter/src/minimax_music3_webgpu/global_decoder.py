"""ORT GenAI q4 Global decoder conversion and graph validation."""

from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import onnx

from .constants import ARTIFACT_FILE_LIMIT
from .external_data import RepackedModel, repack_external_data
from .paths import ArtifactPaths


@dataclass(frozen=True)
class GraphReport:
    attention_nodes: int
    past_inputs: int
    present_outputs: int
    hidden_output: str
    matmul_nbits_nodes: int
    external_locations: tuple[str, ...]
    explicit_sequence_inputs: tuple[str, str]


@dataclass(frozen=True)
class GlobalDecoderReceipt:
    model_path: Path
    repacked: RepackedModel
    graph: GraphReport
    arguments: tuple[str, ...]
    stdout: str
    stderr: str
    elapsed_seconds: float
    exit_code: int
    versions: dict[str, str]


def builder_arguments(source: Path, output: Path, cache: Path, num_hidden_layers: int = 36, fuse_qk_norm_gqa: bool = True) -> list[str]:
    options = [
        "exclude_embeds=true",
        "exclude_lm_head=true",
        "filename=global_decoder.onnx",
        "block_size=128",
        "accuracy_level=4",
        "is_symmetric=true",
        "op_types_to_quantize=MatMul",
        f"fuse_qk_norm_gqa={str(fuse_qk_norm_gqa).lower()}",
    ]
    if num_hidden_layers != 36:
        options.append(f"num_hidden_layers={num_hidden_layers}")
    return [
        "-i", str(source), "-o", str(output), "-p", "int4", "-e", "webgpu", "-c", str(cache),
        "--extra_options", *options,
    ]


def build_global_decoder(paths: ArtifactPaths, num_hidden_layers: int = 36) -> GlobalDecoderReceipt:
    if num_hidden_layers not in {1, 36}:
        raise ValueError("num_hidden_layers must be 1 or 36")
    output = paths.work / f"global-builder-{num_hidden_layers}"
    cache = paths.work / "ortgenai-cache"
    paths.validate_write_targets(output, cache, paths.receipts)
    output.mkdir(parents=True, exist_ok=True)
    arguments = builder_arguments(paths.source / "language_model", output, cache, num_hidden_layers)
    started = time.perf_counter()
    process = subprocess.run(
        [sys.executable, "-m", "onnxruntime_genai.models.builder", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if process.returncode:
        raise RuntimeError(process.stderr or process.stdout)
    built = output / "global_decoder.onnx"
    if not built.is_file():
        raise FileNotFoundError(f"ORT GenAI builder did not produce {built}")
    rewrite_attention_mask_for_gqa(built)
    packed_dir = paths.work / f"global-packed-{num_hidden_layers}"
    paths.validate_write_targets(packed_dir)
    repacked = repack_external_data(built, packed_dir, ARTIFACT_FILE_LIMIT)
    report = validate_global_decoder(repacked.model_path, expected_layers=num_hidden_layers)
    receipt = GlobalDecoderReceipt(
        repacked.model_path, repacked, report, tuple(arguments), process.stdout, process.stderr, elapsed,
        process.returncode,
        {name: importlib.metadata.version(name) for name in ("onnxruntime", "onnxruntime-genai")},
    )
    receipt_path = paths.receipts / f"global-decoder-{num_hidden_layers}.json"
    _atomic_json(receipt_path, _receipt_payload(receipt))
    return receipt


def validate_global_decoder(model_path: Path, expected_layers: int = 36) -> GraphReport:
    model = onnx.load_model(model_path, load_external_data=False)
    graph = model.graph
    names = {value.name for value in graph.input}
    output_names = {value.name for value in graph.output}
    if "inputs_embeds" not in names:
        raise ValueError("decoder is missing inputs_embeds")
    if "attention_mask" in names:
        raise ValueError("decoder must not expose attention_mask")
    explicit_inputs = ("seqlens_k", "total_seq_len")
    if any(name not in names for name in explicit_inputs):
        raise ValueError("decoder is missing explicit sequence inputs")
    input_types = {item.name: item.type.tensor_type for item in graph.input}
    if (
        input_types["seqlens_k"].elem_type != onnx.TensorProto.INT32
        or input_types["total_seq_len"].elem_type != onnx.TensorProto.INT32
        or [dimension.dim_value for dimension in input_types["seqlens_k"].shape.dim] != [2]
        or len(input_types["total_seq_len"].shape.dim) != 0
    ):
        raise ValueError("decoder explicit sequence inputs have invalid types or shapes")
    hidden = next((name for name in output_names if "hidden" in name), None)
    if hidden is None:
        raise ValueError("decoder is missing hidden-state output")
    attention = sum(node.op_type == "GroupQueryAttention" for node in graph.node)
    gqa_nodes = [node for node in graph.node if node.op_type == "GroupQueryAttention"]
    if any(len(node.input) < 7 or node.input[5] != "seqlens_k" or node.input[6] != "total_seq_len" for node in gqa_nodes):
        raise ValueError("decoder GQA nodes must use explicit sequence inputs")
    mask_reformat_types = {"Shape", "Gather", "Cast", "ReduceSum", "Sub"}
    if any(
        node.op_type in mask_reformat_types
        and any("mask" in name.lower() for name in (*node.input, *node.output))
        for node in graph.node
    ):
        raise ValueError("decoder retains mask reformat nodes")
    past = sum("past" in name for name in names)
    present = sum("present" in name for name in output_names)
    if attention != expected_layers or past != expected_layers * 2 or present != expected_layers * 2:
        raise ValueError("decoder cache graph does not match expected layer count")
    q4_nodes = [node for node in graph.node if node.op_type == "MatMulNBits"]
    if not q4_nodes:
        raise ValueError("decoder has no q4 MatMulNBits nodes")
    for node in q4_nodes:
        attributes = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
        if attributes.get("bits") != 4 or attributes.get("block_size") != 128:
            raise ValueError("decoder q4 nodes must use block size 128")
    locations = set()
    for initializer in graph.initializer:
        lower = initializer.name.lower()
        if "embed" in lower and "token" in lower:
            raise ValueError("decoder includes token embedding initializer")
        if "lm_head" in lower:
            raise ValueError("decoder includes full LM-head initializer")
        fields = {entry.key: entry.value for entry in initializer.external_data}
        if fields:
            location = fields.get("location")
            if not location:
                raise ValueError("decoder external initializer has no location")
            relative = Path(location)
            external = (model_path.parent.resolve() / relative).resolve()
            if relative.is_absolute() or not external.is_relative_to(model_path.parent.resolve()):
                raise ValueError("decoder external initializer location escapes graph directory")
            if "offset" not in fields or "length" not in fields:
                raise ValueError("decoder external initializer requires offset and length")
            try:
                offset = int(fields["offset"])
                length = int(fields["length"])
            except ValueError as error:
                raise ValueError("decoder external initializer has invalid range") from error
            if offset < 0 or length < 0 or not external.is_file() or offset + length > external.stat().st_size:
                raise ValueError("decoder external initializer has invalid range")
            locations.add(location)
            if length > ARTIFACT_FILE_LIMIT:
                raise ValueError("decoder initializer exceeds artifact limit")
    return GraphReport(attention, past, present, hidden, len(q4_nodes), tuple(sorted(locations)), explicit_inputs)


def rewrite_attention_mask_for_gqa(model_path: Path, batch_size: int = 2) -> None:
    """Replace builder mask bookkeeping with the GQA inputs supported by WebGPU."""
    model = onnx.load_model(model_path, load_external_data=False)
    graph = model.graph
    if any(item.name == "seqlens_k" for item in graph.input) or any(item.name == "total_seq_len" for item in graph.input):
        raise ValueError("decoder already has explicit sequence inputs")
    if not any(item.name == "attention_mask" for item in graph.input):
        raise ValueError("decoder is missing attention_mask for GQA rewrite")
    gqa_nodes = [node for node in graph.node if node.op_type == "GroupQueryAttention"]
    if not gqa_nodes or any(len(node.input) < 7 for node in gqa_nodes):
        raise ValueError("decoder GQA nodes do not expose sequence inputs")

    producers = {output: node for node in graph.node for output in node.output if output}
    removable: set[int] = set()

    def trace(name: str) -> None:
        node = producers.get(name)
        if node is None:
            return
        index = id(node)
        if index in removable:
            return
        removable.add(index)
        for source in node.input:
            trace(source)

    for node in gqa_nodes:
        trace(node.input[5])
        trace(node.input[6])
        node.input[5] = "seqlens_k"
        node.input[6] = "total_seq_len"
    removable_nodes = [node for node in graph.node if id(node) in removable]
    if not removable_nodes or not any("attention_mask" in node.input for node in removable_nodes):
        raise ValueError("decoder mask reformat subgraph was not found")
    for node in removable_nodes:
        for output in node.output:
            consumers = [candidate for candidate in graph.node if output and output in candidate.input]
            if any(id(candidate) not in removable for candidate in consumers) or output in {item.name for item in graph.output}:
                raise ValueError("decoder mask reformat output has an external consumer")
    retained_nodes = [node for node in graph.node if id(node) not in removable]
    del graph.node[:]
    graph.node.extend(retained_nodes)
    retained_inputs = [item for item in graph.input if item.name != "attention_mask"]
    del graph.input[:]
    graph.input.extend(retained_inputs)
    graph.input.extend([
        onnx.helper.make_tensor_value_info("seqlens_k", onnx.TensorProto.INT32, [batch_size]),
        onnx.helper.make_tensor_value_info("total_seq_len", onnx.TensorProto.INT32, []),
    ])
    onnx.save_model(model, model_path)
    _check_model_structure(model, model_path)


def _check_model_structure(model: onnx.ModelProto, model_path: Path) -> None:
    """Run ONNX structural validation while preserving ORT GenAI's extension domain."""
    checker_model = onnx.ModelProto()
    checker_model.CopyFrom(model)
    for node in checker_model.graph.node:
        if node.domain == "" and node.op_type == "SimplifiedLayerNormalization":
            node.domain = "com.microsoft"
    checker_path = model_path.with_name(f".{model_path.name}.check")
    try:
        onnx.save_model(checker_model, checker_path)
        onnx.checker.check_model(checker_path)
    finally:
        checker_path.unlink(missing_ok=True)


def _receipt_payload(receipt: GlobalDecoderReceipt) -> dict:
    payload = asdict(receipt)
    payload["model_path"] = str(receipt.model_path)
    payload["repacked"]["model_path"] = str(receipt.repacked.model_path)
    for shard in payload["repacked"]["shards"]:
        shard["path"] = str(shard["path"])
    return payload


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
