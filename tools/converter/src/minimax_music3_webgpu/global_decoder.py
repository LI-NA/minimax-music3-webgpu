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
    external_locations: tuple[str, ...]


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
    output = paths.work / "global-builder"
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
    release = paths.release / ("global-one-layer" if num_hidden_layers == 1 else "global")
    repacked = repack_external_data(built, release, ARTIFACT_FILE_LIMIT)
    report = validate_global_decoder(repacked.model_path, expected_layers=num_hidden_layers)
    receipt = GlobalDecoderReceipt(
        built, repacked, report, tuple(arguments), process.stdout, process.stderr, elapsed,
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
    hidden = next((name for name in output_names if "hidden" in name), None)
    if hidden is None:
        raise ValueError("decoder is missing hidden-state output")
    attention = sum(node.op_type == "GroupQueryAttention" for node in graph.node)
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
            locations.add(location)
            length = int(fields.get("length", 0))
            if length > ARTIFACT_FILE_LIMIT:
                raise ValueError("decoder initializer exceeds artifact limit")
    return GraphReport(attention, past, present, hidden, tuple(sorted(locations)))


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
