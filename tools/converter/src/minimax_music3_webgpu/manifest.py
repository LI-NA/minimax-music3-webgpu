"""Release manifest construction with complete file integrity metadata."""

import hashlib
import json
from pathlib import Path
import shutil
import re
from uuid import uuid4

import onnx

from .constants import (
    ARTIFACT_FILE_LIMIT,
    DIFFUSERS_REVISION,
    FLOW_FP16_LINEAR_WEIGHTS,
    HIDDEN_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    Q4_ACCURACY_LEVEL,
    Q4_BITS,
    Q4_BLOCK_SIZE,
    Q4_PROFILE,
    Q4_SYMMETRIC,
    VOCAB_SIZE,
)
from .paths import ArtifactPaths
from .embedding import export_embedding_table
from .reduced_head import export_reduced_head
from .embedding import EmbeddingTableReceipt
from .external_data import RepackedModel
from .rvq_depth import RvqStageReceipt
from .vocoder import EXACT_FP32_SNAKES, validate_vocoder_graph


def emit_manifest(
    path: Path, *, graph: Path, external_data: list[tuple[str, Path]],
    embedding_shards: list[tuple[int, int, Path]], tokenizer_files: list[tuple[str, Path]],
    license_file: Path, kv_pairs: list[tuple[str, str]], reduced_head: Path | None = None,
    hidden_output: str = "hidden_states",
) -> Path:
    entries = [("graph", graph), *external_data, *(("embedding", file) for _, _, file in embedding_shards), *tokenizer_files, ("LICENSE", license_file)]
    if reduced_head is not None:
        entries.append(("reduced head", reduced_head))
    _validate_files(entries)
    root = path.parent
    _validate_embedding_shards(embedding_shards)
    graph_model = onnx.load_model(graph, load_external_data=False)
    output_names = {output.name for output in graph_model.graph.output}
    gpu_outputs = [hidden_output, *[present for _, present in kv_pairs]]
    if not set(gpu_outputs) <= output_names:
        raise ValueError("declared GPU output is missing from decoder graph")
    external = [_file(file, root, onnx_location=location) for location, file in external_data]
    payload = {
        "schemaVersion": 1,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "diffusersRevision": DIFFUSERS_REVISION},
        "quantization": {
            "bits": Q4_BITS,
            "blockSize": Q4_BLOCK_SIZE,
            "accuracyLevel": Q4_ACCURACY_LEVEL,
            "symmetric": Q4_SYMMETRIC,
        },
        "webgpu": {"requiredFeatures": ["shader-f16"], "requiredLimits": {"maxStorageBufferBindingSize": ARTIFACT_FILE_LIMIT, "maxStorageBuffersPerShaderStage": 9}},
        "graph": {**_file(graph, root), "externalData": external, "gpuOutputs": gpu_outputs},
        "embedding": {"rows": 200000, "columns": 4096, "rowBytes": 8192, "shards": [{**_file(file, root), "rowStart": start, "rowCount": count} for start, count, file in embedding_shards]},
        "tokenizerFiles": [{**_file(file, root), "path": location} for location, file in tokenizer_files],
        "licenseFile": _file(license_file, root),
        "kvPairs": [{"pastInput": past, "presentOutput": present} for past, present in kv_pairs],
    }
    if reduced_head is not None:
        payload["reducedHead"] = _onnx_graph(reduced_head, root, ["last_state"])
    _atomic_json(path, payload)
    return path


def emit_rvq_manifest(
    path: Path,
    *,
    rvq_depth: Path,
    feedback: Path,
    embedding: EmbeddingTableReceipt,
    rows: int = 7168,
    columns: int = 4096,
) -> Path:
    root = path.parent
    shards = [(item.row_start, item.row_count, item.path) for item in embedding.shards]
    _validate_files([("RVQ depth", rvq_depth), ("feedback", feedback), *(("RVQ embedding", item.path) for item in embedding.shards)])
    _validate_row_shards(shards, rows, columns)
    payload = {
        "schemaVersion": 1,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "diffusersRevision": DIFFUSERS_REVISION},
        "webgpu": {"requiredFeatures": ["shader-f16"], "requiredLimits": {"maxStorageBufferBindingSize": ARTIFACT_FILE_LIMIT, "maxStorageBuffersPerShaderStage": 9}},
        "rvqDepth": _onnx_graph(rvq_depth, root, ["depth_hidden"]),
        "rvqEmbedding": {
            "rows": rows,
            "columns": columns,
            "rowBytes": columns * 2,
            "shards": [
                {**_file(file, root), "rowStart": start, "rowCount": count}
                for start, count, file in shards
            ],
        },
        "feedback": _onnx_graph(feedback, root, ["inputs_embeds"]),
    }
    _atomic_json(path, payload)
    return path


def emit_rvq_release(paths: ArtifactPaths, stage: RvqStageReceipt) -> Path:
    release = paths.release / "rvq"
    staging = paths.release / f".rvq-{uuid4().hex}.staging"
    try:
        (staging / "rvq-depth").mkdir(parents=True)
        (staging / "embedding").mkdir(parents=True)
        shutil.copy2(stage.graph.model_path, staging / "rvq-depth" / stage.graph.model_path.name)
        for shard in stage.graph.shards:
            shutil.copy2(shard.path, staging / "rvq-depth" / shard.path.name)
        for shard in stage.embedding.shards:
            shutil.copy2(shard.path, staging / "embedding" / shard.path.name)
        shutil.copy2(stage.feedback, staging / "feedback.onnx")
        graph = staging / "rvq-depth" / stage.graph.model_path.name
        embedding = EmbeddingTableReceipt(
            tuple(
                type(item)(
                    staging / "embedding" / item.path.name,
                    item.row_start,
                    item.row_count,
                    item.columns,
                    item.row_bytes,
                    item.size,
                    item.sha256,
                )
                for item in stage.embedding.shards
            )
        )
        manifest = emit_rvq_manifest(
            staging / "manifest.json",
            rvq_depth=graph,
            feedback=staging / "feedback.onnx",
            embedding=embedding,
        )
        _validate_rvq_release(manifest)
        _promote_directory(staging, release)
        return release / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def emit_condition_manifest(path: Path, *, condition_encoder: Path) -> Path:
    _validate_files([("condition encoder", condition_encoder)])
    payload = {
        "schemaVersion": 1,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "diffusersRevision": DIFFUSERS_REVISION,
        },
        "webgpu": {
            "requiredFeatures": ["shader-f16"],
            "requiredLimits": {"maxStorageBufferBindingSize": ARTIFACT_FILE_LIMIT},
        },
        "conditionEncoder": _onnx_graph(condition_encoder, path.parent, ["condition"]),
    }
    _atomic_json(path, payload)
    return path


def emit_condition_release(paths: ArtifactPaths, graph: RepackedModel) -> Path:
    release = paths.release / "condition"
    staging = paths.release / f".condition-{uuid4().hex}.staging"
    try:
        graph_dir = staging / "condition-encoder"
        graph_dir.mkdir(parents=True)
        shutil.copy2(graph.model_path, graph_dir / graph.model_path.name)
        for shard in graph.shards:
            shutil.copy2(shard.path, graph_dir / shard.path.name)
        manifest = emit_condition_manifest(
            staging / "manifest.json",
            condition_encoder=graph_dir / graph.model_path.name,
        )
        _validate_condition_release(manifest)
        _promote_directory(staging, release)
        return release / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def emit_flow_manifest(path: Path, *, flow_step: Path) -> Path:
    _validate_files([("flow step", flow_step)])
    payload = {
        "schemaVersion": 1,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "diffusersRevision": DIFFUSERS_REVISION,
        },
        "quantization": {
            "bits": Q4_BITS,
            "blockSize": Q4_BLOCK_SIZE,
            "accuracyLevel": Q4_ACCURACY_LEVEL,
            "symmetric": Q4_SYMMETRIC,
        },
        "precision": {"float16Weights": list(FLOW_FP16_LINEAR_WEIGHTS)},
        "webgpu": {
            "requiredFeatures": ["shader-f16"],
            "requiredLimits": {
                "maxStorageBufferBindingSize": ARTIFACT_FILE_LIMIT,
                "maxStorageBuffersPerShaderStage": 9,
            },
        },
        "slice": {
            "semanticFrames": 125,
            "latentLength": 430,
            "flowSteps": 30,
            "flowGuidance": 1.7,
        },
        "flow": _onnx_graph(flow_step, path.parent, ["next_latents"]),
    }
    _atomic_json(path, payload)
    return path


def emit_flow_release(paths: ArtifactPaths, flow_step: Path) -> Path:
    model = onnx.load_model(flow_step, load_external_data=False)
    external = _external_files(model, flow_step)
    _validate_files(
        [("flow graph", flow_step), *(("flow external data", file) for _, file in external)]
    )
    if any(file.stat().st_size > ARTIFACT_FILE_LIMIT for _, file in external):
        raise ValueError("flow external-data file exceeds the artifact limit")
    release = paths.release / "flow"
    staging = paths.release / f".flow-{uuid4().hex}.staging"
    try:
        graph_dir = staging / "flow-step"
        graph_dir.mkdir(parents=True)
        graph = graph_dir / "flow-step.onnx"
        shutil.copy2(flow_step, graph)
        for location, source in external:
            target = (graph_dir / location).resolve()
            if not target.is_relative_to(graph_dir.resolve()):
                raise ValueError("flow external-data path escapes graph directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        manifest = emit_flow_manifest(staging / "manifest.json", flow_step=graph)
        _validate_flow_release(manifest)
        _promote_directory(staging, release)
        return release / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def emit_vocoder_manifest(path: Path, *, vocoder: Path) -> Path:
    _validate_files([("vocoder", vocoder)])
    validate_vocoder_graph(vocoder)
    payload = {
        "schemaVersion": 1,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "diffusersRevision": DIFFUSERS_REVISION,
        },
        "webgpu": {
            "requiredFeatures": ["shader-f16"],
            "requiredLimits": {"maxStorageBufferBindingSize": ARTIFACT_FILE_LIMIT},
        },
        "slice": {
            "latentChannels": 128,
            "latentLength": 430,
            "outputSamples": 220_160,
            "sampleRate": 44_100,
            "channels": 2,
        },
        "precision": {
            "convolution": "float16",
            "fp32Snakes": list(EXACT_FP32_SNAKES),
        },
        "vocoder": _onnx_graph(vocoder, path.parent, []),
    }
    _atomic_json(path, payload)
    return path


def emit_vocoder_release(paths: ArtifactPaths, vocoder: Path) -> Path:
    model = onnx.load_model(vocoder, load_external_data=False)
    external = _external_files(model, vocoder)
    _validate_files(
        [("vocoder graph", vocoder), *(("vocoder external data", file) for _, file in external)]
    )
    if vocoder.stat().st_size > ARTIFACT_FILE_LIMIT or any(
        file.stat().st_size > ARTIFACT_FILE_LIMIT for _, file in external
    ):
        raise ValueError("vocoder artifact exceeds the artifact limit")
    release = paths.release / "vocoder"
    staging = paths.release / f".vocoder-{uuid4().hex}.staging"
    try:
        graph_dir = staging / "vocoder"
        graph_dir.mkdir(parents=True)
        graph = graph_dir / "vocoder.onnx"
        shutil.copy2(vocoder, graph)
        for location, source in external:
            target = (graph_dir / location).resolve()
            if not target.is_relative_to(graph_dir.resolve()):
                raise ValueError("vocoder external-data path escapes graph directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        manifest = emit_vocoder_manifest(staging / "manifest.json", vocoder=graph)
        _validate_vocoder_release(manifest)
        _promote_directory(staging, release)
        return release / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _validate_rvq_release(manifest: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    entries = [
        payload["rvqDepth"], payload["feedback"],
        *payload["rvqDepth"]["externalData"], *payload["feedback"]["externalData"],
        *payload["rvqEmbedding"]["shards"],
    ]
    for entry in entries:
        file = manifest.parent / entry["path"]
        if not file.is_file() or file.stat().st_size != entry["bytes"] or _sha256(file) != entry["sha256"]:
            raise ValueError("RVQ release manifest reference is invalid")


def _validate_condition_release(manifest: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    graph = payload["conditionEncoder"]
    for entry in [graph, *graph["externalData"]]:
        file = manifest.parent / entry["path"]
        if not file.is_file() or file.stat().st_size != entry["bytes"] or _sha256(file) != entry["sha256"]:
            raise ValueError("condition release manifest reference is invalid")


def _validate_flow_release(manifest: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    graph = payload["flow"]
    for entry in [graph, *graph["externalData"]]:
        file = manifest.parent / entry["path"]
        if (
            not file.is_file()
            or file.stat().st_size != entry["bytes"]
            or _sha256(file) != entry["sha256"]
        ):
            raise ValueError("flow release manifest reference is invalid")


def _validate_vocoder_release(manifest: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    graph = payload["vocoder"]
    for entry in [graph, *graph["externalData"]]:
        file = manifest.parent / entry["path"]
        if (
            not file.is_file()
            or file.stat().st_size != entry["bytes"]
            or _sha256(file) != entry["sha256"]
        ):
            raise ValueError("vocoder release manifest reference is invalid")


def _validate_row_shards(shards: list[tuple[int, int, Path]], rows: int, columns: int) -> None:
    next_start = 0
    for start, count, file in shards:
        if start != next_start or count <= 0 or file.stat().st_size != count * columns * 2:
            raise ValueError("invalid RVQ embedding shard receipt")
        next_start += count
    if next_start != rows:
        raise ValueError("RVQ embedding shards do not cover the table")


def emit_global_release(paths: ArtifactPaths, num_hidden_layers: int = 36) -> Path:
    if num_hidden_layers not in {1, 36}:
        raise ValueError("num_hidden_layers must be 1 or 36")
    name = "global-one-layer" if num_hidden_layers == 1 else "global"
    release = paths.release / name
    packed = paths.work / f"global-packed-{Q4_PROFILE}-{num_hidden_layers}"
    graph = packed / "global_decoder.onnx"
    if not graph.is_file():
        raise FileNotFoundError(f"missing converted decoder: {graph}")
    staging = paths.release / f".{name}-{uuid4().hex}.staging"
    try:
        staging.mkdir(parents=True)
        _copy_tree(packed, staging)
        graph = staging / "global_decoder.onnx"
        embedding_source = _source_shard(paths.source / "language_model", "model.embed_tokens.weight")
        head_source = _source_shard(paths.source / "language_model", "lm_head.weight")
        embedding = export_embedding_table(embedding_source, staging / "embedding")
        head = export_reduced_head(head_source, staging / "reduced-head")
        _copy_tree(paths.source / "tokenizer", staging / "tokenizer")
        shutil.copy2(paths.source / "LICENSE", staging / "LICENSE")
        model = onnx.load_model(graph, load_external_data=False)
        external = _external_files(model, graph)
        tokenizers = [(file.relative_to(staging).as_posix(), file) for file in sorted((staging / "tokenizer").rglob("*")) if file.is_file()]
        manifest = emit_manifest(staging / "manifest.json", graph=graph, external_data=external,
            embedding_shards=[(item.row_start, item.row_count, item.path) for item in embedding.shards], tokenizer_files=tokenizers,
            license_file=staging / "LICENSE", kv_pairs=_kv_pairs(model), reduced_head=head.model_path,
            hidden_output=_hidden_output(model))
        _validate_release(manifest)
        final = release / "manifest.json"
        _promote_directory(
            staging,
            release,
            receipt_path=paths.receipts / f"global-release-{num_hidden_layers}.json",
            receipt_payload={
                "manifest": str(final),
                "reducedHead": str(release / head.model_path.relative_to(staging)),
            },
        )
        return final
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _source_shard(language_model: Path, tensor_name: str) -> Path:
    indexes = sorted(language_model.glob("*.safetensors.index.json"))
    if len(indexes) != 1:
        raise ValueError("language model must contain exactly one safetensors index")
    weight_map = json.loads(indexes[0].read_text(encoding="utf-8")).get("weight_map", {})
    name = weight_map.get(tensor_name)
    if not isinstance(name, str):
        raise ValueError(f"safetensors index is missing {tensor_name}")
    relative = Path(name)
    root = language_model.resolve()
    shard = (root / relative).resolve()
    if relative.is_absolute() or not shard.is_relative_to(root):
        raise ValueError("safetensors index path escapes language model")
    if not shard.is_file():
        raise FileNotFoundError(f"missing safetensors shard: {shard}")
    return shard


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"missing tokenizer directory: {source}")
    for file in source.rglob("*"):
        if file.is_file():
            target = destination / file.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)


def _promote_directory(
    staging: Path,
    release: Path,
    *,
    receipt_path: Path | None = None,
    receipt_payload: dict | None = None,
) -> None:
    if (receipt_path is None) != (receipt_payload is None):
        raise ValueError("receipt path and payload must be provided together")
    artifact_root = _validate_promotion_paths(staging, release, receipt_path)
    if release.exists():
        _archive_release(artifact_root, release, receipt_path)
    backup = release.with_name(f".{release.name}-{uuid4().hex}.backup")
    moved_old = False
    published = False
    cleanup_backup = True
    try:
        if release.exists():
            release.replace(backup)
            moved_old = True
        staging.replace(release)
        published = True
        if receipt_path is not None and receipt_payload is not None:
            _atomic_json(receipt_path, receipt_payload)
    except Exception:
        if published and release.exists():
            shutil.rmtree(release, ignore_errors=True)
        if moved_old and backup.exists():
            try:
                backup.replace(release)
            except Exception:
                cleanup_backup = False
                raise
        raise
    finally:
        if cleanup_backup and backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def _validate_promotion_paths(
    staging: Path,
    release: Path,
    receipt_path: Path | None,
) -> Path:
    release_root = release.parent.resolve()
    if release_root.name != "release":
        raise ValueError("release must be directly under the artifact release root")
    if release.resolve().parent != release_root:
        raise ValueError("release must be directly under the artifact release root")
    if staging.resolve().parent != release_root or staging.resolve() == release.resolve():
        raise ValueError("staging must be directly under the artifact release root")
    artifact_root = release_root.parent
    if (
        receipt_path is not None
        and receipt_path.resolve().parent != (artifact_root / "receipts").resolve()
    ):
        raise ValueError("receipt must be directly under the artifact receipts root")
    return artifact_root


def _archive_release(
    artifact_root: Path,
    release: Path,
    receipt_path: Path | None,
) -> Path:
    archive_root = artifact_root / "archive" / release.name
    generation = uuid4().hex
    staging = archive_root / f".{generation}.staging"
    archived = archive_root / generation
    try:
        _archive_tree(release, staging / "release")
        if receipt_path is not None and receipt_path.exists():
            _archive_file(receipt_path, staging / "receipt.json")
        staging.replace(archived)
        return archived
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _archive_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for item in sorted(source.rglob("*")):
        if item.is_symlink():
            raise ValueError("release archive source must not contain symbolic links")
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            _archive_file(item, target)
        else:
            raise ValueError("release archive source contains an unsupported entry")
    _verify_archive_tree(source, destination)


def _archive_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError("release archive source must be a regular file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if (
        destination.stat().st_size != source.stat().st_size
        or _sha256(destination) != _sha256(source)
    ):
        raise OSError(f"release archive verification failed: {source}")


def _verify_archive_tree(source: Path, destination: Path) -> None:
    source_entries = {
        item.relative_to(source): item.is_dir()
        for item in source.rglob("*")
    }
    destination_entries = {
        item.relative_to(destination): item.is_dir()
        for item in destination.rglob("*")
    }
    if source_entries != destination_entries:
        raise OSError("release archive tree verification failed")


def _external_files(model: onnx.ModelProto, graph: Path) -> list[tuple[str, Path]]:
    locations = set()
    for tensor in model.graph.initializer:
        fields = {entry.key: entry.value for entry in tensor.external_data}
        if not fields:
            continue
        location = _external_location(graph.parent, fields)
        offset = _nonnegative(fields, "offset")
        length = _nonnegative(fields, "length")
        if length > ARTIFACT_FILE_LIMIT or offset + length > location.stat().st_size:
            raise ValueError("invalid external initializer range")
        locations.add((fields["location"], location))
    return sorted(locations)


def _external_location(root: Path, fields: dict[str, str]) -> Path:
    raw = fields.get("location")
    if not raw:
        raise ValueError("external initializer has no location")
    relative = Path(raw)
    resolved = (root.resolve() / relative).resolve()
    if relative.is_absolute() or not resolved.is_relative_to(root.resolve()):
        raise ValueError("external initializer location escapes graph directory")
    if not resolved.is_file():
        raise ValueError("external initializer file is missing")
    return resolved


def _nonnegative(fields: dict[str, str], field: str) -> int:
    if field not in fields:
        raise ValueError(f"external initializer has no {field}")
    try:
        value = int(fields[field])
    except ValueError as error:
        raise ValueError(f"external initializer has invalid {field}") from error
    if value < 0:
        raise ValueError(f"external initializer has invalid {field}")
    return value


def _hidden_output(model: onnx.ModelProto) -> str:
    names = [output.name for output in model.graph.output if "hidden" in output.name]
    if len(names) != 1:
        raise ValueError("decoder must have exactly one hidden output")
    return names[0]


def _validate_embedding_shards(shards: list[tuple[int, int, Path]]) -> None:
    next_start = 0
    row_bytes = HIDDEN_SIZE * 2
    for start, count, file in shards:
        if start != next_start or count <= 0 or file.stat().st_size != count * row_bytes:
            raise ValueError("invalid embedding shard receipt")
        next_start += count
    if next_start != VOCAB_SIZE:
        raise ValueError("embedding shards do not cover vocabulary")


_KV_INPUT = re.compile(r"past_key_values\.(\d+)\.(key|value)$")
_KV_OUTPUT = re.compile(r"present\.(\d+)\.(key|value)$")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _kv_pairs(model: onnx.ModelProto) -> list[tuple[str, str]]:
    inputs = _kv_names([item.name for item in model.graph.input], _KV_INPUT)
    outputs = _kv_names([item.name for item in model.graph.output], _KV_OUTPUT)
    if inputs.keys() != outputs.keys():
        raise ValueError("decoder KV inputs and outputs do not pair")
    return [(inputs[key], outputs[key]) for key in sorted(inputs)]


def _kv_names(names: list[str], pattern: re.Pattern[str]) -> dict[tuple[int, str], str]:
    matched = {}
    for name in names:
        if "past" in name or "present" in name:
            match = pattern.fullmatch(name)
            if not match:
                raise ValueError("decoder has unexpected KV name")
            key = (int(match.group(1)), match.group(2))
            if key in matched:
                raise ValueError("decoder has duplicate KV name")
            matched[key] = name
    return matched


def _validate_files(entries: list[tuple[str, Path]]) -> None:
    seen = set()
    for location, file in entries:
        key = file.resolve()
        if key in seen:
            raise ValueError("duplicate artifact path")
        seen.add(key)
        if not file.is_file():
            raise ValueError(f"missing referenced file: {file}")


def _onnx_graph(path: Path, root: Path, gpu_outputs: list[str]) -> dict:
    model = onnx.load_model(path, load_external_data=False)
    outputs = {item.name for item in model.graph.output}
    if not set(gpu_outputs) <= outputs:
        raise ValueError("declared GPU output is missing from reduced head")
    return {**_file(path, root), "externalData": [_file(file, root, onnx_location=location) for location, file in _external_files(model, path)], "gpuOutputs": gpu_outputs}


def _validate_release(manifest: Path) -> None:
    root = manifest.parent
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    entries = [payload["graph"], payload["reducedHead"], *payload["graph"]["externalData"], *payload["reducedHead"]["externalData"], *payload["embedding"]["shards"], *payload["tokenizerFiles"], payload["licenseFile"]]
    for entry in entries:
        file = root / entry["path"]
        if not file.is_file() or file.stat().st_size != entry["bytes"] or _sha256(file) != entry["sha256"]:
            raise ValueError("release manifest reference is invalid")


def _file(path: Path, root: Path, onnx_location: str | None = None) -> dict:
    resolved = path.resolve()
    root = root.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("artifact path must be below release root")
    payload = {"path": resolved.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)}
    if onnx_location is not None:
        payload["onnxLocation"] = onnx_location
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
