"""Release manifest construction with complete file integrity metadata."""

import hashlib
import json
from pathlib import Path
import shutil
import re
from uuid import uuid4

import onnx

from .constants import ARTIFACT_FILE_LIMIT, DIFFUSERS_REVISION, HIDDEN_SIZE, MODEL_ID, MODEL_REVISION, VOCAB_SIZE
from .paths import ArtifactPaths
from .embedding import export_embedding_table
from .reduced_head import export_reduced_head


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
        "quantization": {"bits": 4, "blockSize": 128, "accuracyLevel": 4, "symmetric": True},
        "webgpu": {"requiredFeatures": ["shader-f16"], "requiredLimits": {"maxStorageBufferBindingSize": ARTIFACT_FILE_LIMIT}},
        "graph": {**_file(graph, root), "externalData": external, "gpuOutputs": gpu_outputs},
        "embedding": {"rows": 200000, "columns": 4096, "rowBytes": 8192, "shards": [{**_file(file, root), "rowStart": start, "rowCount": count} for start, count, file in embedding_shards]},
        "tokenizerFiles": [{**_file(file, root), "path": location} for location, file in tokenizer_files],
        "licenseFile": _file(license_file, root),
        "kvPairs": [{"pastInput": past, "presentOutput": present} for past, present in kv_pairs],
    }
    if reduced_head is not None:
        payload["reducedHead"] = _onnx_graph(reduced_head, root, ["semantic_logits", "end_logit"])
    _atomic_json(path, payload)
    return path


def emit_global_release(paths: ArtifactPaths, num_hidden_layers: int = 36) -> Path:
    if num_hidden_layers not in {1, 36}:
        raise ValueError("num_hidden_layers must be 1 or 36")
    name = "global-one-layer" if num_hidden_layers == 1 else "global"
    release = paths.release / name
    packed = paths.work / f"global-packed-{num_hidden_layers}"
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
        _promote_directory(staging, release)
        final = release / "manifest.json"
        _atomic_json(paths.receipts / f"global-release-{num_hidden_layers}.json", {"manifest": str(final), "reducedHead": str(release / head.model_path.relative_to(staging))})
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


def _promote_directory(staging: Path, release: Path) -> None:
    backup = release.with_name(f".{release.name}-{uuid4().hex}.backup")
    moved_old = False
    try:
        if release.exists():
            release.replace(backup)
            moved_old = True
        staging.replace(release)
    except Exception:
        if release.exists() and moved_old:
            shutil.rmtree(release, ignore_errors=True)
        if moved_old and backup.exists():
            backup.replace(release)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


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
