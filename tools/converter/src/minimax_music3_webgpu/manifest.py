"""Release manifest construction with complete file integrity metadata."""

import hashlib
import json
from pathlib import Path
import shutil

import onnx

from .constants import ARTIFACT_FILE_LIMIT, DIFFUSERS_REVISION, MODEL_ID, MODEL_REVISION
from .paths import ArtifactPaths
from .embedding import export_embedding_table
from .reduced_head import export_reduced_head


def emit_manifest(
    path: Path, *, graph: Path, external_data: list[tuple[str, Path]],
    embedding_shards: list[tuple[int, int, Path]], tokenizer_files: list[tuple[str, Path]],
    license_file: Path, kv_pairs: list[tuple[str, str]],
) -> Path:
    entries = [("graph", graph), *external_data, *( ("embedding", file) for _, _, file in embedding_shards), *tokenizer_files, ("LICENSE", license_file)]
    _validate_files(entries)
    root = path.parent
    external = [_file(file, root, onnx_location=location) for location, file in external_data]
    payload = {
        "schemaVersion": 1,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "diffusersRevision": DIFFUSERS_REVISION},
        "quantization": {"bits": 4, "blockSize": 128, "accuracyLevel": 4, "symmetric": True},
        "webgpu": {"requiredFeatures": ["shader-f16"], "requiredLimits": {"maxStorageBufferBindingSize": ARTIFACT_FILE_LIMIT}},
        "graph": {**_file(graph, root), "externalData": external, "gpuOutputs": ["hidden_states", *[present for _, present in kv_pairs]]},
        "embedding": {"rows": 200000, "columns": 4096, "rowBytes": 8192, "shards": [{**_file(file, root), "rowStart": start, "rowCount": count} for start, count, file in embedding_shards]},
        "tokenizerFiles": [{**_file(file, root), "path": location} for location, file in tokenizer_files],
        "licenseFile": _file(license_file, root),
        "kvPairs": [{"pastInput": past, "presentOutput": present} for past, present in kv_pairs],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def emit_global_release(paths: ArtifactPaths, num_hidden_layers: int = 36) -> Path:
    if num_hidden_layers not in {1, 36}:
        raise ValueError("num_hidden_layers must be 1 or 36")
    release = paths.release / ("global-one-layer" if num_hidden_layers == 1 else "global")
    graph = release / "global_decoder.onnx"
    if not graph.is_file():
        raise FileNotFoundError(f"missing converted decoder: {graph}")
    embedding_source = _source_shard(paths.source / "language_model", "model.embed_tokens.weight")
    head_source = _source_shard(paths.source / "language_model", "lm_head.weight")
    embedding = export_embedding_table(embedding_source, release / "embedding")
    head = export_reduced_head(head_source, release / "reduced-head")
    _copy_tree(paths.source / "tokenizer", release / "tokenizer")
    shutil.copy2(paths.source / "LICENSE", release / "LICENSE")
    model = onnx.load_model(graph, load_external_data=False)
    locations = sorted({entry.value for tensor in model.graph.initializer for entry in tensor.external_data if entry.key == "location"})
    external = [(location, release / location) for location in locations]
    tokenizers = [(file.relative_to(release).as_posix(), file) for file in sorted((release / "tokenizer").rglob("*")) if file.is_file()]
    pairs = _kv_pairs(model)
    manifest = emit_manifest(release / "manifest.json", graph=graph, external_data=external,
                         embedding_shards=[(item.row_start, item.row_count, item.path) for item in embedding.shards],
                         tokenizer_files=tokenizers, license_file=release / "LICENSE", kv_pairs=pairs)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["reducedHead"] = _onnx_graph(head.model_path, release, ["semantic_logits", "end_logit"])
    _atomic_json(manifest, payload)
    _atomic_json(paths.receipts / f"global-release-{num_hidden_layers}.json", {"manifest": str(manifest), "reducedHead": str(head.model_path)})
    return manifest


def _source_shard(language_model: Path, tensor_name: str) -> Path:
    indexes = sorted(language_model.glob("*.safetensors.index.json"))
    if len(indexes) != 1:
        raise ValueError("language model must contain exactly one safetensors index")
    weight_map = json.loads(indexes[0].read_text(encoding="utf-8")).get("weight_map", {})
    name = weight_map.get(tensor_name)
    if not isinstance(name, str):
        raise ValueError(f"safetensors index is missing {tensor_name}")
    shard = language_model / name
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


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _kv_pairs(model: onnx.ModelProto) -> list[tuple[str, str]]:
    inputs = [item.name for item in model.graph.input if "past" in item.name]
    outputs = [item.name for item in model.graph.output if "present" in item.name]
    if len(inputs) != len(outputs):
        raise ValueError("decoder KV inputs and outputs do not pair")
    return list(zip(inputs, outputs, strict=True))


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
    locations = sorted({entry.value for tensor in model.graph.initializer for entry in tensor.external_data if entry.key == "location"})
    return {**_file(path, root), "externalData": [_file(path.parent / location, root, onnx_location=location) for location in locations], "gpuOutputs": gpu_outputs}


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
