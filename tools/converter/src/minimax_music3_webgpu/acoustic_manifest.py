"""Transactional assembly of fixed and variable browser releases."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Callable, Iterator
from uuid import uuid4

import onnx

from .condition_encoder import export_condition_encoder
from .constants import (
    ACOUSTIC_SOURCE_FILES,
    ARTIFACT_FILE_LIMIT,
    DIFFUSERS_REVISION,
    FLOW_FP16_LINEAR_WEIGHTS,
    MODEL_ID,
    MODEL_REVISION,
    Q4_ACCURACY_LEVEL,
    Q4_BITS,
    Q4_BLOCK_SIZE,
    Q4_SYMMETRIC,
)
from .flow_transformer import export_maximum_flow_step, open_flow_state, validate_maximum_flow_graph
from .paths import ArtifactPaths
from .manifest import _atomic_json
from .vocoder import (
    EXACT_FP32_SNAKES,
    MAXIMUM_VOCODER_CONFIG,
    MiniMaxMusic3VocoderMonoExport,
    load_vocoder_state_dict,
    prepare_vocoder_state_dict,
    publish_mono_vocoder_module,
    validate_dynamic_mono_vocoder_graph,
)


_RELEASES = ("global", "rvq", "condition", "flow", "vocoder")
_MODEL = {
    "id": MODEL_ID,
    "revision": MODEL_REVISION,
    "diffusersRevision": DIFFUSERS_REVISION,
}
_SLICE = {
    "semanticFrames": 125,
    "latentLength": 430,
    "outputSamples": 220160,
    "sampleRate": 44100,
    "channels": 2,
    "flowSteps": 30,
    "globalGuidance": 1.5,
    "flowGuidance": 1.7,
}
_ACOUSTIC = {
    "maxSemanticFrames": 200,
    "windowFrames": 200,
    "hopFrames": 100,
    "overlapLatents": 172,
    "leftCrop": 86,
    "rightCrop": 258,
    "samplesPerLatent": 512,
    "maxLatentLength": 689,
    "flowSteps": 30,
    "flowGuidance": 1.7,
}
_CONDITION_INPUTS = [
    {"name": "frame_hiddens", "dtype": "float16", "shape": [1, 200, 32768]},
    {"name": "nearest_index", "dtype": "int64", "shape": [689]},
    {"name": "active_latent_mask", "dtype": "float16", "shape": [1, 689, 1]},
]
_FLOW_INPUTS = [
    {"name": "latents", "dtype": "float16", "shape": [1, 128, 689]},
    {"name": "condition", "dtype": "float16", "shape": [1, 689, 2048]},
    {"name": "timestep", "dtype": "float16", "shape": [1]},
    {"name": "dt", "dtype": "float32", "shape": [1]},
    {"name": "active_latent_mask", "dtype": "float16", "shape": [1, 689, 1]},
    {"name": "key_attention_bias", "dtype": "float16", "shape": [1, 1, 1, 690]},
    {"name": "noise_prompt", "dtype": "float16", "shape": [1, 128, 172]},
    {"name": "previous_latent", "dtype": "float16", "shape": [1, 128, 172]},
    {"name": "overlap_enabled", "dtype": "float16", "shape": [1]},
    {"name": "guidance", "dtype": "float16", "shape": [1]},
]
_VOCODER_INPUTS = [
    {"name": "latents", "dtype": "float16", "shape": [1, 64, "L"], "maxShape": [1, 64, 689]}
]
_VOCODER_OUTPUTS = [
    {"name": "waveform", "dtype": "float32", "shape": [1, 1, "512L"], "maxShape": [1, 1, 352768]}
]


@dataclass(frozen=True)
class BuiltAcousticGraphs:
    condition_encoder: Path
    flow: Path
    vocoder: Path
    fp32_snakes: tuple[str, ...]


def build_music_5s_release(paths: ArtifactPaths) -> Path:
    manifests = {name: _read_manifest(paths.release / name) for name in _RELEASES}
    _validate_contracts(manifests)
    staging = paths.release / f".music-5s-{uuid4().hex}.staging"
    release = paths.release / "music-5s"
    try:
        staging.mkdir(parents=True)
        copied: set[str] = set()

        def artifact(name: str, value: dict) -> dict:
            result = dict(value)
            result["path"] = _copy_artifact(
                paths.release / name,
                staging,
                name,
                value,
                copied,
            )
            return result

        def graph(name: str, value: dict) -> dict:
            return {
                **artifact(name, value),
                "externalData": [artifact(name, item) for item in value["externalData"]],
                "gpuOutputs": list(value["gpuOutputs"]),
            }

        global_manifest = manifests["global"]
        rvq_manifest = manifests["rvq"]
        condition_manifest = manifests["condition"]
        flow_manifest = manifests["flow"]
        vocoder_manifest = manifests["vocoder"]
        limits: dict[str, int] = {}
        for manifest in manifests.values():
            for key, value in manifest["webgpu"]["requiredLimits"].items():
                limits[key] = max(limits.get(key, 0), value)
        payload = {
            "schemaVersion": 1,
            "model": _MODEL,
            "quantization": flow_manifest["quantization"],
            "precision": {
                **vocoder_manifest["precision"],
                "flowFp16Weights": list(FLOW_FP16_LINEAR_WEIGHTS),
            },
            "webgpu": {"requiredFeatures": ["shader-f16"], "requiredLimits": limits},
            "slice": _SLICE,
            "graph": graph("global", global_manifest["graph"]),
            "reducedHead": graph("global", global_manifest["reducedHead"]),
            "embedding": {
                **global_manifest["embedding"],
                "shards": [artifact("global", item) for item in global_manifest["embedding"]["shards"]],
            },
            "tokenizerFiles": [artifact("global", item) for item in global_manifest["tokenizerFiles"]],
            "licenseFile": artifact("global", global_manifest["licenseFile"]),
            "kvPairs": global_manifest["kvPairs"],
            "rvqDepth": graph("rvq", rvq_manifest["rvqDepth"]),
            "rvqEmbedding": {
                **rvq_manifest["rvqEmbedding"],
                "shards": [artifact("rvq", item) for item in rvq_manifest["rvqEmbedding"]["shards"]],
            },
            "feedback": graph("rvq", rvq_manifest["feedback"]),
            "conditionEncoder": graph("condition", condition_manifest["conditionEncoder"]),
            "flow": graph("flow", flow_manifest["flow"]),
            "vocoder": graph("vocoder", vocoder_manifest["vocoder"]),
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _validate_assembled(manifest_path, payload)
        if release.exists():
            _archive_music_5s_release(paths, release)
        _promote(staging, release)
        return release / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_music_variable_release(
    paths: ArtifactPaths,
    *,
    _build_graphs: Callable[[Path], BuiltAcousticGraphs] | None = None,
    _preflight: Callable[[ArtifactPaths], None] | None = None,
    _validate_components: Callable[[BuiltAcousticGraphs], None] | None = None,
) -> Path:
    preflight = _preflight_large_build if _preflight is None else _preflight
    validate_components = _validate_built_components if _validate_components is None else _validate_components
    staging = paths.release / f".music-variable-{uuid4().hex}.staging"
    release = paths.release / "music-variable"
    receipt = paths.receipts / "music-variable.json"
    with _exclusive_build_lock(paths.release / ".music-variable.build.lock"):
        try:
            preflight(paths)
            staging.mkdir(parents=True)
            prior_manifest = (
                _validate_prior_release(release, receipt) if release.exists() else None
            )
            reused_prior_components = set()
            if _build_graphs is None and prior_manifest is not None:
                fingerprints = _acoustic_build_fingerprints()
                reused_prior_components = {
                    name
                    for name in ("conditionEncoder", "flow", "vocoder")
                    if _prior_component_matches(prior_manifest, name, fingerprints[name])
                }
            built = (
                _build_pinned_acoustic_graphs(
                    staging,
                    release if prior_manifest is not None else None,
                    prior_manifest,
                    validate_components,
                )
                if _build_graphs is None
                else _build_graphs(staging)
            )
            condition_external = _validate_external_initializers(built.condition_encoder)
            flow_external = _validate_external_initializers(built.flow)
            vocoder_external = _validate_external_initializers(built.vocoder)
            if _build_graphs is not None:
                validate_components(built)
            manifests = {name: _read_manifest(paths.release / name) for name in ("global", "rvq")}
            _validate_variable_contracts(manifests)
            if prior_manifest is not None:
                _reuse_prior_acoustic_artifacts(
                    release,
                    prior_manifest,
                    built,
                    condition_external,
                    flow_external,
                    vocoder_external,
                    skip=reused_prior_components,
                )
            copied: set[str] = set()

            def artifact(name: str, value: dict) -> dict:
                result = dict(value)
                result["path"] = _copy_artifact(
                    paths.release / name, staging, name, value, copied, use_hardlink=True
                )
                return result

            def reused_graph(name: str, value: dict) -> dict:
                return {
                    **artifact(name, value),
                    "externalData": [artifact(name, item) for item in value["externalData"]],
                    "gpuOutputs": list(value["gpuOutputs"]),
                }

            global_manifest = manifests["global"]
            rvq_manifest = manifests["rvq"]
            limits = {"maxStorageBufferBindingSize": ARTIFACT_FILE_LIMIT}
            for manifest in manifests.values():
                for key, value in manifest["webgpu"]["requiredLimits"].items():
                    limits[key] = max(limits.get(key, 0), value)
            limits["maxStorageBuffersPerShaderStage"] = max(
                limits.get("maxStorageBuffersPerShaderStage", 0), 9
            )
            payload = {
                "schemaVersion": 1,
                "model": _MODEL,
                "quantization": global_manifest["quantization"],
                "precision": {
                    "convolution": "float16",
                    "fp32Snakes": list(built.fp32_snakes),
                    "flowFp16Weights": list(FLOW_FP16_LINEAR_WEIGHTS),
                },
                "webgpu": {"requiredFeatures": ["shader-f16"], "requiredLimits": limits},
                "acoustic": _ACOUSTIC,
                "graph": reused_graph("global", global_manifest["graph"]),
                "reducedHead": reused_graph("global", global_manifest["reducedHead"]),
                "embedding": {
                    **global_manifest["embedding"],
                    "shards": [artifact("global", item) for item in global_manifest["embedding"]["shards"]],
                },
                "tokenizerFiles": [artifact("global", item) for item in global_manifest["tokenizerFiles"]],
                "licenseFile": artifact("global", global_manifest["licenseFile"]),
                "kvPairs": global_manifest["kvPairs"],
                "rvqDepth": reused_graph("rvq", rvq_manifest["rvqDepth"]),
                "rvqEmbedding": {
                    **rvq_manifest["rvqEmbedding"],
                    "shards": [artifact("rvq", item) for item in rvq_manifest["rvqEmbedding"]["shards"]],
                },
                "feedback": reused_graph("rvq", rvq_manifest["feedback"]),
                "conditionEncoder": {
                    **_graph_artifact(staging, built.condition_encoder, condition_external, ["condition"]),
                    "inputs": _CONDITION_INPUTS,
                    "buildFingerprint": _acoustic_build_fingerprints()["conditionEncoder"],
                },
                "flow": {
                    **_graph_artifact(staging, built.flow, flow_external, ["next_latents"]),
                    "inputs": _FLOW_INPUTS,
                    "buildFingerprint": _acoustic_build_fingerprints()["flow"],
                },
                "vocoder": {
                    **_graph_artifact(staging, built.vocoder, vocoder_external, []),
                    "inputs": _VOCODER_INPUTS,
                    "outputs": _VOCODER_OUTPUTS,
                    "buildFingerprint": _acoustic_build_fingerprints()["vocoder"],
                },
            }
            manifest_path = staging / "manifest.json"
            manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            _validate_assembled(manifest_path, payload)
            receipt_payload = {
                "release": "music-variable",
                "manifest": "release/music-variable/manifest.json",
                "bytes": manifest_path.stat().st_size,
                "sha256": _sha256(manifest_path),
            }
            _promote_variable_release(
                paths, staging, release, receipt, receipt_payload
            )
            return release / "manifest.json"
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def _validate_variable_contracts(manifests: dict[str, dict]) -> None:
    if manifests["global"].get("quantization") != {
        "bits": Q4_BITS,
        "blockSize": Q4_BLOCK_SIZE,
        "accuracyLevel": Q4_ACCURACY_LEVEL,
        "symmetric": Q4_SYMMETRIC,
    }:
        raise ValueError("Global release quantization contract is invalid")


def _validate_prior_release(release: Path, receipt: Path) -> dict:
    if not receipt.is_file():
        raise ValueError("existing music-variable release has no receipt")
    payload = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    _validate_assembled(release / "manifest.json", payload)
    _validate_release_receipt(release, receipt)
    return payload


def _validate_release_receipt(release: Path, receipt: Path) -> None:
    value = json.loads(receipt.read_text(encoding="utf-8"))
    manifest = release / "manifest.json"
    if (
        value.get("release") != "music-variable"
        or value.get("manifest") != "release/music-variable/manifest.json"
        or not manifest.is_file()
        or value.get("bytes") != manifest.stat().st_size
        or value.get("sha256") != _sha256(manifest)
    ):
        raise ValueError("music-variable release receipt is incoherent")


def _reuse_prior_acoustic_artifacts(
    release: Path,
    prior: dict,
    built: BuiltAcousticGraphs,
    condition_external: list[tuple[str, Path]],
    flow_external: list[tuple[str, Path]],
    vocoder_external: list[tuple[str, Path]],
    *,
    skip: set[str] | frozenset[str] = frozenset(),
) -> None:
    if "conditionEncoder" not in skip:
        _reuse_graph_artifacts(
            release,
            prior["conditionEncoder"],
            built.condition_encoder,
            condition_external,
            reuse_graph=True,
        )
    if "vocoder" not in skip:
        _reuse_graph_artifacts(
            release,
            prior["vocoder"],
            built.vocoder,
            vocoder_external,
            reuse_graph=True,
        )
    if "flow" not in skip:
        _reuse_graph_artifacts(
            release,
            prior["flow"],
            built.flow,
            flow_external,
            reuse_graph=False,
        )


def _reuse_graph_artifacts(
    prior_root: Path,
    prior_graph: dict,
    graph: Path,
    external: list[tuple[str, Path]],
    *,
    reuse_graph: bool,
) -> None:
    if reuse_graph:
        _reuse_file_if_unchanged(prior_root, prior_graph, graph)
    available = list(prior_graph["externalData"])
    for _, destination in external:
        destination_size = destination.stat().st_size
        destination_sha = _sha256(destination)
        for index, entry in enumerate(available):
            if entry.get("bytes") == destination_size and entry.get("sha256") == destination_sha:
                source = _verified_manifest_file(prior_root, entry)
                destination.unlink()
                _hardlink_or_copy(source, destination)
                if destination.stat().st_size != destination_size or _sha256(destination) != destination_sha:
                    raise ValueError("reused acoustic artifact integrity failed")
                available.pop(index)
                break


def _reuse_file_if_unchanged(prior_root: Path, entry: dict, destination: Path) -> None:
    source = _verified_manifest_file(prior_root, entry)
    if (
        destination.stat().st_size != entry["bytes"]
        or _sha256(destination) != entry["sha256"]
    ):
        return
    destination.unlink()
    _hardlink_or_copy(source, destination)
    if destination.stat().st_size != entry["bytes"] or _sha256(destination) != entry["sha256"]:
        raise ValueError("reused acoustic artifact integrity failed")


def _verified_manifest_file(root: Path, entry: dict) -> Path:
    relative = _safe_relative(entry.get("path"))
    path = (root / relative).resolve()
    if (
        not path.is_relative_to(root.resolve())
        or not path.is_file()
        or path.stat().st_size != entry.get("bytes")
        or _sha256(path) != entry.get("sha256")
    ):
        raise ValueError("prior release artifact integrity failed")
    return path


def _promote_variable_release(
    paths: ArtifactPaths,
    staging: Path,
    release: Path,
    receipt: Path,
    receipt_payload: dict,
) -> None:
    archive = _archive_prior_release(paths, release, receipt) if release.exists() else None
    backup = release.with_name(f".{release.name}-{uuid4().hex}.backup")
    moved_old = False
    published = False
    preserve_backup = False
    try:
        if release.exists():
            release.replace(backup)
            moved_old = True
        staging.replace(release)
        published = True
        _atomic_json(receipt, receipt_payload)
        _validate_release_receipt(release, receipt)
    except Exception:
        if published and release.exists():
            shutil.rmtree(release, ignore_errors=True)
        if moved_old and backup.exists():
            try:
                backup.replace(release)
            except Exception:
                preserve_backup = True
        if archive is not None:
            try:
                _restore_archived_receipt(archive / "receipt.json", receipt)
            except Exception:
                pass
        raise
    finally:
        if backup.exists() and not preserve_backup:
            shutil.rmtree(backup, ignore_errors=True)


def _archive_prior_release(paths: ArtifactPaths, release: Path, receipt: Path) -> Path:
    _validate_prior_release(release, receipt)
    archive_root = paths.root / "archive" / "music-variable"
    paths.validate_write_targets(archive_root)
    generation = uuid4().hex
    staging = archive_root / f".{generation}.staging"
    archive = archive_root / generation
    try:
        release_archive = staging / "release"
        for source in sorted(release.rglob("*")):
            if not source.is_file():
                continue
            destination = release_archive / source.relative_to(release)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _hardlink_or_copy(source, destination)
        staging.mkdir(parents=True, exist_ok=True)
        _hardlink_or_copy(receipt, staging / "receipt.json")
        if _release_tree_hashes(release_archive) != _release_tree_hashes(release):
            raise ValueError("archived music-variable release integrity failed")
        archived_manifest = json.loads(
            (release_archive / "manifest.json").read_text(encoding="utf-8")
        )
        _validate_assembled(release_archive / "manifest.json", archived_manifest)
        _validate_release_receipt(release_archive, staging / "receipt.json")
        archive_root.mkdir(parents=True, exist_ok=True)
        staging.replace(archive)
        return archive
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _archive_music_5s_release(paths: ArtifactPaths, release: Path) -> Path:
    manifest_path = release / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_assembled(manifest_path, payload)
    archive_root = paths.root / "archive" / "music-5s"
    paths.validate_write_targets(archive_root)
    generation = uuid4().hex
    staging = archive_root / f".{generation}.staging"
    archive = archive_root / generation
    try:
        release_archive = staging / "release"
        for source in sorted(release.rglob("*")):
            if not source.is_file():
                continue
            destination = release_archive / source.relative_to(release)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _hardlink_or_copy(source, destination)
        if _release_tree_hashes(release_archive) != _release_tree_hashes(release):
            raise ValueError("archived music-5s release integrity failed")
        archived_manifest = release_archive / "manifest.json"
        _validate_assembled(
            archived_manifest,
            json.loads(archived_manifest.read_text(encoding="utf-8")),
        )
        archive_root.mkdir(parents=True, exist_ok=True)
        staging.replace(archive)
        return archive
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _release_tree_hashes(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, _sha256(path))
        for path in root.rglob("*")
        if path.is_file()
    }


def _restore_archived_receipt(source: Path, receipt: Path) -> None:
    temporary = receipt.with_name(f".{receipt.name}.{uuid4().hex}.restore")
    try:
        _hardlink_or_copy(source, temporary)
        temporary.replace(receipt)
    finally:
        temporary.unlink(missing_ok=True)


def _build_pinned_acoustic_graphs(
    staging: Path,
    prior_root: Path | None = None,
    prior: dict | None = None,
    validate_components: Callable[[BuiltAcousticGraphs], None] | None = None,
) -> BuiltAcousticGraphs:
    validator = _validate_built_components if validate_components is None else validate_components
    fingerprints = _acoustic_build_fingerprints()
    source_root = staging.parents[1] / "source"
    reuse_condition = _prior_component_matches(
        prior, "conditionEncoder", fingerprints["conditionEncoder"]
    )
    if reuse_condition:
        condition_path = _verified_manifest_file(prior_root, prior["conditionEncoder"])
    else:
        condition = export_condition_encoder(
            source_root / "condition_encoder" / "diffusion_pytorch_model.safetensors",
            staging / "condition",
            frame_count=200,
            latent_length=689,
            maximum_window=True,
        )
        condition_path = condition.model_path

    reuse_flow = _prior_component_matches(prior, "flow", fingerprints["flow"])
    if reuse_flow:
        flow = _verified_manifest_file(prior_root, prior["flow"])
    else:
        with open_flow_state(source_root / "transformer") as state:
            flow = export_maximum_flow_step(
                state, staging / "flow" / "flow.onnx", external_data=True
            )

    reuse_vocoder = _prior_component_matches(prior, "vocoder", fingerprints["vocoder"])
    if reuse_vocoder:
        vocoder = _verified_manifest_file(prior_root, prior["vocoder"])
        fp32_snakes = tuple(prior["precision"]["fp32Snakes"])
    else:
        vocoder_state = load_vocoder_state_dict(
            source_root / "vocoder" / "diffusion_pytorch_model.safetensors"
        )
        prepared, fp32_snakes = prepare_vocoder_state_dict(vocoder_state)
        module = MiniMaxMusic3VocoderMonoExport.from_prepared_state(
            MAXIMUM_VOCODER_CONFIG, prepared, fp32_snakes
        )
        vocoder_result = publish_mono_vocoder_module(module, staging / "vocoder")
        vocoder = vocoder_result.model_path
    candidate = BuiltAcousticGraphs(condition_path, flow, vocoder, fp32_snakes)
    _validate_external_initializers(candidate.condition_encoder)
    _validate_external_initializers(candidate.flow)
    _validate_external_initializers(candidate.vocoder)
    validator(candidate)
    if reuse_condition:
        condition_path = _stage_prior_graph(prior_root, prior["conditionEncoder"], staging)
    if reuse_flow:
        flow = _stage_prior_graph(prior_root, prior["flow"], staging)
    if reuse_vocoder:
        vocoder = _stage_prior_graph(prior_root, prior["vocoder"], staging)
    return BuiltAcousticGraphs(condition_path, flow, vocoder, fp32_snakes)


def _acoustic_build_fingerprints() -> dict[str, str]:
    contracts = {
        "conditionEncoder": {"inputs": _CONDITION_INPUTS, "gpuOutputs": ["condition"]},
        "flow": {"inputs": _FLOW_INPUTS, "gpuOutputs": ["next_latents"]},
        "vocoder": {
            "inputs": _VOCODER_INPUTS,
            "outputs": _VOCODER_OUTPUTS,
            "gpuOutputs": [],
            "fp32Snakes": list(EXACT_FP32_SNAKES),
        },
    }
    source_files = {
        "conditionEncoder": Path(__file__).with_name("condition_encoder.py"),
        "flow": Path(__file__).with_name("flow_transformer.py"),
        "vocoder": Path(__file__).with_name("vocoder.py"),
    }
    result = {}
    for name, contract in contracts.items():
        digest = hashlib.sha256()
        digest.update(source_files[name].read_bytes())
        digest.update(
            json.dumps(
                {"model": _MODEL, "acoustic": _ACOUSTIC, "contract": contract},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        result[name] = digest.hexdigest()
    return result


def _prior_component_matches(prior: dict | None, name: str, fingerprint: str) -> bool:
    if (
        prior is None
        or prior.get("schemaVersion") != 1
        or prior.get("model") != _MODEL
        or prior.get("acoustic") != _ACOUSTIC
        or prior.get("webgpu", {}).get("requiredFeatures") != ["shader-f16"]
    ):
        return False
    entry = prior.get(name)
    if not isinstance(entry, dict) or entry.get("buildFingerprint") != fingerprint:
        return False
    expected = {
        "conditionEncoder": (_CONDITION_INPUTS, None, ["condition"]),
        "flow": (_FLOW_INPUTS, None, ["next_latents"]),
        "vocoder": (_VOCODER_INPUTS, _VOCODER_OUTPUTS, []),
    }[name]
    if entry.get("inputs") != expected[0] or entry.get("gpuOutputs") != expected[2]:
        return False
    if expected[1] is not None and entry.get("outputs") != expected[1]:
        return False
    if name != "vocoder":
        return True
    precision = prior.get("precision", {})
    return precision.get("convolution") == "float16" and precision.get(
        "fp32Snakes"
    ) == list(EXACT_FP32_SNAKES)


def _stage_prior_graph(prior_root: Path | None, entry: dict, staging: Path) -> Path:
    if prior_root is None:
        raise ValueError("prior acoustic release root is missing")
    graph_source = _verified_manifest_file(prior_root, entry)
    graph_destination = staging / _safe_relative(entry.get("path"))
    graph_destination.parent.mkdir(parents=True, exist_ok=True)
    _copy_prior_acoustic_artifact(graph_source, graph_destination, entry)
    copied = {entry["path"]}
    for external in entry.get("externalData", []):
        relative = _safe_relative(external.get("path"))
        if relative.as_posix() in copied:
            continue
        source = _verified_manifest_file(prior_root, external)
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_prior_acoustic_artifact(source, destination, external)
        copied.add(relative.as_posix())
    return graph_destination


def _copy_prior_acoustic_artifact(source: Path, destination: Path, entry: dict) -> None:
    shutil.copy2(source, destination)
    if (
        destination.stat().st_size != entry.get("bytes")
        or _sha256(destination) != entry.get("sha256")
    ):
        raise ValueError("copied prior acoustic artifact integrity failed")


def _validate_built_components(graphs: BuiltAcousticGraphs) -> None:
    if graphs.fp32_snakes != EXACT_FP32_SNAKES:
        raise ValueError("vocoder FP32 Snake contract is invalid")
    _validate_condition_contract(graphs.condition_encoder)
    validate_maximum_flow_graph(graphs.flow)
    validate_dynamic_mono_vocoder_graph(
        graphs.vocoder, fp32_snakes=graphs.fp32_snakes
    )


def _validate_condition_contract(path: Path) -> None:
    onnx.checker.check_model(str(path), full_check=True)
    model = onnx.load_model(path, load_external_data=False)
    _validate_values(
        model,
        inputs=[
            ("frame_hiddens", onnx.TensorProto.FLOAT16, [1, 200, 32768]),
            ("nearest_index", onnx.TensorProto.INT64, [689]),
            ("active_latent_mask", onnx.TensorProto.FLOAT16, [1, 689, 1]),
        ],
        outputs=[("condition", onnx.TensorProto.FLOAT16, [1, 689, 2048])],
    )


def _validate_values(model: onnx.ModelProto, *, inputs: list, outputs: list) -> None:
    actual_inputs = {value.name: value for value in model.graph.input}
    actual_outputs = {value.name: value for value in model.graph.output}
    if set(actual_inputs) != {name for name, _, _ in inputs} or set(actual_outputs) != {
        name for name, _, _ in outputs
    }:
        raise ValueError("variable graph inputs or outputs do not match the runtime contract")
    for name, dtype, shape in [*inputs, *outputs]:
        value = actual_inputs[name] if name in actual_inputs else actual_outputs[name]
        tensor_type = value.type.tensor_type
        actual_shape = [dimension.dim_param or dimension.dim_value for dimension in tensor_type.shape.dim]
        if tensor_type.elem_type != dtype or actual_shape != shape:
            raise ValueError(f"variable graph value {name} does not match the runtime contract")


def _validate_external_initializers(path: Path) -> list[tuple[str, Path]]:
    onnx.checker.check_model(str(path))
    model = onnx.load_model(path, load_external_data=False)
    ranges: dict[str, list[tuple[int, int]]] = {}
    files: dict[str, Path] = {}
    for tensor in model.graph.initializer:
        fields = {field.key: field.value for field in tensor.external_data}
        if not fields:
            if len(tensor.raw_data) > ARTIFACT_FILE_LIMIT:
                raise ValueError("inline initializer exceeds the binding limit")
            continue
        try:
            relative = _safe_relative(fields["location"])
            offset = int(fields["offset"])
            length = int(fields["length"])
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError("external initializer range is invalid") from error
        expected = math.prod(tensor.dims) * onnx.helper.tensor_dtype_to_np_dtype(
            tensor.data_type
        ).itemsize
        source = (path.parent / relative).resolve()
        if (
            offset < 0
            or length <= 0
            or length != expected
            or length > ARTIFACT_FILE_LIMIT
            or not source.is_relative_to(path.parent.resolve())
            or not source.is_file()
        ):
            raise ValueError("external initializer range is invalid")
        location = relative.as_posix()
        files[location] = source
        ranges.setdefault(location, []).append((offset, offset + length))
    for location, intervals in ranges.items():
        intervals.sort()
        if intervals[0][0] != 0 or any(
            current[0] != previous[1]
            for previous, current in zip(intervals, intervals[1:])
        ) or intervals[-1][1] != files[location].stat().st_size:
            raise ValueError("external initializer ranges overlap or do not cover the exact file size")
    return sorted(files.items())


def _graph_artifact(
    root: Path, path: Path, external: list[tuple[str, Path]], gpu_outputs: list[str]
) -> dict:
    return {
        **_file_metadata(path, root),
        "externalData": [
            {**_file_metadata(file, root), "onnxLocation": location}
            for location, file in external
        ],
        "gpuOutputs": gpu_outputs,
    }


def _file_metadata(path: Path, root: Path) -> dict:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("staged artifact path is invalid")
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _preflight_large_build(paths: ArtifactPaths) -> None:
    receipt = _verify_acoustic_source(paths)
    source_bytes = sum(item["size"] for item in receipt["files"])
    reused_bytes = sum(
        path.stat().st_size
        for name in ("global", "rvq")
        for path in (paths.release / name).rglob("*")
        if path.is_file()
    )
    required = max(4 * 1024**3, source_bytes * 2) + reused_bytes
    if shutil.disk_usage(paths.root).free < required:
        raise OSError(f"music-variable build requires at least {required} free bytes")
    profiles = _target_chrome_profiles(paths)
    if any(
        os.path.lexists(profile / marker)
        for profile in profiles
        for marker in ("SingletonLock", "SingletonCookie", "SingletonSocket")
    ):
        raise RuntimeError("Chrome target profile is locked")
    _reject_conflicting_processes(
        repository_root=Path.cwd().resolve(), target_profiles=profiles
    )


def _target_chrome_profiles(paths: ArtifactPaths) -> tuple[Path, ...]:
    configured = os.environ.get("MINIMAX_VARIABLE_CHROME_PROFILE")
    profile = (
        Path(configured)
        if configured
        else paths.root / "browser-profiles" / "variable-duration" / "local"
    )
    return (profile.resolve(),)


def _verify_acoustic_source(paths: ArtifactPaths) -> dict:
    receipt_path = paths.receipts / "source-acoustic.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing acoustic source receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("repository_id") != MODEL_ID or receipt.get("revision") != MODEL_REVISION:
        raise ValueError("acoustic source receipt revisions are invalid")
    files = receipt.get("files")
    if not isinstance(files, list) or {item.get("path") for item in files} != set(ACOUSTIC_SOURCE_FILES):
        raise ValueError("acoustic source receipt file set is invalid")
    for item in files:
        source = paths.source_path(item["path"])
        if (
            not source.is_file()
            or source.stat().st_size != item.get("size")
            or _sha256(source) != item.get("sha256")
        ):
            raise ValueError(f"acoustic source receipt integrity failed: {item['path']}")
    return receipt


def _reject_conflicting_processes(
    processes: list[tuple[int, int, str, str]] | None = None,
    *,
    current_pid: int | None = None,
    repository_root: Path | None = None,
    target_profiles: tuple[Path, ...] = (),
) -> None:
    if processes is None and os.name == "nt":
        command = [
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress",
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout or "[]")
        items = [payload] if isinstance(payload, dict) else payload
        processes = [
            (
                int(item.get("ProcessId", -1)),
                int(item.get("ParentProcessId", -1)),
                str(item.get("Name", "")),
                str(item.get("CommandLine", "")),
            )
            for item in items
        ]
    elif processes is None:
        result = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,comm=,args="],
            check=True,
            capture_output=True,
            text=True,
        )
        processes = []
        for line in result.stdout.splitlines():
            fields = line.strip().split(maxsplit=3)
            if len(fields) >= 3 and fields[0].isdigit() and fields[1].isdigit():
                processes.append(
                    (int(fields[0]), int(fields[1]), fields[2], fields[3] if len(fields) == 4 else "")
                )
    active_pid = os.getpid() if current_pid is None else current_pid
    parents = {process_id: parent_id for process_id, parent_id, _, _ in processes}
    own_tree = set()
    cursor = active_pid
    while cursor > 0 and cursor not in own_tree:
        own_tree.add(cursor)
        cursor = parents.get(cursor, -1)
    chrome_targets = [
        str(path.resolve()).replace("\\", "/").casefold()
        for path in ((repository_root,) if repository_root is not None else ()) + target_profiles
    ]
    for process_id, _, name, command_line in processes:
        lowered = f"{name} {command_line}".lower()
        normalized_command = command_line.replace("\\", "/").casefold()
        if name.lower() in {"chrome", "chrome.exe"} and any(
            target in normalized_command for target in chrome_targets
        ):
            raise RuntimeError("Chrome is using the converter target")
        if process_id not in own_tree and (
            "music3-convert" in lowered or "minimax_music3_webgpu.cli" in lowered
        ):
            raise RuntimeError("another converter process is running")


@contextmanager
def _exclusive_build_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("another music-variable build is running") from error
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)


def _read_manifest(root: Path) -> dict:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing standalone release manifest: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schemaVersion") != 1 or value.get("model") != _MODEL:
        raise ValueError(f"{root.name} release does not use the pinned model contract")
    webgpu = value.get("webgpu", {})
    if webgpu.get("requiredFeatures") != ["shader-f16"]:
        raise ValueError(f"{root.name} release does not require shader-f16")
    return value


def _validate_contracts(manifests: dict[str, dict]) -> None:
    quantization = {
        "bits": Q4_BITS,
        "blockSize": Q4_BLOCK_SIZE,
        "accuracyLevel": Q4_ACCURACY_LEVEL,
        "symmetric": Q4_SYMMETRIC,
    }
    if manifests["global"].get("quantization") != quantization:
        raise ValueError("Global release quantization contract is invalid")
    if manifests["flow"].get("quantization") != quantization:
        raise ValueError("flow release quantization contract is invalid")
    if manifests["flow"].get("precision") != {
        "float16Weights": list(FLOW_FP16_LINEAR_WEIGHTS)
    }:
        raise ValueError("flow release precision contract is invalid")
    if manifests["flow"].get("slice") != {
        "semanticFrames": 125,
        "latentLength": 430,
        "flowSteps": 30,
        "flowGuidance": 1.7,
    }:
        raise ValueError("flow release slice contract is invalid")
    if manifests["vocoder"].get("slice") != {
        "latentChannels": 128,
        "latentLength": 430,
        "outputSamples": 220160,
        "sampleRate": 44100,
        "channels": 2,
    }:
        raise ValueError("vocoder release slice contract is invalid")


def _copy_artifact(
    source_root: Path,
    staging: Path,
    prefix: str,
    entry: dict,
    copied: set[str],
    *,
    use_hardlink: bool = False,
) -> str:
    relative = _safe_relative(entry.get("path"))
    source = (source_root / relative).resolve()
    if not source.is_relative_to(source_root.resolve()) or not source.is_file():
        raise ValueError("standalone artifact path is invalid")
    size = source.stat().st_size
    if size != entry.get("bytes") or size > ARTIFACT_FILE_LIMIT or _sha256(source) != entry.get("sha256"):
        raise ValueError(f"standalone artifact integrity failed: {prefix}/{relative.as_posix()}")
    destination_relative = Path(prefix) / relative
    destination_key = destination_relative.as_posix()
    if destination_key not in copied:
        destination = staging / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if use_hardlink:
            _hardlink_or_copy(source, destination)
        else:
            shutil.copy2(source, destination)
        copied.add(destination_key)
    return destination_key


def _hardlink_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError as error:
        unsupported = {
            errno.EXDEV,
            getattr(errno, "ENOTSUP", -1),
            getattr(errno, "EOPNOTSUPP", -1),
        }
        if error.errno not in unsupported and getattr(error, "winerror", None) not in {1, 50}:
            raise
        shutil.copy2(source, destination)


def _safe_relative(value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("artifact path is invalid")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("artifact path is invalid")
    return path


def _validate_assembled(path: Path, payload: dict) -> None:
    artifacts = [
        payload["graph"], payload["reducedHead"], payload["rvqDepth"], payload["feedback"],
        payload["conditionEncoder"], payload["flow"], payload["vocoder"],
        *payload["graph"]["externalData"], *payload["reducedHead"]["externalData"],
        *payload["rvqDepth"]["externalData"], *payload["feedback"]["externalData"],
        *payload["conditionEncoder"]["externalData"], *payload["flow"]["externalData"],
        *payload["vocoder"]["externalData"], *payload["embedding"]["shards"],
        *payload["rvqEmbedding"]["shards"], *payload["tokenizerFiles"], payload["licenseFile"],
    ]
    for entry in artifacts:
        file = path.parent / entry["path"]
        if (
            not file.is_file()
            or file.stat().st_size != entry["bytes"]
            or file.stat().st_size > ARTIFACT_FILE_LIMIT
            or _sha256(file) != entry["sha256"]
        ):
            raise ValueError("assembled release integrity validation failed")


def _promote(staging: Path, release: Path) -> None:
    backup = release.with_name(f".{release.name}-{uuid4().hex}.backup")
    moved = False
    preserve_backup = False
    try:
        if release.exists():
            release.replace(backup)
            moved = True
        staging.replace(release)
    except Exception:
        if moved and backup.exists():
            if release.exists():
                shutil.rmtree(release)
            try:
                backup.replace(release)
            except Exception:
                preserve_backup = True
        raise
    finally:
        if backup.exists() and not preserve_backup:
            shutil.rmtree(backup, ignore_errors=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
