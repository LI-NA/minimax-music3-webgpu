"""Transactional assembly of the fixed five-second browser release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

from .constants import ARTIFACT_FILE_LIMIT, DIFFUSERS_REVISION, MODEL_ID, MODEL_REVISION
from .paths import ArtifactPaths


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
            "precision": vocoder_manifest["precision"],
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
        _promote(staging, release)
        return release / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


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
    quantization = {"bits": 4, "blockSize": 128, "accuracyLevel": 4, "symmetric": True}
    if manifests["global"].get("quantization") != quantization:
        raise ValueError("Global release quantization contract is invalid")
    if manifests["flow"].get("quantization") != quantization:
        raise ValueError("flow release quantization contract is invalid")
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
        shutil.copy2(source, destination)
        copied.add(destination_key)
    return destination_key


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
    try:
        if release.exists():
            release.replace(backup)
            moved = True
        staging.replace(release)
    except Exception:
        if moved and backup.exists():
            if release.exists():
                shutil.rmtree(release)
            backup.replace(release)
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
