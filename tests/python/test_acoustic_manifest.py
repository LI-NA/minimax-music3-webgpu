import hashlib
import json
from pathlib import Path

import pytest

from minimax_music3_webgpu.acoustic_manifest import build_music_5s_release
from minimax_music3_webgpu.cli import main
from minimax_music3_webgpu.constants import DIFFUSERS_REVISION, MODEL_ID, MODEL_REVISION
from minimax_music3_webgpu.paths import ArtifactPaths


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file(root: Path, relative: str, data: bytes = b"fixture") -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": relative, "bytes": len(data), "sha256": _sha(path)}


def _graph(root: Path, relative: str, output: str) -> dict:
    graph = _file(root, relative, b"onnx")
    external_relative = f"{Path(relative).parent.as_posix()}/weights.bin"
    external = _file(root, external_relative, b"weights")
    external["onnxLocation"] = "weights.bin"
    return {**graph, "externalData": [external], "gpuOutputs": [output]}


def _base() -> dict:
    return {
        "schemaVersion": 1,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "diffusersRevision": DIFFUSERS_REVISION,
        },
        "webgpu": {
            "requiredFeatures": ["shader-f16"],
            "requiredLimits": {"maxStorageBufferBindingSize": 134217728},
        },
    }


def _standalone_releases(paths: ArtifactPaths) -> None:
    global_root = paths.release / "global"
    global_root.mkdir(parents=True)
    embedding = _file(global_root, "embedding/embedding.fp16", b"1234")
    embedding.update({"rowStart": 0, "rowCount": 1})
    tokenizer = _file(global_root, "tokenizer/tokenizer.json", b"{}")
    license_file = _file(global_root, "LICENSE", b"license")
    (global_root / "manifest.json").write_text(json.dumps({
        **_base(),
        "quantization": {"bits": 4, "blockSize": 128, "accuracyLevel": 4, "symmetric": True},
        "graph": _graph(global_root, "global.onnx", "hidden_states"),
        "reducedHead": _graph(global_root, "head/head.onnx", "last_state"),
        "embedding": {"rows": 1, "columns": 2, "rowBytes": 4, "shards": [embedding]},
        "tokenizerFiles": [tokenizer],
        "licenseFile": license_file,
        "kvPairs": [{"pastInput": "past.0", "presentOutput": "present.0"}],
    }))

    rvq_root = paths.release / "rvq"
    rvq_root.mkdir()
    rvq_embedding = _file(rvq_root, "embedding/embedding.fp16", b"1234")
    rvq_embedding.update({"rowStart": 0, "rowCount": 1})
    (rvq_root / "manifest.json").write_text(json.dumps({
        **_base(),
        "rvqDepth": _graph(rvq_root, "rvq/rvq.onnx", "depth_hidden"),
        "feedback": {**_file(rvq_root, "feedback.onnx", b"onnx"), "externalData": [], "gpuOutputs": ["inputs_embeds"]},
        "rvqEmbedding": {"rows": 1, "columns": 2, "rowBytes": 4, "shards": [rvq_embedding]},
    }))

    condition_root = paths.release / "condition"
    condition_root.mkdir()
    (condition_root / "manifest.json").write_text(json.dumps({
        **_base(),
        "conditionEncoder": _graph(condition_root, "condition/condition.onnx", "condition"),
    }))

    flow_root = paths.release / "flow"
    flow_root.mkdir()
    (flow_root / "manifest.json").write_text(json.dumps({
        **_base(),
        "quantization": {"bits": 4, "blockSize": 128, "accuracyLevel": 4, "symmetric": True},
        "slice": {"semanticFrames": 125, "latentLength": 430, "flowSteps": 30, "flowGuidance": 1.7},
        "flow": _graph(flow_root, "flow/flow.onnx", "next_latents"),
    }))

    vocoder_root = paths.release / "vocoder"
    vocoder_root.mkdir()
    (vocoder_root / "manifest.json").write_text(json.dumps({
        **_base(),
        "slice": {"latentChannels": 128, "latentLength": 430, "outputSamples": 220160, "sampleRate": 44100, "channels": 2},
        "precision": {"convolution": "float16", "fp32Snakes": ["blocks.0.snake1", "blocks.1.snake1"]},
        "vocoder": _graph(vocoder_root, "vocoder/vocoder.onnx", "waveform"),
    }))


def test_build_music_5s_copies_and_rehashes_every_exact_release_artifact(tmp_path: Path) -> None:
    paths = ArtifactPaths.from_root(tmp_path, repository_root=tmp_path)
    _standalone_releases(paths)

    manifest_path = build_music_5s_release(paths)
    manifest = json.loads(manifest_path.read_text())

    assert manifest["model"] == _base()["model"]
    assert manifest["slice"] == {
        "semanticFrames": 125,
        "latentLength": 430,
        "outputSamples": 220160,
        "sampleRate": 44100,
        "channels": 2,
        "flowSteps": 30,
        "globalGuidance": 1.5,
        "flowGuidance": 1.7,
    }
    assert manifest["graph"]["path"] == "global/global.onnx"
    assert manifest["rvqDepth"]["path"] == "rvq/rvq/rvq.onnx"
    assert manifest["conditionEncoder"]["path"] == "condition/condition/condition.onnx"
    assert manifest["flow"]["path"] == "flow/flow/flow.onnx"
    assert manifest["vocoder"]["path"] == "vocoder/vocoder/vocoder.onnx"
    assert manifest["tokenizerFiles"][0]["path"] == "global/tokenizer/tokenizer.json"
    assert manifest["licenseFile"]["path"] == "global/LICENSE"
    assert manifest["flow"]["externalData"][0]["onnxLocation"] == "weights.bin"

    artifacts = [
        manifest["graph"], manifest["reducedHead"], manifest["rvqDepth"],
        manifest["feedback"], manifest["conditionEncoder"], manifest["flow"], manifest["vocoder"],
        *manifest["graph"]["externalData"], *manifest["reducedHead"]["externalData"],
        *manifest["rvqDepth"]["externalData"], *manifest["feedback"]["externalData"],
        *manifest["conditionEncoder"]["externalData"], *manifest["flow"]["externalData"],
        *manifest["vocoder"]["externalData"], *manifest["embedding"]["shards"],
        *manifest["rvqEmbedding"]["shards"], *manifest["tokenizerFiles"], manifest["licenseFile"],
    ]
    for artifact in artifacts:
        path = manifest_path.parent / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert _sha(path) == artifact["sha256"]
        assert artifact["bytes"] <= 134217728


def test_build_music_5s_rejects_tampered_standalone_artifact_without_replacing_release(tmp_path: Path) -> None:
    paths = ArtifactPaths.from_root(tmp_path, repository_root=tmp_path)
    _standalone_releases(paths)
    existing = paths.release / "music-5s"
    existing.mkdir(parents=True)
    (existing / "sentinel.txt").write_text("keep")
    (paths.release / "flow" / "flow" / "weights.bin").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="integrity"):
        build_music_5s_release(paths)

    assert (existing / "sentinel.txt").read_text() == "keep"


def test_build_music_5s_cli_uses_requested_artifacts_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[ArtifactPaths] = []
    monkeypatch.setattr(
        "minimax_music3_webgpu.cli.build_music_5s_release",
        lambda paths: received.append(paths),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["minimax-music3-webgpu", "build-music-5s", "--artifacts-dir", "artifacts"],
    )

    main()

    assert received == [ArtifactPaths.from_root(Path("artifacts"))]
