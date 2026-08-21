import builtins
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import pytest


REFERENCE_TOOLS = Path(__file__).parents[2] / "tools" / "reference"
sys.path.insert(0, str(REFERENCE_TOOLS))

import reference_case  # noqa: E402


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _write_wav(path: Path, frames: int, sample_rate: int = 44_100, channels: int = 2) -> Path:
    data_bytes = frames * channels * 2
    header = (
        b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, sample_rate * channels * 2, channels * 2, 16)
        + b"data" + struct.pack("<I", data_bytes)
    )
    path.write_bytes(header + bytes(data_bytes))
    return path


def _manifest() -> dict:
    return {
        "schemaVersion": 1,
        "model": {
            "id": "MiniMaxAI/MiniMax-Music3",
            "revision": "fbdf52fbaaca799592917417eb05f1899f1255ec",
            "diffusersRevision": "3681e65996b4d2589219720101a6acbfd25073f8",
        },
        "acoustic": {
            "maxSemanticFrames": 200, "windowFrames": 200, "hopFrames": 100,
            "overlapLatents": 172, "leftCrop": 86, "rightCrop": 258,
            "samplesPerLatent": 512, "maxLatentLength": 689,
            "flowSteps": 30, "flowGuidance": 1.7,
        },
    }


def _metrics(*, retained_frames: int = 250, termination: str = "max-frames") -> dict:
    fixture = json.loads((REFERENCE_TOOLS / "fixed_case.json").read_text(encoding="utf-8"))
    tokens = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "prompt-contract.json").read_text(encoding="utf-8")
    )
    return {
        "prompt": fixture["prompt"],
        "lyrics": fixture["lyrics"],
        "assembledPrompt": fixture["assembledPrompt"],
        "tokenIds": [tokens["conditional"], tokens["unconditional"]],
        "seed": 7, "durationSeconds": 10, "retainedFrames": retained_frames, "termination": termination,
        "globalGuidance": 1.5, "semanticTopK": 50, "residualTopK": 50, "temperature": 1.0,
        "flowGuidance": 1.7, "flowSteps": 30,
        "samplerRevision": fixture["samplerRevision"],
        "flowScheduleRevision": reference_case.FLOW_SCHEDULE_REVISION,
        "browser": "Mozilla/5.0 Chrome/140.0.7339.81 Safari/537.36",
        "ortVersion": "1.30.0-dev.20260813-72e1c9c9b8",
        "appVersion": "0.1.0-experimental",
        "appRevision": "a" * 64,
    }


def _metrics_with_manifest(manifest: Path, value: dict | None = None) -> dict:
    return {**(value or _metrics()), "manifestHash": hashlib.sha256(manifest.read_bytes()).hexdigest()}


def _paths(tmp_path: Path, metrics: dict | None = None, frames: int = 440_832):
    manifest = _write_json(tmp_path / "manifest.json", _manifest())
    return manifest, _write_json(tmp_path / "metrics.json", _metrics_with_manifest(manifest, metrics)), _write_wav(
        tmp_path / "webgpu.wav", frames
    )


def test_builds_exact_ten_second_case_with_planner_sized_canonical_wav(tmp_path: Path) -> None:
    manifest, metrics, wav = _paths(tmp_path)
    imports = []
    original_import = builtins.__import__

    def reject_model_import(name, *args, **kwargs):
        imports.append(name)
        if name == "torch" or name.startswith("diffusers"):
            raise AssertionError("cloud comparison tool imported a model runtime")
        return original_import(name, *args, **kwargs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(builtins, "__import__", reject_model_import)
        case = reference_case.create_case(tmp_path / "cases", "webgpu-ten-seconds", manifest, metrics, wav)
        receipt = reference_case.verify_case(case, manifest, metrics, wav)

    assert not any(name == "torch" or name.startswith("diffusers") for name in imports)
    assert receipt["input"]["prompt"] == _metrics()["prompt"]
    assert receipt["input"]["lyrics"] == _metrics()["lyrics"]
    assert receipt["input"]["promptTokenCount"] == 40
    assert receipt["generation"] == {
        "seed": 7, "durationSeconds": 10, "retainedFrames": 250, "termination": "max-frames",
    }
    assert receipt["audio"] == {
        "sampleRate": 44_100, "channels": 2, "samplesPerChannel": 440_832, "bytes": 1_763_372,
    }
    assert receipt["environment"] == {
        "browser": {
            "userAgent": _metrics()["browser"],
            "chromeVersion": "140.0.7339.81",
        },
        "runtime": {
            "onnxRuntimeWeb": _metrics()["ortVersion"],
            "appVersion": _metrics()["appVersion"],
            "appRevision": _metrics()["appRevision"],
        },
    }
    assert receipt["provenance"]["manifestSha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert list(case.iterdir()) == [case / "receipt.json"]
    fixture = json.loads((REFERENCE_TOOLS / "fixed_case.json").read_text(encoding="utf-8"))
    assert sorted(json.loads(metrics.read_text(encoding="utf-8"))) == sorted(fixture["comparisonMetricKeys"])


@pytest.mark.parametrize("manifest_hash", [None, "0" * 64])
def test_rejects_missing_or_mismatched_worker_manifest_hash(tmp_path: Path, manifest_hash: str | None) -> None:
    manifest = _write_json(tmp_path / "manifest.json", _manifest())
    metrics_value = _metrics()
    if manifest_hash is not None:
        metrics_value["manifestHash"] = manifest_hash
    metrics = _write_json(tmp_path / "metrics.json", metrics_value)
    wav = _write_wav(tmp_path / "webgpu.wav", 440_832)
    with pytest.raises(ValueError, match="manifestHash"):
        reference_case.build_receipt(manifest, metrics, wav)


def test_build_reads_each_attachment_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, metrics, wav = _paths(tmp_path)
    reads = {manifest: 0, metrics: 0, wav: 0}
    original_open = Path.open

    def count_attachment_open(path: Path, *args, **kwargs):
        if path in reads:
            reads[path] += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", count_attachment_open)
    reference_case.build_receipt(manifest, metrics, wav)
    assert reads == {manifest: 1, metrics: 1, wav: 1}


def test_natural_end_accepts_only_explicit_early_frame_counts_and_planner_sized_wav(tmp_path: Path) -> None:
    metrics_value = _metrics(retained_frames=201, termination="natural-end")
    manifest, metrics, wav = _paths(tmp_path, metrics_value, frames=354_304)
    case = reference_case.create_case(tmp_path / "cases", "natural-end", manifest, metrics, wav)
    assert reference_case.verify_case(case, manifest, metrics, wav)["audio"]["samplesPerChannel"] == 354_304

    for retained in (0, 250):
        _write_json(metrics, _metrics_with_manifest(manifest, _metrics(retained_frames=retained, termination="natural-end")))
        with pytest.raises(ValueError, match="natural-end"):
            reference_case.build_receipt(manifest, metrics, wav)
    _write_json(metrics, _metrics_with_manifest(manifest, _metrics(retained_frames=249, termination="max-frames")))
    with pytest.raises(ValueError, match="max-frames"):
        reference_case.build_receipt(manifest, metrics, wav)


@pytest.mark.parametrize(("field", "value"), [
    ("prompt", "other"), ("lyrics", "other"), ("assembledPrompt", "other"),
    ("tokenIds", [[0] * 40, [0] * 40]), ("seed", 8), ("durationSeconds", 5),
    ("durationSeconds", 10.0),
    ("globalGuidance", 1.6), ("semanticTopK", 49), ("residualTopK", 51),
    ("temperature", 0.9), ("flowGuidance", 1.8), ("flowSteps", 29),
    ("samplerRevision", "other"), ("flowScheduleRevision", "other"),
])
def test_rejects_every_non_fixed_design_field(tmp_path: Path, field: str, value) -> None:
    manifest, metrics, wav = _paths(tmp_path)
    _write_json(metrics, _metrics_with_manifest(manifest, {**_metrics(), field: value}))
    with pytest.raises(ValueError, match="fixed comparison contract|must be an integer"):
        reference_case.build_receipt(manifest, metrics, wav)


@pytest.mark.parametrize("field", [
    "maxSemanticFrames", "windowFrames", "hopFrames", "overlapLatents", "leftCrop",
    "rightCrop", "samplesPerLatent", "maxLatentLength", "flowSteps", "flowGuidance",
])
def test_rejects_every_wrong_variable_acoustic_constant(tmp_path: Path, field: str) -> None:
    manifest_value = _manifest()
    manifest_value["acoustic"][field] = 999
    manifest = _write_json(tmp_path / "manifest.json", manifest_value)
    metrics = _write_json(tmp_path / "metrics.json", _metrics_with_manifest(manifest))
    wav = _write_wav(tmp_path / "webgpu.wav", 440_832)
    with pytest.raises(ValueError, match="acoustic"):
        reference_case.build_receipt(manifest, metrics, wav)


@pytest.mark.parametrize(("field", "value"), [
    ("browser", "Mozilla/5.0 Firefox/140.0"),
    ("browser", "Mozilla/5.0 Chrome/140 Safari/537.36"),
    ("ortVersion", "1.23.0"),
    ("appVersion", "0.0.0"),
    ("appVersion", "dev"),
    ("appVersion", "1.0.0"),
    ("appRevision", "unknown"),
    ("appRevision", "0000000"),
])
def test_rejects_placeholder_or_unpinned_runtime_provenance(tmp_path: Path, field: str, value: str) -> None:
    manifest, metrics, wav = _paths(tmp_path)
    _write_json(metrics, _metrics_with_manifest(manifest, {**_metrics(), field: value}))
    with pytest.raises(ValueError, match="runtime provenance"):
        reference_case.build_receipt(manifest, metrics, wav)


def test_rejects_missing_or_extra_flat_comparison_field(tmp_path: Path) -> None:
    manifest, metrics, wav = _paths(tmp_path)
    missing = _metrics_with_manifest(manifest)
    del missing["temperature"]
    _write_json(metrics, missing)
    with pytest.raises(ValueError, match="metrics fields"):
        reference_case.build_receipt(manifest, metrics, wav)

    _write_json(metrics, {**_metrics_with_manifest(manifest), "extra": True})
    with pytest.raises(ValueError, match="metrics fields"):
        reference_case.build_receipt(manifest, metrics, wav)


@pytest.mark.parametrize(("rate", "channels", "frames"), [
    (48_000, 2, 440_832), (44_100, 1, 440_832),
    (44_100, 2, 440_831), (44_100, 2, 440_833),
])
def test_rejects_wrong_wav_rate_channels_or_planned_samples(tmp_path: Path, rate: int, channels: int, frames: int) -> None:
    manifest = _write_json(tmp_path / "manifest.json", _manifest())
    metrics = _write_json(tmp_path / "metrics.json", _metrics_with_manifest(manifest))
    wav = _write_wav(tmp_path / "webgpu.wav", frames, rate, channels)
    with pytest.raises(ValueError, match="canonical WAV|planned samples"):
        reference_case.build_receipt(manifest, metrics, wav)


@pytest.mark.parametrize("mutation", ["truncate", "extra", "data-size", "riff-size"])
def test_rejects_truncated_extra_or_inconsistent_wav_bytes(tmp_path: Path, mutation: str) -> None:
    manifest, metrics, wav = _paths(tmp_path)
    data = bytearray(wav.read_bytes())
    if mutation == "truncate": data.pop()
    elif mutation == "extra": data.append(0)
    elif mutation == "data-size": struct.pack_into("<I", data, 40, 1)
    else: struct.pack_into("<I", data, 4, 1)
    wav.write_bytes(data)
    with pytest.raises(ValueError, match="canonical WAV"):
        reference_case.build_receipt(manifest, metrics, wav)


def test_schedule_fixture_hashes_exact_timestep_and_dt_bits() -> None:
    fixture = json.loads((REFERENCE_TOOLS / "fixed_case.json").read_text(encoding="utf-8"))
    bits = fixture["timestepF32Bits"] + fixture["dtF32Bits"]
    raw = struct.pack(f"<{len(bits)}I", *bits)
    assert len(fixture["timestepF32Bits"]) == 30
    assert len(fixture["dtF32Bits"]) == 30
    assert fixture["flowScheduleRevision"] == f"sha256:{hashlib.sha256(raw).hexdigest()}"
    assert reference_case.FLOW_SCHEDULE_REVISION == fixture["flowScheduleRevision"]


def test_verify_rejects_symlinked_case_receipt_or_attachment(tmp_path: Path) -> None:
    manifest, metrics, wav = _paths(tmp_path)
    case = reference_case.create_case(tmp_path / "cases", "valid", manifest, metrics, wav)
    linked_wav = tmp_path / "linked.wav"
    try:
        linked_wav.symlink_to(wav)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="regular file"):
        reference_case.verify_case(case, manifest, metrics, linked_wav)

    real_receipt = case / "real.json"
    (case / "receipt.json").replace(real_receipt)
    (case / "receipt.json").symlink_to(real_receipt)
    with pytest.raises(ValueError, match="regular contained file"):
        reference_case.verify_case(case, manifest, metrics, wav)


def test_publication_is_atomic_and_cleans_failed_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, metrics, wav = _paths(tmp_path)
    output_root = tmp_path / "cases"

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr(reference_case.os, "replace", fail_replace)
    with pytest.raises(OSError, match="publication failure"):
        reference_case.create_case(output_root, "atomic-case", manifest, metrics, wav)
    assert not (output_root / "atomic-case").exists()
    assert list(output_root.iterdir()) == []


def test_verify_rejects_changed_hash_receipt(tmp_path: Path) -> None:
    manifest, metrics, wav = _paths(tmp_path)
    case = reference_case.create_case(tmp_path / "cases", "hash-case", manifest, metrics, wav)
    receipt_path = case / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["provenance"]["wavSha256"] = "0" * 64
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="WAV hash"):
        reference_case.verify_case(case, manifest, metrics, wav)
