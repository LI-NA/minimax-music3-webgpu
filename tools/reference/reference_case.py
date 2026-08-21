from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import struct
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence


MODEL_ID = "MiniMaxAI/MiniMax-Music3"
MODEL_REVISION = "fbdf52fbaaca799592917417eb05f1899f1255ec"
DIFFUSERS_REVISION = "3681e65996b4d2589219720101a6acbfd25073f8"
PINNED_ORT_VERSION = "1.30.0-dev.20260813-72e1c9c9b8"
FIXED_ACOUSTIC = {
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
_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_CHROME_IN_USER_AGENT = re.compile(r"(?:^| )Chrome/([1-9]\d*\.\d+\.\d+\.\d+)(?: |$)")
_APP_REVISION = re.compile(r"^[0-9a-f]{7,64}$")


def _load_fixed_case() -> dict[str, Any]:
    path = Path(__file__).with_name("fixed_case.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    timestep = value.get("timestepF32Bits")
    dt = value.get("dtF32Bits")
    if (
        value.get("schemaVersion") != 1
        or value.get("flowSteps") != 30
        or not isinstance(value.get("prompt"), str)
        or not isinstance(value.get("lyrics"), str)
        or not isinstance(value.get("assembledPrompt"), str)
        or not isinstance(value.get("samplerRevision"), str)
        or not isinstance(value.get("comparisonMetricKeys"), list)
        or len(value["comparisonMetricKeys"]) != len(set(value["comparisonMetricKeys"]))
    ):
        raise RuntimeError("fixed comparison schedule fixture is invalid")
    if not isinstance(timestep, list) or not isinstance(dt, list) or len(timestep) != 30 or len(dt) != 30:
        raise RuntimeError("fixed comparison schedule fixture is invalid")
    bits = timestep + dt
    if any(type(item) is not int or item < 0 or item > 0xFFFFFFFF for item in bits):
        raise RuntimeError("fixed comparison schedule fixture is invalid")
    raw = struct.pack(f"<{len(bits)}I", *bits)
    revision = f"sha256:{hashlib.sha256(raw).hexdigest()}"
    if value.get("flowScheduleRevision") != revision:
        raise RuntimeError("fixed comparison schedule fixture hash is invalid")
    return value


_FIXED_CASE = _load_fixed_case()
FIXED_PROMPT = _FIXED_CASE["prompt"]
FIXED_LYRICS = _FIXED_CASE["lyrics"]
FIXED_ASSEMBLED_PROMPT = _FIXED_CASE["assembledPrompt"]
FIXED_METRIC_KEYS = frozenset(_FIXED_CASE["comparisonMetricKeys"])
FLOW_SCHEDULE_REVISION = _FIXED_CASE["flowScheduleRevision"]
_PROMPT_CONTRACT = json.loads(
    (Path(__file__).parents[2] / "tests" / "fixtures" / "prompt-contract.json").read_text(encoding="utf-8")
)
FIXED_TOKEN_ROWS = [_PROMPT_CONTRACT["conditional"], _PROMPT_CONTRACT["unconditional"]]
if any(not isinstance(row, list) or len(row) != 40 for row in FIXED_TOKEN_ROWS):
    raise RuntimeError("fixed comparison prompt token fixture is invalid")
FIXED_SAMPLER = {
    "globalGuidance": _FIXED_CASE["globalGuidance"],
    "semanticTopK": _FIXED_CASE["semanticTopK"],
    "residualTopK": _FIXED_CASE["residualTopK"],
    "temperature": _FIXED_CASE["temperature"],
    "samplerRevision": _FIXED_CASE["samplerRevision"],
    "flowGuidance": _FIXED_CASE["flowGuidance"],
    "flowSteps": _FIXED_CASE["flowSteps"],
    "flowScheduleRevision": FLOW_SCHEDULE_REVISION,
}


def _load_duration_planner() -> Callable[..., dict[str, object]]:
    path = Path(__file__).parents[2] / "tools" / "converter" / "src" / "minimax_music3_webgpu" / "duration_plan.py"
    spec = importlib.util.spec_from_file_location("_reference_duration_plan", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("project duration planner is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plan_retained_frames


_PLAN_RETAINED_FRAMES = _load_duration_planner()


def _regular_file(path: Path, label: str) -> Path:
    try:
        mode = os.lstat(path).st_mode
    except OSError as error:
        raise ValueError(f"{label} must be a regular file") from error
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular file")
    return path


def _read_file(path: Path, label: str) -> bytes:
    _regular_file(path, label)
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"{label} could not be read") from error


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _parse_json(data: bytes, label: str) -> dict[str, Any]:
    try:
        return _object(json.loads(data.decode("utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return _parse_json(_read_file(path, label), label)


def _validate_manifest(manifest: dict[str, Any]) -> dict[str, str]:
    model = _object(manifest.get("model"), "manifest model")
    expected = {"id": MODEL_ID, "revision": MODEL_REVISION, "diffusersRevision": DIFFUSERS_REVISION}
    if model != expected:
        raise ValueError("manifest model provenance does not match the fixed comparison contract")
    acoustic = _object(manifest.get("acoustic"), "manifest acoustic")
    if acoustic != FIXED_ACOUSTIC:
        raise ValueError("manifest acoustic constants do not match the fixed comparison contract")
    return expected


def _validate_metrics(metrics: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if set(metrics) != FIXED_METRIC_KEYS:
        raise ValueError("generation metrics fields do not match the fixed comparison contract")
    prompt = _text(metrics.get("prompt"), "prompt")
    lyrics = _text(metrics.get("lyrics"), "lyrics")
    seed = _integer(metrics.get("seed"), "seed")
    duration = _integer(metrics.get("durationSeconds"), "durationSeconds")
    fixed = {
        "prompt": FIXED_PROMPT,
        "lyrics": FIXED_LYRICS,
        "seed": _FIXED_CASE["seed"],
        "durationSeconds": _FIXED_CASE["durationSeconds"],
    }
    actual = {"prompt": prompt, "lyrics": lyrics, "seed": seed, "durationSeconds": duration}
    if actual != fixed:
        raise ValueError("generation inputs do not match the fixed comparison contract")

    assembled = _text(metrics.get("assembledPrompt"), "assembledPrompt")
    rows = metrics.get("tokenIds")
    if assembled != FIXED_ASSEMBLED_PROMPT or rows != FIXED_TOKEN_ROWS:
        raise ValueError("assembled prompt and token IDs do not match the fixed comparison contract")
    input_data: dict[str, Any] = {
        "prompt": prompt,
        "lyrics": lyrics,
        "promptTokenCount": 40,
        "assembledPrompt": assembled,
        "tokenIds": rows,
    }

    retained = _integer(metrics.get("retainedFrames"), "retainedFrames")
    termination = metrics.get("termination")
    if termination == "max-frames" and retained != 250:
        raise ValueError("max-frames requires exactly 250 retained frames")
    if termination == "natural-end" and not 1 <= retained <= 249:
        raise ValueError("natural-end requires 1 through 249 retained frames")
    if termination not in ("max-frames", "natural-end"):
        raise ValueError("termination must explicitly be max-frames or natural-end")
    generation = {"seed": seed, "durationSeconds": duration, "retainedFrames": retained, "termination": termination}

    if any(metrics.get(key) != value for key, value in FIXED_SAMPLER.items()):
        raise ValueError("sampler settings do not match the fixed comparison contract")

    browser = metrics.get("browser")
    app_version = metrics.get("appVersion")
    app_revision = metrics.get("appRevision")
    chrome = _CHROME_IN_USER_AGENT.search(browser) if isinstance(browser, str) else None
    if (
        chrome is None
        or metrics.get("ortVersion") != PINNED_ORT_VERSION
        or app_version != "0.1.0-experimental"
        or not isinstance(app_revision, str)
        or not _APP_REVISION.fullmatch(app_revision)
        or set(app_revision) == {"0"}
    ):
        raise ValueError("runtime provenance is missing, placeholder, or unpinned")
    environment = {
        "browser": {"userAgent": browser, "chromeVersion": chrome.group(1)},
        "runtime": {
            "onnxRuntimeWeb": PINNED_ORT_VERSION,
            "appVersion": app_version,
            "appRevision": app_revision,
        },
    }
    return input_data, generation, environment


def _wav_metadata(data: bytes, expected_samples: int, expected_bytes: int) -> dict[str, int]:
    if len(data) < 44:
        raise ValueError("WAV must be a canonical WAV")
    try:
        riff_size = struct.unpack_from("<I", data, 4)[0]
        fmt_size, audio_format, channels, sample_rate, byte_rate, block_align, bits = struct.unpack_from(
            "<IHHIIHH", data, 16
        )
        data_size = struct.unpack_from("<I", data, 40)[0]
    except struct.error as error:
        raise ValueError("WAV must be a canonical WAV") from error
    canonical = (
        data[0:4] == b"RIFF"
        and data[8:16] == b"WAVEfmt "
        and data[36:40] == b"data"
        and riff_size == len(data) - 8
        and fmt_size == 16
        and audio_format == 1
        and channels == 2
        and sample_rate == 44_100
        and byte_rate == 176_400
        and block_align == 4
        and bits == 16
        and data_size == len(data) - 44
        and data_size % 4 == 0
        and len(data) == expected_bytes
    )
    if not canonical:
        raise ValueError("WAV must be a canonical WAV with the planned byte length")
    samples = data_size // 4
    if samples != expected_samples:
        raise ValueError("WAV does not contain the planned samples")
    return {"sampleRate": sample_rate, "channels": channels, "samplesPerChannel": samples, "bytes": len(data)}


def build_receipt(manifest_path: Path, metrics_path: Path, wav_path: Path) -> dict[str, Any]:
    manifest_bytes = _read_file(manifest_path, "WebGPU manifest")
    metrics_bytes = _read_file(metrics_path, "generation metrics")
    wav_bytes = _read_file(wav_path, "WAV")
    manifest = _parse_json(manifest_bytes, "WebGPU manifest")
    metrics = _parse_json(metrics_bytes, "generation metrics")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if metrics.get("manifestHash") != manifest_sha256:
        raise ValueError("generation metrics manifestHash does not match the supplied WebGPU manifest")
    input_data, generation, environment = _validate_metrics(metrics)
    model = _validate_manifest(manifest)
    plan = _PLAN_RETAINED_FRAMES(
        retained_frames=generation["retainedFrames"], prompt_tokens=40, termination=generation["termination"]
    )
    audio = _wav_metadata(wav_bytes, int(plan["samplesPerChannel"]), int(plan["wavBytes"]))
    return {
        "schemaVersion": 1,
        "kind": "webgpu-cloud-comparison-case",
        "input": input_data,
        "generation": generation,
        "sampler": FIXED_SAMPLER.copy(),
        "provenance": {
            "model": {"id": model["id"], "revision": model["revision"]},
            "diffusersRevision": model["diffusersRevision"],
            "manifestSha256": manifest_sha256,
            "metricsSha256": hashlib.sha256(metrics_bytes).hexdigest(),
            "wavSha256": hashlib.sha256(wav_bytes).hexdigest(),
        },
        "audio": audio,
        "environment": environment,
        "comparisonScope": "cloud-structural-and-manual-audio",
    }


def create_case(output_root: Path, case_id: str, manifest_path: Path, metrics_path: Path, wav_path: Path) -> Path:
    if not _CASE_ID.fullmatch(case_id):
        raise ValueError("case id must contain only lowercase letters, digits, and hyphens")
    receipt = build_receipt(manifest_path, metrics_path, wav_path)
    root = output_root.resolve()
    target = (root / case_id).resolve()
    if target.parent != root:
        raise ValueError("case target escapes the output root")
    if target.exists():
        raise FileExistsError(f"comparison case already exists: {target}")
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".{case_id}.{uuid.uuid4().hex}.tmp"
    staging.mkdir()
    receipt_path = staging / "receipt.json"
    try:
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, target)
    except BaseException:
        if receipt_path.exists():
            receipt_path.unlink()
        if staging.exists():
            staging.rmdir()
        raise
    return target


def verify_case(case_dir: Path, manifest_path: Path, metrics_path: Path, wav_path: Path) -> dict[str, Any]:
    try:
        case_mode = os.lstat(case_dir).st_mode
    except OSError as error:
        raise ValueError("comparison case must be a regular contained directory") from error
    if case_dir.is_symlink() or not stat.S_ISDIR(case_mode):
        raise ValueError("comparison case must be a regular contained directory")
    case = case_dir.resolve()
    receipt_path = case / "receipt.json"
    try:
        receipt_mode = os.lstat(receipt_path).st_mode
    except OSError as error:
        raise ValueError("comparison receipt must be a regular contained file") from error
    if receipt_path.is_symlink() or not stat.S_ISREG(receipt_mode) or receipt_path.resolve().parent != case:
        raise ValueError("comparison receipt must be a regular contained file")
    if set(case.iterdir()) != {receipt_path}:
        raise ValueError("comparison case must contain only receipt.json")
    receipt = _load_json(receipt_path, "comparison receipt")
    expected = build_receipt(manifest_path, metrics_path, wav_path)
    provenance = _object(receipt.get("provenance"), "receipt provenance")
    for key, label in (
        ("manifestSha256", "manifest hash"),
        ("metricsSha256", "metrics hash"),
        ("wavSha256", "WAV hash"),
    ):
        if provenance.get(key) != expected["provenance"][key]:
            raise ValueError(f"{label} does not match the attached file")
    if receipt != expected:
        raise ValueError("comparison receipt content does not match its attached inputs")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or verify a pure-Python WebGPU/cloud comparison case.")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--metrics", type=Path, required=True)
    build.add_argument("--wav", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--case-id", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--case", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--metrics", type=Path, required=True)
    verify.add_argument("--wav", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        print(create_case(args.output_root, args.case_id, args.manifest, args.metrics, args.wav))
    else:
        verify_case(args.case, args.manifest, args.metrics, args.wav)
        print(f"verified {args.case}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
