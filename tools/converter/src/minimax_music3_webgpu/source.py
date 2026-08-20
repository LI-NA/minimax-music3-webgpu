"""Selective, pinned source-model download support."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from huggingface_hub import snapshot_download

from .constants import MODEL_ID, MODEL_REVISION
from .paths import ArtifactPaths


@dataclass(frozen=True)
class SourceFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class SourceReceipt:
    repository_id: str
    revision: str
    files: tuple[SourceFile, ...]


def global_source_patterns() -> tuple[str, ...]:
    return (
        "LICENSE",
        "modular_model_index.json",
        "language_model/*",
        "tokenizer/*",
    )


def download_global_source(paths: ArtifactPaths) -> SourceReceipt:
    cache_dir = paths.root / "hf-cache"
    receipt_path = paths.receipts / "source-global.json"
    paths.validate_write_targets(
        paths.source,
        paths.work,
        paths.release,
        paths.receipts,
        cache_dir,
        receipt_path,
    )
    paths.source.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=global_source_patterns(),
        local_dir=paths.source,
        cache_dir=cache_dir,
    )
    source_files = tuple(
        _source_file(path, paths.source)
        for path in sorted(paths.source.rglob("*"))
        if path.is_file()
        and _matches_global_source_path(path.relative_to(paths.source))
    )
    receipt = SourceReceipt(
        repository_id=MODEL_ID,
        revision=MODEL_REVISION,
        files=source_files,
    )
    _write_receipt(receipt_path, receipt)
    return receipt


def _matches_global_source_path(path: Path) -> bool:
    relative_path = path.as_posix()
    return (
        relative_path in {"LICENSE", "modular_model_index.json"}
        or relative_path.startswith(("language_model/", "tokenizer/"))
    )


def _source_file(path: Path, source_root: Path) -> SourceFile:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return SourceFile(
        path=path.relative_to(source_root).as_posix(),
        size=path.stat().st_size,
        sha256=digest.hexdigest(),
    )


def _write_receipt(path: Path, receipt: SourceReceipt) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "repository_id": receipt.repository_id,
        "revision": receipt.revision,
        "files": [asdict(file) for file in receipt.files],
    }
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)
