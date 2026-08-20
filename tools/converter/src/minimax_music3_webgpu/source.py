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
    paths.source.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=global_source_patterns(),
        local_dir=paths.source,
        cache_dir=paths.root / "hf-cache",
    )
    source_files = tuple(
        _source_file(path, paths.source)
        for path in sorted(paths.source.rglob("*"))
        if path.is_file()
    )
    receipt = SourceReceipt(
        repository_id=MODEL_ID,
        revision=MODEL_REVISION,
        files=source_files,
    )
    _write_receipt(paths.receipts / "source-global.json", receipt)
    return receipt


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
