"""Selective acoustic source-model download support."""

from dataclasses import dataclass

from huggingface_hub import snapshot_download

from .constants import ACOUSTIC_SOURCE_FILES, MODEL_ID, MODEL_REVISION
from .paths import ArtifactPaths
from .source import SourceFile, _source_file, _write_receipt


@dataclass(frozen=True)
class AcousticSourceReceipt:
    repository_id: str
    revision: str
    files: tuple[SourceFile, ...]


def acoustic_source_patterns() -> tuple[str, ...]:
    return ACOUSTIC_SOURCE_FILES


def download_acoustic_source(paths: ArtifactPaths) -> AcousticSourceReceipt:
    cache_dir = paths.root / "hf-cache"
    receipt_path = paths.receipts / "source-acoustic.json"
    paths.validate_write_targets(
        paths.source,
        paths.receipts,
        cache_dir,
        receipt_path,
    )
    paths.source.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=acoustic_source_patterns(),
        local_dir=paths.source,
        cache_dir=cache_dir,
    )
    selected_files = tuple(paths.source_path(path) for path in acoustic_source_patterns())
    paths.validate_write_targets(*selected_files)
    missing_files = [
        path.relative_to(paths.source).as_posix()
        for path in selected_files
        if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"acoustic source download is missing required files: {', '.join(missing_files)}"
        )
    receipt = AcousticSourceReceipt(
        repository_id=MODEL_ID,
        revision=MODEL_REVISION,
        files=tuple(_source_file(path, paths.source) for path in selected_files),
    )
    _write_receipt(receipt_path, receipt)
    return receipt
