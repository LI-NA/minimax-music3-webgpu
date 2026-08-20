import hashlib
import json
from pathlib import Path

from minimax_music3_webgpu.paths import ArtifactPaths
from minimax_music3_webgpu.source import download_global_source, global_source_patterns


def test_global_source_patterns_exclude_legacy_and_acoustic_weights() -> None:
    patterns = global_source_patterns()
    assert patterns == (
        "LICENSE",
        "modular_model_index.json",
        "language_model/*",
        "tokenizer/*",
    )
    assert all("qwen_7B" not in pattern for pattern in patterns)
    assert all("transformer" not in pattern for pattern in patterns)


def test_download_global_source_writes_pinned_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    paths = ArtifactPaths.from_root(repository_root / "artifacts", repository_root)

    def fake_snapshot_download(**kwargs):
        assert kwargs == {
            "repo_id": "MiniMaxAI/MiniMax-Music3",
            "revision": "fbdf52fbaaca799592917417eb05f1899f1255ec",
            "allow_patterns": global_source_patterns(),
            "local_dir": paths.source,
            "cache_dir": paths.root / "hf-cache",
        }
        (paths.source / "tokenizer").mkdir(parents=True)
        (paths.source / "LICENSE").write_text("license", encoding="utf-8")
        (paths.source / "tokenizer" / "vocab.json").write_text("{}", encoding="utf-8")
        return str(paths.source)

    monkeypatch.setattr("minimax_music3_webgpu.source.snapshot_download", fake_snapshot_download)

    receipt = download_global_source(paths)

    receipt_path = paths.receipts / "source-global.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt.repository_id == "MiniMaxAI/MiniMax-Music3"
    assert receipt.revision == "fbdf52fbaaca799592917417eb05f1899f1255ec"
    assert payload == {
        "repository_id": "MiniMaxAI/MiniMax-Music3",
        "revision": "fbdf52fbaaca799592917417eb05f1899f1255ec",
        "files": [
            {
                "path": "LICENSE",
                "size": 7,
                "sha256": hashlib.sha256(b"license").hexdigest(),
            },
            {
                "path": "tokenizer/vocab.json",
                "size": 2,
                "sha256": hashlib.sha256(b"{}").hexdigest(),
            },
        ],
    }
