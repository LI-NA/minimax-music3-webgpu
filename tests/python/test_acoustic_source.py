import hashlib
import json
from pathlib import Path
import sys

import pytest

from minimax_music3_webgpu.acoustic_source import (
    acoustic_source_patterns,
    download_acoustic_source,
)
from minimax_music3_webgpu.paths import ArtifactPaths
from minimax_music3_webgpu.cli import main


def test_acoustic_source_patterns_select_only_required_components() -> None:
    assert acoustic_source_patterns() == (
        "condition_encoder/config.json",
        "condition_encoder/diffusion_pytorch_model.safetensors",
        "rvq_depth_decoder/config.json",
        "rvq_depth_decoder/diffusion_pytorch_model.safetensors",
        "scheduler/scheduler_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        "vocoder/config.json",
        "vocoder/diffusion_pytorch_model.safetensors",
    )
    assert all("qwen_7B" not in path for path in acoustic_source_patterns())
    assert "dav.pth" not in acoustic_source_patterns()
    assert "flowmatching_vae.pth" not in acoustic_source_patterns()


def test_download_acoustic_source_writes_exact_pinned_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    paths = ArtifactPaths.from_root(repository_root / "artifacts", repository_root)

    def fake_snapshot_download(**kwargs):
        assert kwargs == {
            "repo_id": "MiniMaxAI/MiniMax-Music3",
            "revision": "fbdf52fbaaca799592917417eb05f1899f1255ec",
            "allow_patterns": acoustic_source_patterns(),
            "local_dir": paths.source,
            "cache_dir": paths.root / "hf-cache",
        }
        for relative_path in acoustic_source_patterns():
            target = paths.source / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(relative_path.encode())
        for relative_path in (
            "qwen_7B/model.safetensors",
            "dav.pth",
            "flowmatching_vae.pth",
            "transformer/unlisted.bin",
        ):
            target = paths.source / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("legacy", encoding="utf-8")
        return str(paths.source)

    monkeypatch.setattr(
        "minimax_music3_webgpu.acoustic_source.snapshot_download",
        fake_snapshot_download,
    )

    receipt = download_acoustic_source(paths)

    receipt_path = paths.receipts / "source-acoustic.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt.repository_id == "MiniMaxAI/MiniMax-Music3"
    assert receipt.revision == "fbdf52fbaaca799592917417eb05f1899f1255ec"
    assert payload == {
        "repository_id": receipt.repository_id,
        "revision": receipt.revision,
        "files": [
            {
                "path": relative_path,
                "size": len(relative_path.encode()),
                "sha256": hashlib.sha256(relative_path.encode()).hexdigest(),
            }
            for relative_path in acoustic_source_patterns()
        ],
    }
    assert not list(paths.receipts.glob(".source-acoustic.json.*.tmp"))


def test_download_acoustic_source_rejects_incomplete_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    paths = ArtifactPaths.from_root(repository_root / "artifacts", repository_root)

    def fake_snapshot_download(**kwargs):
        return str(paths.source)

    monkeypatch.setattr(
        "minimax_music3_webgpu.acoustic_source.snapshot_download",
        fake_snapshot_download,
    )

    with pytest.raises(FileNotFoundError, match="condition_encoder/config.json"):
        download_acoustic_source(paths)

    assert not (paths.receipts / "source-acoustic.json").exists()


def test_download_acoustic_command_uses_requested_artifact_root(
    tmp_path: Path, monkeypatch
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    requested_root = repository_root / "artifacts"
    received_paths = []
    monkeypatch.chdir(repository_root)
    monkeypatch.setattr(
        "minimax_music3_webgpu.cli.download_acoustic_source",
        received_paths.append,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "music3-convert",
            "download-acoustic",
            "--artifacts-dir",
            str(requested_root),
        ],
    )

    main()

    assert received_paths == [ArtifactPaths.from_root(requested_root, repository_root)]
