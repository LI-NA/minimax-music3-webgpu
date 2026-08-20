from pathlib import Path

import pytest

from minimax_music3_webgpu.paths import ArtifactPaths


def test_artifact_paths_reject_root_outside_repository(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()

    with pytest.raises(ValueError, match="repository"):
        ArtifactPaths.from_root(tmp_path / "outside", repository_root)


def test_artifact_paths_stay_below_artifacts_root(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    artifacts_root = repository_root / "artifacts"
    repository_root.mkdir()

    paths = ArtifactPaths.from_root(artifacts_root, repository_root)

    for path in (paths.root, paths.source, paths.work, paths.release, paths.receipts):
        assert path.is_relative_to(paths.root)
