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


@pytest.mark.parametrize(
    "target_name", ("source", "work", "release", "receipts", "hf-cache")
)
def test_artifact_paths_reject_escaped_write_target(
    tmp_path: Path, target_name: str
) -> None:
    repository_root = tmp_path / "repository"
    outside = tmp_path / "outside"
    repository_root.mkdir()
    outside.mkdir()
    artifacts_root = repository_root / "artifacts"
    artifacts_root.mkdir()
    target = artifacts_root / target_name

    try:
        target.symlink_to(outside, target_is_directory=True)
    except OSError:
        real_resolve = Path.resolve

        def resolve_with_escaped_target(path: Path, *args, **kwargs) -> Path:
            if path == target:
                return outside
            return real_resolve(path, *args, **kwargs)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(Path, "resolve", resolve_with_escaped_target)

        with pytest.raises(ValueError, match="artifact root"):
            ArtifactPaths.from_root(artifacts_root, repository_root)
        monkeypatch.undo()
    else:
        with pytest.raises(ValueError, match="artifact root"):
            ArtifactPaths.from_root(artifacts_root, repository_root)
