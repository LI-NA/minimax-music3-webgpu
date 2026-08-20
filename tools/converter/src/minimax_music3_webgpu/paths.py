"""Artifact path definitions."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    source: Path
    work: Path
    release: Path
    receipts: Path

    @classmethod
    def from_root(
        cls, root: Path, repository_root: Path = Path.cwd()
    ) -> "ArtifactPaths":
        repository = repository_root.resolve()
        artifact_root = root.resolve()
        if not artifact_root.is_relative_to(repository):
            raise ValueError("artifact root must remain inside the repository")
        return cls(
            root=artifact_root,
            source=artifact_root / "source",
            work=artifact_root / "work",
            release=artifact_root / "release",
            receipts=artifact_root / "receipts",
        )
