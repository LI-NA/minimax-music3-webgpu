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
        paths = cls(
            root=artifact_root,
            source=artifact_root / "source",
            work=artifact_root / "work",
            release=artifact_root / "release",
            receipts=artifact_root / "receipts",
        )
        paths.validate_write_targets(
            paths.source,
            paths.work,
            paths.release,
            paths.receipts,
            paths.root / "hf-cache",
        )
        return paths

    def validate_write_targets(self, *targets: Path) -> None:
        root = self.root.resolve()
        for target in targets:
            if not target.resolve().is_relative_to(root):
                raise ValueError("artifact write target must remain under artifact root")
