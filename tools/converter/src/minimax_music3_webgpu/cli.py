"""Command-line entry point for conversion tools."""

import argparse
from pathlib import Path

from .paths import ArtifactPaths
from .source import download_global_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MiniMax Music 3 models for WebGPU.")
    subparsers = parser.add_subparsers(dest="command")
    download_parser = subparsers.add_parser("download-global")
    download_parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.command == "download-global":
        download_global_source(ArtifactPaths.from_root(args.artifacts_dir))


if __name__ == "__main__":
    main()
