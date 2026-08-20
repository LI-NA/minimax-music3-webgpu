"""Command-line entry point for conversion tools."""

import argparse
from pathlib import Path

from .acoustic_source import download_acoustic_source
from .paths import ArtifactPaths
from .source import download_global_source
from .global_decoder import build_global_decoder
from .manifest import emit_global_release
from .manifest import emit_rvq_release
from .rvq_depth import build_rvq_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MiniMax Music 3 models for WebGPU.")
    subparsers = parser.add_subparsers(dest="command")
    download_parser = subparsers.add_parser("download-global")
    download_parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    acoustic_parser = subparsers.add_parser("download-acoustic")
    acoustic_parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    build_parser = subparsers.add_parser("build-global")
    build_parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    build_parser.add_argument("--layers", type=int, choices=(1, 36), default=36)
    rvq_parser = subparsers.add_parser("build-rvq")
    rvq_parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.command == "download-global":
        download_global_source(ArtifactPaths.from_root(args.artifacts_dir))
    if args.command == "download-acoustic":
        download_acoustic_source(ArtifactPaths.from_root(args.artifacts_dir))
    if args.command == "build-global":
        paths = ArtifactPaths.from_root(args.artifacts_dir)
        build_global_decoder(paths, args.layers)
        emit_global_release(paths, args.layers)
    if args.command == "build-rvq":
        paths = ArtifactPaths.from_root(args.artifacts_dir)
        emit_rvq_release(paths, build_rvq_stage(paths))


if __name__ == "__main__":
    main()
