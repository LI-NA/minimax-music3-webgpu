"""Command-line entry point for conversion tools."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MiniMax Music 3 models for WebGPU.")
    parser.parse_args()


if __name__ == "__main__":
    main()
