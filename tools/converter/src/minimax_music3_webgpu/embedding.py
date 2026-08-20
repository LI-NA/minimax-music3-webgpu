"""FP16 embedding-table sharding."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch

from .constants import ARTIFACT_FILE_LIMIT, HIDDEN_SIZE

MAX_CONVERSION_ROWS = 16_384


@dataclass(frozen=True)
class EmbeddingShard:
    path: Path
    row_start: int
    row_count: int
    columns: int
    row_bytes: int
    size: int
    sha256: str


@dataclass(frozen=True)
class EmbeddingTableReceipt:
    shards: tuple[EmbeddingShard, ...]


def shard_fp16_rows(
    rows: np.ndarray, output_dir: Path, max_file_bytes: int
) -> EmbeddingTableReceipt:
    if rows.ndim != 2 or rows.dtype != np.float16:
        raise ValueError("rows must be a two-dimensional FP16 array")
    row_bytes = rows.shape[1] * rows.dtype.itemsize
    rows_per_shard = max_file_bytes // row_bytes
    if rows_per_shard == 0:
        raise ValueError("max_file_bytes cannot hold one row")

    output_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for row_start in range(0, len(rows), rows_per_shard):
        chunk = np.ascontiguousarray(rows[row_start : row_start + rows_per_shard])
        path = output_dir / f"embedding-{len(shards):03d}.fp16"
        chunk.tofile(path)
        shards.append(
            EmbeddingShard(
                path=path,
                row_start=row_start,
                row_count=len(chunk),
                columns=rows.shape[1],
                row_bytes=row_bytes,
                size=path.stat().st_size,
                sha256=_sha256(path),
            )
        )
    return EmbeddingTableReceipt(tuple(shards))


def export_embedding_table(source_shard: Path, output_dir: Path) -> EmbeddingTableReceipt:
    output_dir.mkdir(parents=True, exist_ok=True)
    with safe_open(source_shard, framework="pt", device="cpu") as source:
        tensor = source.get_slice("model.embed_tokens.weight")
        shape = tensor.get_shape()
        if len(shape) != 2 or shape[1] != HIDDEN_SIZE:
            raise ValueError("model.embed_tokens.weight has an unexpected shape")
        row_bytes = HIDDEN_SIZE * np.dtype(np.float16).itemsize
        rows_per_shard = ARTIFACT_FILE_LIMIT // row_bytes
        shards = []
        for row_start in range(0, shape[0], rows_per_shard):
            row_end = min(row_start + MAX_CONVERSION_ROWS, row_start + rows_per_shard, shape[0])
            chunk = tensor[row_start:row_end].to(dtype=torch.float16).numpy()
            receipt = shard_fp16_rows(chunk, output_dir, ARTIFACT_FILE_LIMIT)
            for item in receipt.shards:
                path = output_dir / f"embedding-{len(shards):03d}.fp16"
                item.path.replace(path)
                shards.append(
                    EmbeddingShard(
                        path=path,
                        row_start=row_start + item.row_start,
                        row_count=item.row_count,
                        columns=item.columns,
                        row_bytes=item.row_bytes,
                        size=item.size,
                        sha256=item.sha256,
                    )
                )
    return EmbeddingTableReceipt(tuple(shards))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
