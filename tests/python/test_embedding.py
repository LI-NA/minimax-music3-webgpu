import hashlib

import numpy as np
import torch

from minimax_music3_webgpu import embedding
from minimax_music3_webgpu.embedding import export_embedding_table, shard_fp16_rows


def test_shard_fp16_rows_preserves_order_and_boundary(tmp_path) -> None:
    rows = np.arange(17 * 8, dtype=np.float16).reshape(17, 8)
    receipt = shard_fp16_rows(rows, tmp_path, max_file_bytes=8 * 8 * 2)

    assert [item.row_count for item in receipt.shards] == [8, 8, 1]
    restored = np.concatenate(
        [
            np.fromfile(item.path, dtype=np.float16).reshape(item.row_count, 8)
            for item in receipt.shards
        ]
    )
    np.testing.assert_array_equal(restored, rows)


def test_export_embedding_table_keeps_global_shard_names_across_batches(
    tmp_path, monkeypatch
) -> None:
    rows = np.arange(5 * 4, dtype=np.float32).reshape(5, 4)
    monkeypatch.setattr(embedding, "HIDDEN_SIZE", 4)
    monkeypatch.setattr(embedding, "ARTIFACT_FILE_LIMIT", 4 * 2 * 2)
    monkeypatch.setattr(embedding, "MAX_CONVERSION_ROWS", 2)
    monkeypatch.setattr(embedding, "safe_open", _safe_open(rows))

    receipt = export_embedding_table(tmp_path / "source.safetensors", tmp_path / "packed")

    assert [item.path.name for item in receipt.shards] == [
        "embedding-000.fp16",
        "embedding-001.fp16",
        "embedding-002.fp16",
    ]
    assert [(item.row_start, item.row_count) for item in receipt.shards] == [
        (0, 2),
        (2, 2),
        (4, 1),
    ]
    for item in receipt.shards:
        assert item.size == item.row_count * item.row_bytes
        assert item.sha256 == hashlib.sha256(item.path.read_bytes()).hexdigest()
    restored = np.concatenate(
        [np.fromfile(item.path, dtype=np.float16).reshape(item.row_count, 4) for item in receipt.shards]
    )
    np.testing.assert_array_equal(restored, rows.astype(np.float16))


def _safe_open(rows: np.ndarray):
    class Source:
        def get_slice(self, name: str):
            assert name == "model.embed_tokens.weight"
            return Slice(rows)

    class Slice:
        def __init__(self, value: np.ndarray) -> None:
            self.value = value

        def get_shape(self) -> tuple[int, int]:
            return self.value.shape

        def __getitem__(self, index):
            return torch.from_numpy(self.value[index])

    class Context:
        def __enter__(self):
            return Source()

        def __exit__(self, *args) -> None:
            return None

    return lambda *args, **kwargs: Context()
