import numpy as np

from minimax_music3_webgpu.embedding import shard_fp16_rows


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
