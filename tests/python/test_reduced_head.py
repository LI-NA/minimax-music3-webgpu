import numpy as np

from minimax_music3_webgpu.reduced_head import select_reduced_head_rows


def test_select_reduced_head_rows_preserves_exact_semantic_and_end_rows() -> None:
    weights = np.arange(20 * 3, dtype=np.float16).reshape(20, 3)

    semantic, end = select_reduced_head_rows(
        weights,
        semantic_start=5,
        semantic_count=4,
        end_token_id=2,
    )

    np.testing.assert_array_equal(semantic, weights[5:9])
    np.testing.assert_array_equal(end, weights[2:3])
