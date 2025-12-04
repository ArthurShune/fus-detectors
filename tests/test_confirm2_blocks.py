import numpy as np

from scripts import confirm2_eval


def test_reduce_blocks_identity_for_size_one() -> None:
    arr = np.arange(10, dtype=np.float32)
    reduced, blocks = confirm2_eval._reduce_blocks(arr, 1)
    assert blocks == arr.size
    np.testing.assert_allclose(reduced, arr)


def test_reduce_blocks_takes_max_per_block() -> None:
    arr = np.array([1, 5, 2, 3, 9, 4], dtype=np.float32)
    reduced, blocks = confirm2_eval._reduce_blocks(arr, 2)
    assert blocks == 3
    np.testing.assert_allclose(reduced, np.array([5, 3, 9], dtype=np.float32))
