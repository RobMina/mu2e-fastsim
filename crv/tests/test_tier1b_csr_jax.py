"""Complete device CSR expansion, block boundaries and invalid donor lanes."""
import numpy as np
import pytest

pytest.importorskip("jax")

from crvfastsim.tier1b_jax import _jax
from crvfastsim.tier1b_csr_jax import prepare, layout, block


@pytest.mark.parametrize("batch", [1, 3, 7, 32])
def test_complete_histories_zero_donors_and_padding(batch):
    jax, j = _jax()
    keys = np.repeat(np.arange(5), [0, 7, 0, 19, 3])
    keep = np.arange(len(keys)) % 3 != 1
    csr = prepare(keys, 5, keep)
    picks = np.array([1, 0, 3, -1, 3, 4, 2, 4])
    valid = np.array([True] * 7 + [False])
    plan = jax.jit(layout)(csr, j.asarray(picks), j.asarray(valid))
    expected = [(i, r) for i, p in enumerate(picks) if valid[i] and p >= 0
                for r in np.flatnonzero((keys == p) & keep)]
    got = []
    for start in range(0, int(plan["total"]), batch):
        ix, rows, ok = jax.jit(block)(csr, plan, j.asarray(start), j.arange(batch))
        got.extend(zip(np.asarray(ix)[ok].tolist(), np.asarray(rows)[ok].tolist()))
    assert got == expected
    assert int(plan["total"]) == len(expected)


def test_empty_history_and_empty_selection():
    jax, j = _jax()
    csr = prepare([], 3)
    plan = jax.jit(layout)(csr, j.asarray([0, -1, 2]), j.asarray([True, True, False]))
    assert int(plan["total"]) == 0
    _, _, valid = jax.jit(block)(csr, plan, j.asarray(0), j.arange(5))
    assert not np.asarray(valid).any()


def test_invalid_library_rejected():
    with pytest.raises(ValueError, match="sorted"):
        prepare([2, 0], 3)
    with pytest.raises(ValueError, match="in-range"):
        prepare([3], 3)
