"""Coincidence-finder component labels: min-label propagation (coinc_jax._labels) gives the same integers as the
transitive-closure reference (_labels_closure) on random link graphs, long chains and padded groups."""
import numpy as np
import pytest

pytest.importorskip('jax')
import jax.numpy as jnp

from crvfastsim.reco import coinc_jax as J


def adjacency(links, valid):
    """Symmetric adjacency of the valid hits, diagonal set for valid hits (as find_dense builds A1 and A2)."""
    A = links | np.swapaxes(links, 1, 2)
    A |= np.eye(valid.shape[1], dtype=bool)[None]
    return A & valid[:, :, None] & valid[:, None, :]


def check(A, valid):
    got = np.asarray(J._labels(jnp.asarray(A), jnp.asarray(valid)))
    ref = np.asarray(J._labels_closure(jnp.asarray(A), jnp.asarray(valid)))
    assert got.dtype == ref.dtype
    np.testing.assert_array_equal(got, ref)
    return got


@pytest.mark.parametrize("H,p", [(16, 0.05), (32, 0.08), (64, 0.03), (187, 0.01), (187, 0.002)])
def test_random_graphs_match_closure(H, p):
    rng = np.random.default_rng(H)
    G = 64
    valid = rng.random((G, H)) < 0.8
    valid[0] = False          # an empty group
    valid[1, :] = True        # a full group
    links = rng.random((G, H, H)) < p
    check(adjacency(links, valid), valid)


def test_long_chains_and_isolated_hits():
    H = 128
    valid = np.ones((3, H), bool)
    valid[2, ::3] = False
    links = np.zeros((3, H, H), bool)
    order = np.random.default_rng(1).permutation(H)    # one chain through every hit, in a random index order
    links[0, order[:-1], order[1:]] = True
    links[1, np.arange(H - 1), np.arange(1, H)] = True  # a chain in index order; group 2 keeps only isolated hits
    got = check(adjacency(links, valid), valid)
    assert np.all(got[0] == 0) and np.all(got[1] == 0)
    np.testing.assert_array_equal(got[2], np.where(valid[2], np.arange(H), -1))
