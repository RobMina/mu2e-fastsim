"""Explicit-key ionisation: reference distributions, geometry and batching invariants."""
from copy import copy
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from test_digi import DG, _step, _steps, cond  # noqa: E402,F401
from trkfastsim.digi import digi_clusters_jax as DJ  # noqa: E402


def _table(c, n, **kwargs):
    row = _step(c, 400, **kwargs)
    st = {k: np.full(n, v) for k, v in row.items()}
    st["_us"] = DG.unique_straw(st["sid"])
    return st


def _assert_moments(a, b, sigma=6):
    a, b = np.asarray(a, float), np.asarray(b, float)
    se = np.sqrt(a.var() / len(a) + b.var() / len(b))
    assert abs(a.mean() - b.mean()) < sigma * se + 1e-12
    va, vb = (a - a.mean()) ** 2, (b - b.mean()) ** 2
    sevar = np.sqrt(va.var() / len(va) + vb.var() / len(vb))
    assert abs(va.mean() - vb.mean()) < sigma * sevar + 1e-12


def test_empty_and_optional_import(cond):
    empty = DJ.make_clusters(jax.random.key(1), cond, {}, np.array([], np.int64))
    assert tuple(x.shape for x in empty) == ((0,), (0, 3), (0,))
    assert tuple(x.dtype for x in empty) == (np.int64, np.float64, np.int64)
    root = Path(__file__).resolve().parent.parent
    code = """
import builtins
original = builtins.__import__
def no_jax(name, *args, **kwargs):
    if name == 'jax' or name.startswith('jax.'):
        raise ImportError('deliberately absent')
    return original(name, *args, **kwargs)
builtins.__import__ = no_jax
from trkfastsim.digi import digi_clusters_jax as dj
assert dj.jax is None
try:
    dj.make_clusters(None, None, {}, [])
except ImportError as exc:
    assert 'JAX is required' in str(exc)
else:
    raise AssertionError('backend must reject absent JAX')
"""
    subprocess.run([sys.executable, "-c", code], cwd=root, env=os.environ.copy(), check=True)


def test_single_poisson_and_short_step(cond):
    c = cond
    st = _table(c, 8192, eion=7.0 * c.EAverage)
    st["shape"][:4096] = 3
    st["len"][4096:] = c.meanFreePath * 0.5
    idx = np.arange(len(st["eion"]))
    cs, pos, ne = DJ.make_clusters(jax.random.key(29), c, st, idx)
    ref = DG.make_clusters(np.random.default_rng(202), c, st, idx)
    np.testing.assert_array_equal(cs, idx)
    np.testing.assert_array_equal(pos, np.stack([st[k] for k in ("x0", "y0", "z0")], 1))
    assert np.all(ne >= 1) and ne.dtype == np.int64 and pos.dtype == np.float64
    _assert_moments(ne, ref[2])
    assert abs(ne.mean() - (7.0 + np.exp(-7.0))) < 0.15
    st["eion"][:] = 0
    np.testing.assert_array_equal(DJ.make_clusters(jax.random.key(4), c, st, idx)[2], 1)


@pytest.mark.slow
def test_nonminion_count_electron_and_position_distributions(cond):
    c = cond
    st = _table(c, 2048, ion=1, eion=2e-3)
    st["len"][:] = 5.0 * c.meanFreePath
    st["width"][:] = 0.4
    idx = np.arange(len(st["eion"]))
    cs, pos, ne = DJ.make_clusters(jax.random.key(71), c, st, idx)
    rs, rp, rn = DG.make_clusters(np.random.default_rng(123), c, st, idx)
    _assert_moments(np.bincount(cs, minlength=len(idx)), np.bincount(rs, minlength=len(idx)))
    _assert_moments(ne, rn)
    p0 = np.array([st[k][0] for k in ("x0", "y0", "z0")])
    p1 = np.array([st[k][0] for k in ("x1", "y1", "z1")])
    frac = (pos[:, 2] - p0[2]) / (p1[2] - p0[2])
    assert np.all((frac >= 0) & (frac <= 1))
    assert abs(frac.mean() - 0.5) < 0.015 and abs(frac.var() - 1 / 12) < 0.005
    for axis in (0, 1):
        offset = pos[:, axis] - p0[axis]
        _assert_moments(offset, rp[:, axis] - p0[axis])
        assert abs(offset.mean()) < 0.015 and abs(offset.std() - 0.4) < 0.015
    assert np.max(np.bincount(cs)) <= c.maxnclu


def test_energy_and_nonminion_caps_and_zero_multi(cond):
    c = cond
    st = _table(c, 12, ion=1, eion=1.0)
    st["len"][:] = 100.0 * c.meanFreePath
    st["eion"][4:8] = 4.5 * c.EIonize[0]
    st["eion"][8:] = 0.5 * c.EIonize[0]
    cs, _, _ = DJ.make_clusters(jax.random.key(4), c, st, np.arange(12))
    np.testing.assert_array_equal(np.bincount(cs, minlength=12), np.repeat([c.maxnclu, 4, 0], 4))
    st["ion"][:] = 0
    cs, _, ne = DJ.make_clusters(jax.random.key(4), c, st, np.arange(4, 12))
    np.testing.assert_array_equal(np.bincount(cs, minlength=12)[4:], np.repeat([4, 0], 4))
    np.testing.assert_array_equal(ne, 1)
    assert DJ.make_clusters(jax.random.key(4), c, st, np.arange(8, 12))[1].shape == (0, 3)


def test_minion_reference_and_electron_expansion(cond):
    c = cond
    st = _table(c, 1024, eion=2e-3)
    idx = np.arange(len(st["eion"]))
    cs, pos, ne = DJ.make_clusters(jax.random.key(19), c, st, idx)
    rs, rp, rn = DG.make_clusters(np.random.default_rng(70), c, st, idx)
    counts = np.bincount(cs, minlength=len(idx))
    _assert_moments(counts, np.bincount(rs, minlength=len(idx)))
    np.testing.assert_array_equal(ne, 1)
    assert np.all(counts * c.EIonize[0] <= st["eion"])
    start = np.r_[True, (cs[1:] != cs[:-1]) | np.any(pos[1:] != pos[:-1], axis=1)]
    rstart = np.r_[True, (rs[1:] != rs[:-1]) | np.any(rp[1:] != rp[:-1], axis=1)]
    sizes = np.diff(np.r_[np.flatnonzero(start), len(cs)])
    rsizes = np.diff(np.r_[np.flatnonzero(rstart), len(rs)])
    _assert_moments(sizes, rsizes)
    assert np.max(sizes) > 1 and 50 < counts.mean() < 140


def test_minion_residual_and_strict_truncation(cond):
    c = cond
    n = DJ.BATCH_SIZE
    keys = DJ._step_keys(jax.random.key(101), jnp.arange(n, dtype=jnp.int64))
    nc = np.tile([1, 2, 14, 30], n // 4)
    energy = np.tile([3e-2, 5e-5, 2e-3, 3e-2], n // 4)
    ne, residual = DJ._minion_clusters(keys, jnp.asarray(c.EIonize), jnp.asarray(c.intNProb),
                                       jnp.asarray(energy), jnp.asarray(nc), 32)
    ne, residual = np.asarray(ne), np.asarray(residual)
    valid = np.arange(32)[None, :] < nc[:, None]
    assert np.all(ne[valid] >= 1) and np.all(ne[~valid] == 0)
    assert np.all((residual >= -1e-16) & (residual < c.EIonize[0]))
    assert np.all(ne.sum(1) * c.EIonize[0] <= energy)
    ref = DG._minion_clusters(np.random.default_rng(15), c, energy, nc)
    ref_tot = np.add.reduceat(ref, np.r_[0, np.cumsum(nc)[:-1]])
    _assert_moments(ne.sum(1), ref_tot)
    emax = jnp.full(n, c.EIonize[1])
    sampled = DJ._trunc_ne(keys, jnp.asarray(c.intNProb), jnp.asarray(c.EIonize), emax)
    np.testing.assert_array_equal(sampled, 1)
    ne, res = DJ._minion_clusters(keys, jnp.asarray(c.EIonize), jnp.asarray(c.intNProb),
                                  jnp.full(n, c.EIonize[0]), jnp.ones(n, jnp.int64), 1)
    np.testing.assert_array_equal(ne, 1)
    np.testing.assert_array_equal(res, 0)


@pytest.mark.slow
def test_uncapped_residual_and_large_minion_cluster_bucket(cond):
    keys = DJ._step_keys(jax.random.key(22), jnp.arange(4, dtype=jnp.int64))
    energy = jnp.array([1.0, 65.25, 4097.25, 8193.25])
    ne, residual = DJ._minion_clusters(keys, jnp.array([1.0]), jnp.array([1.0]),
                                       energy, jnp.ones(4, jnp.int64), 1)
    np.testing.assert_array_equal(ne[:, 0], np.floor(energy))
    np.testing.assert_array_equal(residual, np.array([0.0, 0.25, 0.25, 0.25]))
    c = cond
    st = _table(c, 256, eion=8e-3)
    st["len"][:] = 80 * c.meanFreePath
    cs, pos, ne = DJ.make_clusters(jax.random.key(25), c, st, np.arange(256))
    start = np.r_[True, (cs[1:] != cs[:-1]) | np.any(pos[1:] != pos[:-1], axis=1)]
    initial_clusters = np.bincount(cs[start], minlength=256)
    assert np.all(initial_clusters > c.maxnclu) and initial_clusters.max() > 64
    assert abs(initial_clusters.mean() - 80) < 6 * np.sqrt(80 / 256)
    np.testing.assert_array_equal(ne, 1)


def test_reproducible_subset_order_and_fixed_batches(cond, monkeypatch):
    c = cond
    st = _table(c, DJ.BATCH_SIZE + 9, eion=2e-3)
    n = len(st["eion"])
    st["shape"][::7] = 3
    st["ion"][::2] = 1
    st["len"][::3] = c.meanFreePath * 3
    st["width"][:] = 0.3
    idx = np.arange(n)[::-1]
    key = jax.random.key(82)
    a = DJ.make_clusters(key, c, st, idx)
    b = DJ.make_clusters(key, c, st, idx)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    selected = np.array([0, 1, 10, DJ.BATCH_SIZE, n - 1])
    sub = DJ.make_clusters(key, c, st, selected)
    for i in selected:
        for x, y in zip(a[1:], sub[1:]):
            np.testing.assert_array_equal(x[a[0] == i], y[sub[0] == i])
    change = DJ.make_clusters(jax.random.key(83), c, st, idx)
    assert not np.array_equal(a[1], change[1])
    single_steps = idx[st["shape"][idx] == 3]
    np.testing.assert_array_equal(a[0][:len(single_steps)], single_steps)
    def forbidden(*args, **kwargs):
        raise AssertionError("NumPy RNG used by the JAX backend")
    for name in ("default_rng", "random", "poisson", "normal", "standard_normal", "standard_gamma"):
        monkeypatch.setattr(np.random, name, forbidden)
    DJ.make_clusters(key, c, st, selected)


def test_overflow_fails_closed_and_invalid_input(cond, monkeypatch):
    c = cond
    st = _table(c, 1, eion=1.0)
    st["len"][:] = 500.0 * c.meanFreePath
    monkeypatch.setattr(DJ, "MAX_CLUSTER_BUCKET", 32)
    with pytest.raises(OverflowError, match="MAX_CLUSTER_BUCKET"):
        DJ.make_clusters(jax.random.key(1), c, st, np.array([0]))
    st["len"][:] = 1e100
    with pytest.raises(OverflowError, match="sampling range"):
        DJ.make_clusters(jax.random.key(1), c, st, np.array([0]))
    st["len"][:] = np.nan
    with pytest.raises(ValueError, match="finite"):
        DJ.make_clusters(jax.random.key(1), c, st, np.array([0]))
    with pytest.raises(IndexError):
        DJ.make_clusters(jax.random.key(1), c, st, np.array([-1]))


def test_wire_coordinates_reference_and_degenerate_chord(cond):
    c = copy(cond)
    c.wire_mid = c.wire_mid.copy()
    c.wire_mid[400] += [0, 0.1, -0.2]
    positions = c.straw_mid[400] + np.array([[12, 0, 0], [-14, 1, 0], [0, -8, 3], [2, 1, -4]])
    us = np.full(4, 400)
    for actual, expected in zip(DJ.wire_coordinates(c, us, positions), DG.wire_coordinates(c, us, positions)):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    st = _table(c, 8, ion=1)
    st["width"][:] = 1.0
    st["x1"][:4], st["y1"][:4], st["z1"][:4] = st["x0"][:4], st["y0"][:4], st["z0"][:4]
    st["x1"][4:] = st["x0"][4:] + 2.0
    st["y1"][4:], st["z1"][4:] = st["y0"][4:], st["z0"][4:]
    cs, pos, ne = DJ.make_clusters(jax.random.key(3), c, st, np.arange(8))
    assert np.isfinite(pos).all()
    np.testing.assert_array_equal(pos[cs < 4], np.tile([st[k][0] for k in ("x0", "y0", "z0")], (np.sum(cs < 4), 1)))
    np.testing.assert_array_equal(pos[:, 1], st["y0"][cs])
    np.testing.assert_array_equal(pos[:, 2], st["z0"][cs])


def test_cluster_gain_distributions_and_unclipped_tail(cond):
    c = copy(cond)
    for ne in (1, 3, c.nGainGauss):
        counts = np.full(8192, ne)
        actual = DJ.cluster_gain(jax.random.key(ne), c, counts)
        expected = DG.cluster_gain(np.random.default_rng(ne), c, counts)
        _assert_moments(actual, expected)
        assert abs(actual.mean() / c.gasGain - 1) < 0.05
    c.gainRMSSlope = 10.0
    actual = DJ.cluster_gain(jax.random.key(8), c, np.full(8192, c.nGainGauss))
    assert np.any(actual < 0)
    assert DJ.cluster_gain(jax.random.key(1), c, np.empty(0, np.int64)).shape == (0,)


def test_poisson_split_equals_vmapped_jax_poisson():
    """POISSON_SPLIT (lanes sorted by mean, run in chunks with only the algorithm(s) they use) gives
    vmap(jax.random.poisson)'s draws bit for bit: zero, small, around the Knuth / rejection boundary at 10 (float64
    means that round to 10.0f included), large and NaN means, typed and raw keys, float64 and float32 means, any chunk."""
    assert DJ._poisson_parts() is not None      # this JAX has the per-algorithm samplers: the split is not off
    rng = np.random.default_rng(3)
    mean = np.concatenate([np.zeros(37), rng.uniform(0, 1, 300), rng.uniform(1, 9.99, 300),
                           [9.999999, 10.0, 10.000001, 10 - 1e-9, 10 + 1e-9], rng.uniform(10, 40, 200),
                           [1e3, 1e5, 3e6, np.nan]])
    rng.shuffle(mean)
    ref = jax.jit(jax.vmap(lambda k, m: jax.random.poisson(k, m, dtype=jnp.int64)))
    split = jax.jit(DJ._poisson_chunks, static_argnums=2)
    for keys in (jax.random.split(jax.random.key(11), len(mean)), jax.random.split(jax.random.PRNGKey(11), len(mean))):
        for dt in (np.float64, np.float32):
            m = jnp.asarray(mean.astype(dt))
            want = np.asarray(ref(keys, m))
            assert (want > 10).sum() > 100 and (want == 0).sum() > 37
            for chunk in (1, 7, 64, 2048):
                np.testing.assert_array_equal(np.asarray(split(keys, m, chunk)), want)


def test_poisson_split_gives_the_same_clusters(cond, monkeypatch):
    """make_clusters with the Poisson split on and off: the same clusters, positions and electron counts, with cluster
    and electron means on both sides of the Knuth / rejection boundary."""
    c = cond
    n = 600
    rng = np.random.default_rng(5)
    st = _table(c, n, eion=2e-3)
    st["eion"] = 10 ** rng.uniform(-4.5, -2.0, n)
    st["len"] = c.meanFreePath * rng.uniform(0.1, 30.0, n)
    st["shape"][::3] = 3
    st["ion"][::2] = 1
    st["width"][:] = 0.3
    key = jax.random.key(9)
    out = {}
    for on in (False, True):
        monkeypatch.setattr(DJ, "POISSON_SPLIT", on)
        jax.clear_caches()                       # the setting is read when a kernel is traced
        out[on] = DJ.make_clusters(key, c, st, np.arange(n))
    assert len(out[True][2]) > n and np.asarray(out[True][2]).max() > 10
    for x, y in zip(out[False], out[True]):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _poisson_chi2(draws, lam):
    """chi2/ndf of draws against the Poisson pmf of mean lam, over bins with expected count >= 20 (tails pooled)."""
    import math
    n = draws.size
    lo, hi = max(0, int(lam - 8 * math.sqrt(lam + 1))), int(lam + 8 * math.sqrt(lam + 1)) + 2
    ks = np.arange(lo, hi)
    pmf = np.exp(-lam + ks * math.log(lam) - np.array([math.lgamma(k + 1) for k in ks])) if lam > 0 else (ks == 0) * 1.0
    expect, seen = n * pmf, np.array([(draws == k).sum() for k in ks], float)
    keep = expect >= 20
    rest_e, rest_s = n - expect[keep].sum(), n - seen[keep].sum()
    chi2 = float((((seen[keep] - expect[keep]) ** 2) / expect[keep]).sum())
    ndf = int(keep.sum()) - 1
    if rest_e >= 20:
        chi2 += (rest_s - rest_e) ** 2 / rest_e
        ndf += 1
    return chi2 / max(ndf, 1)


@pytest.mark.parametrize("dt", [np.float64, np.float32])
@pytest.mark.parametrize("terms,trials", [(None, None), (3, 0)])
def test_poisson_lanes_has_the_poisson_law(dt, terms, trials):
    """_poisson_lanes (POISSON_METHOD "lanes"): Poisson draws for means below, at and above the inversion / PTRS
    boundary, with the default fused pass and with the exact continuation loop doing most of the work (3 terms, no
    trial in the fused pass); means and pmf chi2/ndf against the Poisson law, typed keys, both float types."""
    n = 60000
    kw = {} if terms is None else {"terms": terms, "trials": trials}
    f = jax.jit(lambda k, m: DJ._poisson_lanes(k, m, **kw))
    for i, lam in enumerate((0.0, 0.3, 2.5, 9.99, 10.0, 13.0, 47.0, 1.0e3, 1.0e5)):
        keys = jax.random.split(jax.random.key(100 + i), n)
        x = np.asarray(f(keys, jnp.full(n, lam, dt)))
        assert x.dtype == np.int64 and x.min() >= 0
        if lam == 0:
            assert not x.any()
            continue
        assert abs(x.mean() - lam) < 5 * np.sqrt(lam / n), (lam, x.mean())
        assert abs(x.var() / lam - 1) < 0.05, (lam, x.var())
        if lam < 100:
            assert _poisson_chi2(x, lam) < 2.0, (lam, _poisson_chi2(x, lam))


def test_poisson_lanes_edges_and_determinism():
    """Non-finite and negative means take jax.random.poisson's draw for those lanes; the same keys give the same draws;
    raw and typed keys both work; an unknown POISSON_METHOD is refused."""
    mean = jnp.asarray([0.0, 1.5, 12.0, np.nan, -2.0, np.inf, 3e6, 7.0])
    for keys in (jax.random.split(jax.random.key(4), mean.size), jax.random.split(jax.random.PRNGKey(4), mean.size)):
        got = np.asarray(jax.jit(DJ._poisson_lanes)(keys, mean))
        np.testing.assert_array_equal(got, np.asarray(jax.jit(DJ._poisson_lanes)(keys, mean)))
        ref = np.asarray(jax.vmap(lambda k, m: jax.random.poisson(k, m, dtype=jnp.int64))(keys, mean))
        bad = ~np.isfinite(np.asarray(mean)) | (np.asarray(mean) < 0)
        np.testing.assert_array_equal(got[bad], ref[bad])
        assert got[0] == 0 and (got[~bad] >= 0).all() and abs(got[6] - 3e6) < 10 * np.sqrt(3e6)


def test_poisson_lanes_clusters_match_the_jax_draws_statistically(cond, monkeypatch):
    """make_clusters with POISSON_METHOD "lanes" and "jax": other draws, the same law: cluster and electron totals
    over 4,000 steps agree within 4 sigma (Poisson errors), means on both sides of the inversion / PTRS boundary;
    an unknown method is refused."""
    c = cond
    n = 4000
    rng = np.random.default_rng(5)
    st = _table(c, n, eion=2e-3)
    st["eion"] = 10 ** rng.uniform(-4.5, -2.0, n)
    st["len"] = c.meanFreePath * rng.uniform(0.1, 30.0, n)
    st["shape"][::3] = 3
    st["ion"][::2] = 1
    st["width"][:] = 0.3
    out = {}
    for method in ("jax", "lanes"):
        monkeypatch.setattr(DJ, "POISSON_METHOD", method)
        jax.clear_caches()
        out[method] = DJ.make_clusters(jax.random.key(9), c, st, np.arange(n))
    nj, nl = len(out["jax"][2]), len(out["lanes"][2])
    assert nj > n and abs(nl - nj) < 4 * np.sqrt(2 * nj), (nj, nl)
    ej, el = int(np.asarray(out["jax"][2]).sum()), int(np.asarray(out["lanes"][2]).sum())
    assert abs(el - ej) < 4 * np.sqrt(2 * ej) * 3, (ej, el)       # electrons per cluster overdisperse the total
    monkeypatch.setattr(DJ, "POISSON_METHOD", "knuth")
    jax.clear_caches()
    with pytest.raises(ValueError, match="POISSON_METHOD"):
        DJ.make_clusters(jax.random.key(9), c, st, np.arange(n))
