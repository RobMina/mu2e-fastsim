"""Device metadata preserves reference grouping, visit ordering and support cells."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jax")

from crvfastsim import tier1b as T
from crvfastsim.tier1b_jax import _jax
from crvfastsim.tier1b_metadata_jax import crossing_arrays, metadata, prepare
from test_tier1b_topology import _crossings, _ex, geom


def _bins():
    return pd.DataFrame([(g, k, e) for g in range(4) for k in range(2)
                         for e in ([150., 300., 1500.] if k == 0 else [1., 2., 10., 20., 40., 80.])],
                        columns=["group", "kind", "edge"])


def assert_metadata(fx, geom):
    jax, _ = _jax()
    bins = _bins()
    x, v, meta = jax.jit(metadata)(crossing_arrays(fx), prepare(geom, bins))
    mu, vis, visits = T.muon_visits(fx, geom)
    n = len(fx)
    for name, expected in (("mu", mu), ("vis", vis)):
        np.testing.assert_array_equal(np.asarray(x[name])[:n], expected, err_msg=name)
    finite = np.asarray(v["key"]) != np.iinfo(np.int64).max
    np.testing.assert_array_equal(np.asarray(v["key"])[finite], visits.mu.astype(np.int64) * 100000 + visits.vis.astype(np.int64))
    for name in ("lid", "wmin", "wmax"):
        np.testing.assert_array_equal(np.asarray(v[name])[finite], visits[name], err_msg=name)
    np.testing.assert_array_equal(np.asarray(v["entry"])[finite], visits[["ex", "ey", "ez"]].to_numpy())
    expected_vkey = np.unique(visits.mu.to_numpy().astype(np.int64) * 1000000 + visits.lid.to_numpy().astype(np.int64))
    np.testing.assert_array_equal(np.unique(np.asarray(v["vkey"])[np.asarray(v["vkey"]) != np.iinfo(np.int64).max]), expected_vkey)
    p = np.linalg.norm(fx[["px0", "py0", "pz0"]].to_numpy(), axis=1)
    cells = T.uv_cells(fx.bar.to_numpy(), fx[["x0", "y0", "z0"]].to_numpy(), p,
                       T.last_crossing(fx), T.visit_mask(fx, geom), geom)
    for name, expected in (("ctx", T.crossing_context(fx, geom)), ("last", T.last_crossing(fx)),
                           ("mask", T.visit_mask(fx, geom)), ("cells", cells),
                           ("bins", T._bin(bins, T._group_index(geom, fx.bar), p, fx.len.to_numpy()))):
        np.testing.assert_array_equal(np.asarray(meta[name])[:n], expected, err_msg=name)
    assert not np.asarray(meta["valid"])[n:].any()


@pytest.mark.parametrize("n", [0, 1, 3, 63, 257])
def test_random_order_multi_muon_and_padding(geom, n):
    rng = np.random.default_rng(421)
    fx = _crossings(geom, rng.integers(0, geom.nbars, n))
    fx["event"] = rng.choice([9, 2, 87], n)
    fx["run"] = rng.choice([1400, 1399], n)
    fx["sp"] = rng.choice([1, 7], n)
    fx["t0"] = rng.integers(0, 12, n).astype(float)
    fx["px0"] = rng.uniform(-3000, 3000, n)
    fx["len"] = rng.choice([1., 2., 10., 20., 40., 80.], n)
    assert_metadata(fx, geom)


def test_holes_in_valid_lanes_do_not_add_muons(geom):
    jax, j = _jax()
    fx = _crossings(geom, [_ex(geom, 3, 4), _ex(geom, 2, 4), _ex(geom, 1, 4)])
    arr = crossing_arrays(fx)
    take = j.asarray([0, 0, 1, 0, 2, 0, 0, 0])
    padded = {k: v[take] for k, v in arr.items()}
    padded["valid"] = j.asarray([True, False, True, False, True, False, False, False])
    base = jax.jit(metadata)(arr, prepare(geom, _bins()))
    got = jax.jit(metadata)(padded, prepare(geom, _bins()))
    for name in ("mu", "vis"):
        np.testing.assert_array_equal(np.asarray(got[0][name])[[0, 2, 4]], np.asarray(base[0][name])[:3])
    for name in ("ctx", "bins", "cells", "last", "mask"):
        np.testing.assert_array_equal(np.asarray(got[2][name])[[0, 2, 4]], np.asarray(base[2][name])[:3])


def test_visit_reentry_time_ties_and_upstream_boundary(geom):
    bars = [_ex(geom, 3, 4, module=0), _ex(geom, 3, 5, module=0), _ex(geom, 2, 5, module=1),
            _ex(geom, 3, 6, module=0), _ex(geom, 1, 5, module=1)]
    fx = _crossings(geom, bars)
    fx["t0"] = [1., 1., 2., 3., 3.]
    fx[["x0", "y0", "z0"]] = [[0., 0., 0.], [0., 0., 0.], [299.999, 0., 0.],
                                     [1000., 0., 0.], [300., 0., 0.]]
    assert_metadata(fx, geom)
    shuffled = pd.concat([fx.assign(event=8, sp=7), fx.assign(event=2)], ignore_index=True).iloc[[6, 0, 8, 3, 5, 1, 9, 4, 2, 7]]
    assert_metadata(shuffled, geom)
