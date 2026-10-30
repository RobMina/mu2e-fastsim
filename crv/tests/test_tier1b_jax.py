"""Strict-v3 device fixed-uniform contracts; run in the project image on a compute node."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jax")

from crvfastsim import tier1b as T
from crvfastsim.tier1b_jax import DeviceRandom, DeviceTier1b, ReplayRandom, _closest, _jax, _main_select, _uv_select
from valhist.device_tier1b import RecordRandom, compare_frames
from test_tier1b_samebar import _lib3
from test_tier1b_templates import _gen, _parts, geom
from test_tier1b_topology import _crossings, _ex

OPTS = dict(context=True, contain=True, point=True, topology=True, unvisited=True, nearmiss=True, chord=True)


def test_main_support_context_fallback():
    jax, j = _jax()
    lib = dict(code=j.asarray([0] * 20 + [1] * 3 + [8] * 20),
               bin=j.asarray([0] * 23 + [1] * 20), order=j.arange(43))
    p, lo, hi = jax.jit(_main_select)(lib, j.asarray([0, 0, 1, 2]), j.asarray([0, 1, 0, 0]), j.asarray([.5, .99, 0., .5]))
    np.testing.assert_array_equal(lo, [0, 0, 23, 43])
    np.testing.assert_array_equal(hi, [20, 23, 43, 43])
    np.testing.assert_array_equal(p[:3], [10, 22, 23])


def test_uv_finest_supported_else_coarsest_nonempty():
    jax, j = _jax()
    lib = {}
    for level, n in enumerate([2, 20, 25, 30, 40]):
        lib[f"code{level}"], lib[f"order{level}"] = j.ones(n, j.int64), j.arange(n)
    cells = j.asarray([[1, 1, 1, 1, 1], [9, 9, 9, 9, 1], [9, 9, 9, 9, 9]])
    p, lev = jax.jit(_uv_select)(lib, cells, j.asarray([.5, .5, .5]))
    np.testing.assert_array_equal(lev, [1, 4, -1])
    np.testing.assert_array_equal(p, [10, 20, -1])


def test_replay_shape_and_exhaustion():
    r = ReplayRandom([np.zeros(2)])
    with pytest.raises(ValueError, match="shape"):
        r.random(1)
    with pytest.raises(ValueError, match="exhausted"):
        r.random(0)


def test_device_random_repeatable_and_uniform():
    a, b = DeviceRandom(4), DeviceRandom(4)
    x, y = np.asarray(a.random(10000)), np.asarray(b.random(10000))
    np.testing.assert_array_equal(x, y)
    assert x.dtype == np.float64 and ((x >= 0) & (x < 1)).all()
    assert abs(x.mean() - .5) < .02
    assert not np.array_equal(x, a.random(10000))


def test_closest_replays_reference_with_parallel_and_entering_rays(geom):
    jax, j = _jax()
    rng = np.random.default_rng(1)
    pos = np.c_[rng.uniform(-9000, 2000, 100), np.full(100, 6000.), rng.uniform(19000, 27000, 100)]
    u = np.c_[rng.normal(0, .4, 100), -np.ones(100), rng.normal(0, .4, 100)]
    u[:2] = [0, -1, 0]
    a = T.closest_approach(pos, u, geom)
    b = jax.jit(_closest)(j.asarray(pos), j.asarray(u), *[j.asarray(x) for x in T.group_boxes(geom)])
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1], rtol=1e-11, atol=1e-8)
    np.testing.assert_array_equal(T.nearmiss_cell(np.full(len(pos), 1000.), a[0], a[1]),
                                  T.nearmiss_cell(np.full(len(pos), 1000.), np.asarray(b[0]), np.asarray(b[1])))
    lo, hi = T.group_boxes(geom)
    q = pos + np.asarray(b[2])[:, None] * T._unit(u)
    objective = np.linalg.norm(np.maximum(np.maximum(lo[np.asarray(b[0])] - q, 0.), q - hi[np.asarray(b[0])]), axis=1)
    np.testing.assert_allclose(objective, a[1], rtol=1e-11, atol=1e-8)


@pytest.mark.parametrize("empty", [False, True])
def test_synthetic_strict_v3_replay_and_padding(geom, empty):
    lib, _ = _lib3(geom)
    if "ctx" not in lib["libxing"]:
        lib["libxing"]["ctx"] = 0
    fx = _crossings(geom, [_ex(geom, 3, 9), _ex(geom, 2, 9)], event=1)
    fx["x1"] += 15.
    if empty:
        fx = fx.iloc[:0]
    gen = _gen([1, 2, 3])
    parts = _parts([1]) if not empty else _parts([])
    rr = RecordRandom(7)
    host = T.sample(fx, parts, gen, lib, geom, rr, **OPTS)
    dev = DeviceTier1b(lib, geom, batch_size=3)
    got = dev.sample(fx, parts, gen, ReplayRandom(rr.draws), **OPTS)
    compare_frames(host[0], got[0])
    compare_frames(host[1], got[1])
    assert host[2]["desc_steps"] == got[2]["desc_steps"]
    other = DeviceTier1b(lib, geom, batch_size=7).sample(fx, parts, gen, ReplayRandom(rr.draws), **OPTS)
    compare_frames(got[0], other[0])
    pd.testing.assert_frame_equal(got[1], other[1])
    from crvfastsim.tier1b_metadata_jax import crossing_arrays
    jax, j = _jax()
    arrays = crossing_arrays(fx)
    selection = dev.plan_descendants(arrays, j.zeros(len(arrays["bar"])))
    assert all(isinstance(a, jax.Array) for a in jax.tree.leaves(selection))
    blocks = list(dev.device_blocks(selection))
    assert sum(int(b["valid"].sum()) for b in blocks) == int(selection["plan"]["total"])
    assert all(isinstance(a, jax.Array) for b in blocks for a in b.values())
    for b in blocks:
        keep = np.asarray(b["valid"])
        ix, rows = np.asarray(b["ix"])[keep], np.asarray(b["rows"])[keep]
        for c in T.KEY:
            assert b[c].dtype == j.uint32
            np.testing.assert_array_equal(np.asarray(b[c])[keep], fx[c].to_numpy()[ix])
        assert b["pdg"].dtype == j.int32
        np.testing.assert_array_equal(np.asarray(b["lsp"])[keep], dev.ld.sp.to_numpy()[rows])
        np.testing.assert_array_equal(np.asarray(b["parent"])[keep], fx.sp.to_numpy()[ix])
        np.testing.assert_array_equal(np.asarray(b["src"])[keep], np.zeros(len(rows), np.int64))
        np.testing.assert_allclose(np.asarray(b["t0"])[keep], fx.t0.to_numpy()[ix] + dev.ld.dt0.to_numpy()[rows], rtol=0, atol=1e-13)
    with pytest.raises(ValueError, match="uniforms"):
        dev.plan_descendants(arrays, j.zeros(len(arrays["bar"]) + 1))
    with pytest.raises(ValueError, match="shape"):
        dev.device_metadata(dict(arrays, mom=j.zeros((len(arrays["bar"]), 2))))


def test_unsupported_options_fail_closed(geom):
    lib, _ = _lib3(geom)
    if "ctx" not in lib["libxing"]:
        lib["libxing"]["ctx"] = 0
    dev = DeviceTier1b(lib, geom)
    with pytest.raises(ValueError, match="strict_v3"):
        dev.sample(None, None, None, None, **dict(OPTS, follow=True))
    with pytest.raises(ValueError, match="positive"):
        DeviceTier1b(lib, geom, batch_size=0)
