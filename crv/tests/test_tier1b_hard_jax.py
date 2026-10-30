"""Strict-v4 donor law: exact pattern/rate fallback and complete bounded redraw replay."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jax")
from crvfastsim import tier1b as T
from crvfastsim.tier1b_jax import DeviceTier1b, ReplayRandom
from crvfastsim.tier1b_metadata_jax import crossing_arrays
from crvfastsim.tier1b_hard_jax import lookup
from valhist.device_tier1b import RecordRandom
from test_tier1b_jax import _lib3, _jax
from test_tier1b_templates import geom
from test_tier1b_topology import _crossings, _ex


def _hard_library(geom):
    lib, _ = _lib3(geom)
    lib["libxing"]["ctx"] = 0
    n = len(lib["libxing"])
    assert n % 2 == 0
    lib["xgeo"].update(mu=np.repeat(np.arange(n // 2), 2), nxt=np.where(np.arange(n) % 2 == 0, np.arange(n) + 1, -1),
                       p=np.full(n, 3000.), bar=np.array([_ex(geom, 3, 4), _ex(geom, 2, 4)] * (n // 2)))
    lib["libdesc"]["edep"] = np.where(lib["libdesc"].xid < 2, 100., 0.)
    return T.attach_hard(lib, geom)


def test_tuple_lookup_no_lossy_key_packing():
    jax, j = _jax()
    keys = j.asarray([[0, 99, 0], [1, 0, 0], [1, 17**11, 3], [4, 5, 1]])
    want = j.asarray([[0, 99, 0], [1, 17**11, 2], [1, 17**11, 3], [5, 0, 0]])
    idx, ok = jax.jit(lookup)(keys, want)
    np.testing.assert_array_equal(idx, [0, 2, 2, 3])
    np.testing.assert_array_equal(ok, [True, False, True, False])


@pytest.mark.parametrize("n,force", [(0, None), (5, None), (8, 0.), (8, 1.)])
def test_hard_selection_exact_uniform_replay(geom, n, force):
    lib = _hard_library(geom)
    frames = [_crossings(geom, [_ex(geom, 3, 9), _ex(geom, 2, 9)], event=10-i) for i in range(n)]
    fx = pd.concat(frames, ignore_index=True) if frames else _crossings(geom, []).astype({"sp": np.int64})
    if force is not None:
        lib["hard"]["rate_pn"]["rate_pn"] = force
    dev = DeviceTier1b(lib, geom, batch_size=3)
    arrays = crossing_arrays(fx)
    selection = dev.plan_descendants(arrays, np.zeros(len(arrays["bar"])))
    picks, lo, hi = [np.asarray(selection[c])[:len(fx)] for c in ("picks", "lo", "hi")]
    rr = RecordRandom(14)
    ref, info = T._hard_picks(fx, geom, lib, picks, lo, hi, np.asarray(dev.support["order"]), rr)
    replay = ReplayRandom(rr.draws)
    got, actual = dev._hard_picks(selection, replay)
    assert replay.index == len(rr.draws)
    np.testing.assert_array_equal(np.asarray(got["picks"])[:len(fx)], ref)
    for c in info:
        assert actual[c] == pytest.approx(info[c], abs=1e-13)


def test_patterns_more_than_24_crossings_have_zero_rate(geom):
    lib = _hard_library(geom)
    dev = DeviceTier1b(lib, geom)
    fx = _crossings(geom, [_ex(geom, 3, 9)] * 25)
    arrays = crossing_arrays(fx)
    xx = dev.device_metadata(arrays)[0]
    pat = dev.hard_patterns(xx, dev.metadata_constants, dev.hard)
    assert int(pat["count"][0]) == 25
    assert float(pat["rate"][0]) == 0.
