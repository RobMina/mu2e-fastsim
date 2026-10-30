"""Sparse/known hard rates, absent catalogues and the exact 50-redraw limit."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jax")
from crvfastsim import tier1b as T
from crvfastsim.tier1b_jax import DeviceTier1b, ReplayRandom
from crvfastsim.tier1b_metadata_jax import crossing_arrays
from test_tier1b_hard_jax import _hard_library
from test_tier1b_templates import geom
from test_tier1b_topology import _crossings, _ex
from valhist.device_tier1b import RecordRandom


@pytest.mark.parametrize("nlib,nhard,pn", [(49, 0, 1.), (50, 50, 0.), (50, 0, 1.)])
def test_exact_rate_key_threshold_and_complement(geom, nlib, nhard, pn):
    lib = _hard_library(geom)
    lib["hard"]["rate_key"]["nlib"] = nlib
    lib["hard"]["rate_key"]["nhard"] = nhard
    lib["hard"]["rate_pn"]["rate_pn"] = pn
    dev = DeviceTier1b(lib, geom)
    fx = _crossings(geom, [_ex(geom, 3, 9), _ex(geom, 2, 9)])
    x = dev.device_metadata(crossing_arrays(fx))[0]
    pat = dev.hard_patterns(x, dev.metadata_constants, dev.hard)
    expected = nhard / nlib if nlib >= 50 else pn
    assert float(pat["rate"][0]) == expected


def test_absent_catalogue_keeps_full_support(geom):
    lib = _hard_library(geom)
    lib["hard"]["kcat"] = lib["hard"]["kcat"].iloc[:0]
    dev = DeviceTier1b(lib, geom)
    fx = _crossings(geom, [_ex(geom, 3, 9), _ex(geom, 2, 9)])
    x = crossing_arrays(fx)
    sel = dev.plan_descendants(x, np.zeros(len(x["bar"])))
    got, info = dev._hard_picks(sel, ReplayRandom([np.zeros(1), np.zeros(1)]))
    np.testing.assert_array_equal(got["picks"], sel["picks"])
    assert info["hard"] == info["redrawn"] == 0
    assert info["no_key"] > 0


def test_all_hard_support_retains_reference_residual_after_50(geom):
    lib = _hard_library(geom)
    lib["hard"]["is_hard_xing"][:] = True
    lib["hard"]["rate_pn"]["rate_pn"] = 0.
    lib["hard"]["rate_key"]["nhard"] = 0
    dev = DeviceTier1b(lib, geom)
    fx = pd.concat([_crossings(geom, [_ex(geom, 3, 9), _ex(geom, 2, 9)], event=3),
                    _crossings(geom, [_ex(geom, 3, 9), _ex(geom, 2, 9)], event=1)], ignore_index=True)
    x = crossing_arrays(fx)
    sel = dev.plan_descendants(x, np.zeros(len(x["bar"])))
    pick, lo, hi = [np.asarray(sel[c])[:len(fx)] for c in ("picks", "lo", "hi")]
    rr = RecordRandom(842)
    ref, expected = T._hard_picks(fx, geom, lib, pick, lo, hi, np.asarray(dev.support["order"]), rr)
    assert len(rr.draws) == 52
    replay = ReplayRandom(rr.draws)
    got, info = dev._hard_picks(sel, replay)
    np.testing.assert_array_equal(np.asarray(got["picks"])[:len(fx)], ref)
    assert replay.index == 52 and info["left_hard"] == expected["left_hard"] == len(fx)
