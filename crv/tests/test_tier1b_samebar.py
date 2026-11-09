"""Opt-in same-bar descendant options of tier 1b: chord placement and the follow chain draw (template sidecar v3)."""
import numpy as np
import pandas as pd
import pytest

from crvfastsim import tier1b as T

from test_tier1b_templates import _gen, _lib, _parts, _template_side, geom  # noqa: F401  (fixture)
from test_tier1b_topology import _crossings, _ex


def test_chord_map_keeps_fraction_inside_and_distance_outside():
    cl = np.array([[0.0, -20.0, 0.0]] * 4)                  # library: vertical, 20 mm
    cf = np.array([[30.0, -40.0, 0.0]] * 4)                 # fast: 50 mm, inclined
    d0 = np.array([[0.0, -10.0, 0.0],                       # half way along the library chord
                   [0.0, 5.0, 0.0],                         # 5 mm before the entry
                   [0.0, -26.0, 0.0],                       # 6 mm beyond the exit
                   [2.0, -10.0, 0.0]])                      # half way, 2 mm off the chord
    d1 = d0 + np.array([0.0, -1.0, 0.0])
    ok, s0, s1, mom = T._chord_map(d0, d1, np.array([[0.0, -1.0, 0.0]] * 4), cl, cf, np.ones(4, bool))
    uf = cf[0] / 50.0
    assert ok.all()
    assert np.allclose(s0[0], 25.0 * uf)
    assert np.allclose(s0[1], -5.0 * uf)
    assert np.allclose(s0[2], 56.0 * uf)
    assert np.isclose(np.linalg.norm(s0[3] - 25.0 * uf), 2.0) and np.isclose((s0[3] - 25.0 * uf) @ uf, 0.0)
    assert np.allclose(s1 - s0, uf) and np.allclose(mom, uf)  # step vector and momentum rotated chord to chord
    ok2, *_ = T._chord_map(d0, d1, d1, np.zeros((4, 3)), cf, np.ones(4, bool))
    assert not ok2.any()                                     # degenerate library chord: left to the default placement


def test_next_crossing():
    from valhist.descendant_templates import next_crossing
    mu = np.array([0, 0, 1, 0, 1])
    t0 = np.array([5.0, 1.0, 2.0, 3.0, 1.0])
    assert list(next_crossing(mu, t0, 10)) == [-1, 13, -1, 10, 12]


def test_follow_picks_chain_within_cell():
    fx = pd.DataFrame({"run": 1400, "subrun": 1, "event": 1, "sp": 1, "t0": [3.0, 1.0, 2.0]})
    # sorted library order = xid order here; cells: crossing at t=1 [0, 4), t=2 [4, 8), t=3 [0, 4)
    lo, hi = np.array([0, 0, 4]), np.array([4, 4, 8])
    nxt = np.full(8, -1, np.int64)
    nxt[2] = 6                                               # library crossing 2 is followed by 6 (in [4, 8))
    nxt[6] = 7                                               # 6 by 7, in the wrong cell for the third crossing
    pick, n = T._follow_picks(fx, np.arange(8), lo, hi, np.array([1, 2, 5]), nxt, np.arange(8))
    assert list(pick) == [1, 2, 6] and n == 1                # t=2 follows 2 -> 6; t=3 keeps its own pick (7 not in cell)


def _lib3(geom):
    lib = _lib(geom, templates=False)
    side = _template_side(geom, lib)
    xs = side["xing"]
    bar = xs.bar.to_numpy()
    exit_ = geom.center[bar] + np.array([100.0, -geom.halflen[bar[0], 1], 0.0])
    side["xing"] = xs.assign(qx=exit_[:, 0], qy=exit_[:, 1], qz=exit_[:, 2], len=2 * geom.halflen[bar[0], 1],
                             nxt=-1)
    return T.attach_templates(lib, side, geom), T.attach_templates(lib, _template_side(geom, lib), geom)


def test_chord_follow_inert_when_off(geom):
    lib3, lib2 = _lib3(geom)
    fx = _crossings(geom, [_ex(geom, 3, 9), _ex(geom, 2, 9)], event=1)
    gen, parts = _gen([1], x=fx.x0.iloc[0]), _parts([1])
    for opts in ({}, {"topology": True}, {"topology": True, "unvisited": True, "nearmiss": True}):
        r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
        a = T.sample(fx, parts, gen, lib2, geom, r1, **opts)
        b = T.sample(fx, parts, gen, lib3, geom, r2, **opts)
        pd.testing.assert_frame_equal(a[0], b[0])
        pd.testing.assert_frame_equal(a[1], b[1])
        assert r1.random() == r2.random()
    with pytest.raises(ValueError):
        T.sample(fx, parts, gen, lib2, geom, np.random.default_rng(0), chord=True)
    with pytest.raises(ValueError):
        T.sample(fx, parts, gen, lib2, geom, np.random.default_rng(0), follow=True)


def test_chord_moves_only_same_bar_rows_and_no_random_number(geom):
    lib3, _ = _lib3(geom)
    lib3["libdesc"] = pd.concat([lib3["libdesc"], lib3["libdesc"].assign(samebar=1, dx0=0.0, dy0=-9.9, dz0=0.0,
                                                                          dx1=0.0, dy1=-10.9, dz1=0.0)],
                                ignore_index=True).sort_values("xid", kind="stable").reset_index(drop=True)
    fx = _crossings(geom, [_ex(geom, 3, 9)], event=1)
    fx[["x1"]] += 15.0                                       # fast chord inclined across the bar
    r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
    a = T._desc_for_crossings(fx, lib3, geom, r1, contain=True)
    b = T._desc_for_crossings(fx, lib3, geom, r2, contain=True, chord=True)
    assert r1.random() == r2.random() and b[3]["chord"] == 1
    oa, ob = a[0], b[0]
    same_a, same_b = oa[oa.bar == fx.bar.iloc[0]], ob[ob.bar == fx.bar.iloc[0]]
    pd.testing.assert_frame_equal(oa[oa.bar != fx.bar.iloc[0]], ob[ob.bar != fx.bar.iloc[0]])
    # half way down the library chord (9.9 of 19.8 mm) -> half way along the fast chord
    e = fx[["x0", "y0", "z0"]].to_numpy()[0]
    q = fx[["x1", "y1", "z1"]].to_numpy()[0]
    assert np.allclose(same_b[["x0", "y0", "z0"]].to_numpy()[0], e + 0.5 * (q - e))
    assert not np.allclose(same_a[["x0", "y0", "z0"]].to_numpy()[0], e + 0.5 * (q - e))
