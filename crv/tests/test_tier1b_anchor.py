"""Opt-in root-coherent tier-1b draws: step-tree roots, anchors, forward rows and the linked chain draw."""
import numpy as np
import pandas as pd
import pytest

from crvfastsim import tier1b as T

from test_tier1b_templates import _gen, _lib, _parts, _template_side, geom  # noqa: F401  (fixture)
from test_tier1b_topology import _crossings, _ex


def test_step_roots():
    p = pd.DataFrame({"run": 1, "subrun": 1, "event": 1, "key": [1, 2, 3, 4, 5], "parent": [-1, 1, 2, 3, 1]})
    assert list(T.step_roots(p)) == [1, 2, 2, 2, 5]
    two = pd.concat([p, p.assign(event=2)], ignore_index=True)
    assert list(T.step_roots(two)) == [1, 2, 2, 2, 5] * 2


def _tiny(geom):
    """Muon 0 crosses 0 -> 1 -> 2 (EX layers 3, 2, 1), muon 1 crosses 3. Tree 9 touches 0, 1, 2; tree 8 only 1; tree 7 on 3."""
    bars = [_ex(geom, 3, 9), _ex(geom, 2, 9), _ex(geom, 1, 9), _ex(geom, 3, 4)]
    lx = pd.DataFrame({"xid": np.arange(4), "bin": [0, 0, 0, 0], "ctx": [0, 2, 4, 0]})
    ld = pd.DataFrame({"xid": [0, 1, 1, 2, 3], "root": [9, 9, 8, 9, 7], "edep": [1.0, 2.0, 3.0, 4.0, 5.0],
                       "samebar": 1})
    g = {"mu": np.array([0, 0, 0, 1]), "nxt": np.array([1, 2, -1, -1]), "bar": np.array(bars),
         "p": np.full(4, 3000.0)}
    return {"libxing": lx, "libdesc": ld, "xgeo": g}, bars


def test_attach_anchors(geom):
    lib, _ = _tiny(geom)
    A = T.attach_anchors(lib, geom)["anchor"]
    assert list(A["axid"]) == [0, 0, 1, 0, 3]
    assert list(A["dk"]) == [0, 1, 0, 2, 0]
    assert list(A["joined"]) == [True, True, False, False]
    assert list(A["entered"]) == [False, True, True, False]
    assert A["n_roots"] == 3 and A["n_multi"] == 1
    assert np.isclose(A["forward_edep_share"], 6.0 / 15.0)


def test_anchor_rows_follow_the_muon_and_drop_without_a_crossing(geom):
    lib, bars = _tiny(geom)
    lib = T.attach_anchors(lib, geom)
    sym = lib["anchor"]["sym"][:3]
    # fast muon with the library pattern: picks 0 and 1 send the forward rows along it
    cur, rows, rx, free, info = T._anchor_rows(np.array([0, 1, 3]), np.array([1, 2, -1]), sym, lib)
    assert sorted(zip(cur, rows)) == [(0, 0), (1, 1), (1, 2), (2, 3), (2, 4)]
    assert info["forward_free"] == 0 and not free.any()
    np.testing.assert_array_equal(rx, lib["libdesc"].xid.to_numpy()[rows])
    # the fast muon ends after its first crossing: forward rows stay on the anchor as free rows
    cur, rows, _, free, info = T._anchor_rows(np.array([0]), np.array([-1]), sym[:1], lib)
    assert list(cur) == [0, 0, 0] and list(rows) == [0, 1, 3] and list(free) == [False, True, True]
    assert info["forward_edep_free"] == 6.0
    # the distance differs but the symbol is further along the muon
    long_ = np.array([sym[0], -1, sym[1], sym[2]])                       # -1: a symbol no library crossing has
    cur, rows, _, free, _ = T._anchor_rows(np.array([0, 3, 3, 3]), np.array([1, 2, 3, -1]), long_, lib)
    assert sorted(zip(cur, rows)) == [(0, 0), (1, 4), (2, 1), (2, 4), (3, 3), (3, 4)] and not free.any()


def test_anchor_rows_unfed_crossing_keeps_rows_from_behind(geom):
    lib, _ = _tiny(geom)
    lib = T.attach_anchors(lib, geom)
    s = lib["anchor"]["sym"]
    # the first pick (crossing 3) has no next library crossing, so it does not feed the second (pick 1), which keeps the
    # row tree 9 brings into crossing 1 from behind, next to its own tree 8
    cur, rows, rx, free, info = T._anchor_rows(np.array([3, 1]), np.array([1, -1]), np.array([s[3], s[1]]), lib)
    assert sorted(zip(cur, rows)) == [(0, 4), (1, 1), (1, 2)] and not free.any()
    assert info["unfed"] == 2 and info["kept_entered"] == 1 and info["kept_entered_edep"] == 2.0
    np.testing.assert_array_equal(rx, lib["libdesc"].xid.to_numpy()[rows])
    # fed by pick 0 (whose next crossing 1 has the symbol), the same pick takes tree 9 only through the chain
    # (the muon ends there, so the tree's third-layer row stays free on the anchor)
    cur, rows, _, free, info = T._anchor_rows(np.array([0, 1]), np.array([1, -1]), np.array([s[0], s[1]]), lib)
    assert sorted(zip(cur, rows)) == [(0, 0), (0, 3), (1, 1), (1, 2)] and info["kept_entered"] == 0
    assert list(free[np.argsort(rows)]) == [False, False, False, True]


def test_link_picks(geom):
    lib, bars = _tiny(geom)
    lib = T.attach_anchors(lib, geom)
    A = lib["anchor"]
    fx = _crossings(geom, bars[:3])
    nxt, sym = T._next_crossing(fx, geom), T._symbols(geom, fx.bar.to_numpy())
    lo, hi = np.zeros(3, np.int64), np.full(3, 4, np.int64)
    pick, info = T._link_picks(fx, geom, lib, np.array([0, 3, 3]), lo, hi, np.arange(4), nxt, sym, np.random.default_rng(1))
    assert list(pick) == [0, 1, 2] and info["linked"] == 2                 # the tree joins all three
    pick, info = T._link_picks(fx, geom, lib, np.array([3, 1, 2]), lo, hi, np.arange(4), nxt, sym, np.random.default_rng(1))
    assert pick[0] == 3 and info["linked"] == 0                            # no tree: redraws avoid entered crossings
    assert not A["entered"][pick].any()


def _rootlib(geom):
    lib = _lib(geom, templates=False)
    n = len(lib["libxing"])
    lib["libdesc"] = lib["libdesc"].assign(root=np.arange(len(lib["libdesc"])) + 1000)
    lib["libxing"] = lib["libxing"].assign(ctx=0)
    side = _template_side(geom, lib)
    xs = side["xing"]
    bar = xs.bar.to_numpy()
    exit_ = geom.center[bar] + np.array([100.0, -geom.halflen[bar[0], 1], 0.0])
    side["xing"] = xs.assign(qx=exit_[:, 0], qy=exit_[:, 1], qz=exit_[:, 2], len=2 * geom.halflen[bar[0], 1], nxt=-1)
    return T.attach_anchors(T.attach_templates(lib, side, geom), geom), n


def test_inert_without_option_and_same_as_default_without_trees(geom):
    fx = _crossings(geom, [_ex(geom, 3, 9), _ex(geom, 2, 9)], event=1)
    gen, parts = _gen([1], x=fx.x0.iloc[0]), _parts([1])
    lib, _ = _rootlib(geom)
    plain = _lib(geom, templates=False)
    a = T.sample(fx, parts, gen, plain, geom, np.random.default_rng(3))
    b = T.sample(fx, parts, gen, lib, geom, np.random.default_rng(3))
    cols = ["bar", "t0", "edep", "x0", "y0", "z0"]
    pd.testing.assert_frame_equal(a[0][cols], b[0][cols])
    # one-crossing trees only: the anchored draw takes each pick's own rows
    c = T.sample(fx, parts, gen, lib, geom, np.random.default_rng(3), context=True, anchor=True)
    assert len(c[0]) >= len(fx)


def test_module_scope_cuts_a_tree_at_the_module(geom):
    lib, bars = _tiny(geom)
    lib["xgeo"]["bar"] = np.array([bars[0], bars[1], _ex(geom, 1, 9, module=2), bars[3]])
    A = T.attach_anchors(lib, geom)["anchor"]
    assert list(A["axid"]) == [0, 0, 1, 2, 3] and list(A["dk"]) == [0, 1, 0, 0, 0]   # tree 9 restarts in module 2
    W = T.attach_anchors(lib, geom, scope="muon")["anchor"]
    assert list(W["axid"]) == [0, 0, 1, 0, 3] and list(W["dk"]) == [0, 1, 0, 2, 0]
    with pytest.raises(ValueError):
        T.attach_anchors(lib, geom, scope="event")


def test_default_set_options_deterministic_per_seed(geom):
    """The default set's sampling options on a roots library: the same seed gives the same tables, row for row."""
    from crvfastsim import tier1b_assets
    opts = {k: v for k, v in tier1b_assets.SETS[tier1b_assets.DEFAULT].options.items() if k != "topology"}
    assert opts.get("anchor") and not opts.get("hard")
    fx = _crossings(geom, [_ex(geom, 3, 9), _ex(geom, 2, 9), _ex(geom, 1, 9)], event=1)
    gen, parts = _gen([1], x=fx.x0.iloc[0]), _parts([1])
    lib, _ = _rootlib(geom)
    a = T.sample(fx, parts, gen, lib, geom, np.random.default_rng([5, 1000]), **opts)
    b = T.sample(fx, parts, gen, lib, geom, np.random.default_rng([5, 1000]), **opts)
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])
    assert len(a[0]) > len(fx)
