"""Prepared library row orders: the indexed and the per-call sorting paths give the same support, picks and steps."""
import numpy as np
import pandas as pd
import pytest

from crvfastsim import tier1b as T


@pytest.fixture(scope="module")
def geom():
    from crvfastsim import geometry
    try:
        return geometry.from_offline_config(geometry.find_counters("Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no counters file: {e}")


def _library(geom, seed=3, per_bin=5):
    """Synthetic library with rows in no particular order: several crossings per bin with tied (bin, ctx) cells,
    a variable number of descendants per crossing (some none), uv flags; xid unique but shuffled."""
    g = np.random.default_rng(seed)
    nb = len(T.GROUPS) * T.N_PBINS * T.N_LBINS
    nx = nb * per_bin
    xid = g.permutation(nx)
    lx = pd.DataFrame({"xid": xid, "bin": np.repeat(np.arange(nb), per_bin)[g.permutation(nx)],
                       "ctx": g.integers(0, 3, nx), "ux": g.normal(size=nx).astype(np.float32),
                       "uy": -np.abs(g.normal(size=nx)).astype(np.float32) - 1, "uz": g.normal(size=nx).astype(np.float32)})
    nd = g.integers(0, 4, nx)
    dx = np.repeat(lx.xid.to_numpy(), nd)
    n = len(dx)
    o = g.permutation(n)
    f32 = lambda a: a.astype(np.float32)
    ld = pd.DataFrame({"xid": dx[o], "sp": np.arange(n) + 2, "samebar": g.integers(0, 2, n), "pdg": 11,
                       "dt0": g.random(n), "dt1": g.random(n) + 1, "dx0": f32(g.normal(size=n)),
                       "dy0": f32(g.normal(size=n)), "dz0": f32(g.normal(size=n) * 50), "dx1": f32(g.normal(size=n)),
                       "dy1": f32(g.normal(size=n)), "dz1": f32(g.normal(size=n) * 50), "px0": f32(g.normal(size=n)),
                       "py0": f32(g.normal(size=n)), "pz0": f32(g.normal(size=n)), "p1": f32(g.random(n)),
                       "len": f32(g.random(n) * 10), "edep": f32(g.random(n)), "dcx": 0.0, "dcy": 0.0, "dcz": 0.0,
                       "entrylen": 0.0, "lx0": 2.0, "ly0": 3.0, "lz0": 4.0, "lx1": 2.0, "ly1": 2.0, "lz1": 4.0,
                       "uv": g.random(n) < 0.2})
    libbins = pd.DataFrame([(0, k, e) for k in range(len(T.GROUPS)) for e in (1e3, 2e3, 3e3)] +
                           [(1, k, e) for k in range(len(T.GROUPS)) for e in (5.0, 12.0, 20.0, 22.0, 25.0, 30.0)],
                           columns=["kind", "group", "edge"])
    return {"libbins": libbins, "libxing": lx, "libdesc": ld}


def _crossings(geom, seed=5, n=400):
    g = np.random.default_rng(seed)
    bars = g.integers(0, geom.nbars, n)
    c, h = geom.center[bars], geom.halflen[bars]
    top = c.copy()
    top[:, 1] += h[:, 1] - 0.1
    p = g.uniform(500, 4000, n)
    return pd.DataFrame({"run": 1400, "subrun": 1, "event": g.integers(0, 40, n), "nstep": 1, "bar": bars,
                         "sp": g.integers(1, 3, n), "t0": g.uniform(0, 1000, n), "t1": 0.0, "x0": top[:, 0],
                         "y0": top[:, 1], "z0": top[:, 2], "x1": top[:, 0], "y1": top[:, 1] - 20, "z1": top[:, 2],
                         "px0": g.normal(size=n) * 50, "py0": -p, "pz0": g.normal(size=n) * 50, "p1": p - 5,
                         "len": g.uniform(4, 28, n), "edep": 3.0})


def test_index_is_permutations_only(geom):
    lib = _library(geom)
    ind = T.index_library(lib)
    assert ind["libdesc"] is lib["libdesc"] and ind["libxing"] is lib["libxing"]
    idx = ind["index"]
    assert idx["desc_by_xid"].dtype == np.int32 and len(idx["desc_by_xid"]) == len(lib["libdesc"])
    s = T.index_library({**lib, "libdesc": lib["libdesc"].sort_values("xid", kind="stable").reset_index(drop=True)})
    assert s["index"]["desc_by_xid"] is None                  # already in order: nothing stored


@pytest.mark.parametrize("context", [False, True])
def test_support_and_picks_match_per_call_sorting(geom, monkeypatch, context):
    monkeypatch.setattr(T, "MIN_CTX_CROSSINGS", 3)
    lib = _library(geom)
    fx = _crossings(geom)
    ind = T.index_library(lib)
    a = T._support(fx, lib, geom, context, None)
    b = T._support(fx, ind, geom, context, T._indexed(ind))
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[2], b[2])
    pd.testing.assert_frame_equal(a[0], b[0], check_exact=True)
    u = np.random.default_rng(9).random(len(fx))
    pick = lambda s: s[0].xid.to_numpy()[s[1] + (u * (s[2] - s[1])).astype(np.int64)]
    np.testing.assert_array_equal(pick(a), pick(b))


@pytest.mark.parametrize("opts", [dict(), dict(context=True, contain=True, point=True),
                                  dict(context=True, contain=True, point=True, drop_unvisited=True)])
def test_descendants_identical_with_and_without_index(geom, monkeypatch, opts):
    monkeypatch.setattr(T, "MIN_CTX_CROSSINGS", 3)
    lib = _library(geom)
    fx = _crossings(geom)
    ra, rb = np.random.default_rng(21), np.random.default_rng(21)
    ta, tb = {}, {}
    a = T._desc_for_crossings(fx, lib, geom, ra, trace=ta, **opts)
    b = T._desc_for_crossings(fx, T.index_library(lib), geom, rb, trace=tb, **opts)
    pd.testing.assert_frame_equal(a[0], b[0], check_exact=True)
    assert a[1:] == b[1:]
    assert ta.keys() == tb.keys()
    for k in ta:
        np.testing.assert_array_equal(ta[k], tb[k], err_msg=k)
    np.testing.assert_array_equal(ra.random(8), rb.random(8))


def test_stale_index_falls_back(geom):
    lib = T.index_library(_library(geom))
    stale = {**lib, "libdesc": lib["libdesc"].iloc[:-1]}
    assert T._indexed(stale) is None and T._indexed({k: v for k, v in lib.items() if k != "index"}) is None


def test_same_length_reordered_frames_fall_back_and_match(geom, monkeypatch):
    monkeypatch.setattr(T, "MIN_CTX_CROSSINGS", 3)
    lib = T.index_library(_library(geom))
    fx = _crossings(geom)
    g = np.random.default_rng(1)
    for name in ("libdesc", "libxing"):
        moved = {**lib, name: lib[name].iloc[g.permutation(len(lib[name]))].reset_index(drop=True)}
        assert T._indexed(moved) is None
        per_call = {k: v for k, v in moved.items() if k != "index"}
        for opts in (dict(), dict(context=True, contain=True, point=True, drop_unvisited=True)):
            a = T._desc_for_crossings(fx, per_call, geom, np.random.default_rng(4), **opts)
            b = T._desc_for_crossings(fx, moved, geom, np.random.default_rng(4), **opts)
            pd.testing.assert_frame_equal(a[0], b[0], check_exact=True)
            assert a[1:] == b[1:]


def test_order_check_is_exact():
    k = np.array([3, 1, 3, 2, 1])
    o = np.argsort(k, kind="stable")
    assert T._is_stable_order(o, k)
    assert not T._is_stable_order(o[[1, 0, 2, 3, 4]], k)      # ties out of row order
    assert not T._is_stable_order(None, k) and T._is_stable_order(None, np.sort(k))
    c = np.array([1, 0, 0, 1, 0])
    o2 = np.lexsort((c, k))
    assert T._is_stable_order(o2, c, k) and not T._is_stable_order(o, c, k)
