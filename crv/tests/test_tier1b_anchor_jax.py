"""Device twin of the root-coherent (anchored) tier-1b draws of strict_v5: fixed-uniform replay against tier1b.sample."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jax")

from crvfastsim import tier1b as T
from crvfastsim.tier1b_jax import DeviceTier1b, ReplayRandom, _anchor_rows, _jax
from valhist.device_tier1b import RecordRandom, compare_frames
from test_tier1b_samebar import _lib3
from test_tier1b_templates import _gen, _parts, geom  # noqa: F401  (fixture)
from test_tier1b_topology import _crossings, _ex
from test_tier1b_anchor import _tiny

OPTS = dict(context=True, contain=True, point=True, topology=True, unvisited=True, nearmiss=True, chord=True, anchor=True)


def _anchor_lib(geom):
    """_lib3 with a same-bar and an other-bar row per crossing; library crossings 2k -> 2k+1 form one muon (EX layers 3
    and 2 of module 1) whose same-bar rows share a step tree, so 2k anchors the tree and 2k+1 is entered from behind."""
    lib, _ = _lib3(geom)
    lib["libxing"] = lib["libxing"].assign(ctx=0)
    ld = lib["libdesc"]
    xid = ld.xid.to_numpy()
    same = ld.assign(samebar=1, uv=False, sp=8, edep=1.0 + xid % 3, dx0=0.0, dy0=-5.0, dz0=0.0, dx1=0.0, dy1=-6.0,
                     dz1=0.0, root=100 + xid // 2)
    other = ld.assign(uv=False, sp=9, edep=2.0, root=10000 + xid)
    lib["libdesc"] = pd.concat([ld.assign(root=50000 + xid), same, other],
                               ignore_index=True).sort_values("xid", kind="stable").reset_index(drop=True)
    g = lib["xgeo"]
    n = len(g["nxt"])
    even = np.arange(0, n - 1, 2)
    g["nxt"] = np.full(n, -1, np.int64)
    g["nxt"][even] = even + 1
    g["mu"] = np.arange(n) // 2
    g["bar"] = np.where(np.arange(n) % 2 == 0, _ex(geom, 3, 4), _ex(geom, 2, 4))
    return T.attach_anchors(lib, geom)


def _events(geom):
    a = _crossings(geom, [_ex(geom, 3, 9), _ex(geom, 2, 9)], event=1)
    b = _crossings(geom, [_ex(geom, 3, 9)], event=2)                     # the muon ends: forward rows free
    c = _crossings(geom, [_ex(geom, 2, 9), _ex(geom, 3, 9), _ex(geom, 2, 10)], event=3)
    c[["py0", "p1"]] = [-1200.0, 1200.0]                                 # another momentum bin
    fx = pd.concat([a, b, c], ignore_index=True)
    fx["x1"] += 15.0
    return fx


def test_anchor_rows_match_reference(geom):
    jax, j = _jax()
    lib, _ = _tiny(geom)
    lib = T.attach_anchors(lib, geom)
    A = lib["anchor"]
    dev = {"dk": j.asarray(A["dk"]), "axid": j.asarray(A["axid"]), "sym": j.asarray(A["sym"])}
    ld = {"xid": j.asarray(lib["libdesc"].xid.to_numpy())}
    sym = A["sym"]
    for pick, nxt, fsym in ((np.array([0, 1, 3]), np.array([1, 2, -1]), sym[:3]),
                            (np.array([0]), np.array([-1]), sym[:1]),
                            (np.array([0, 3, 3, 3]), np.array([1, 2, 3, -1]), np.array([sym[0], -1, sym[1], sym[2]]))):
        cur, rows, rx, free, _ = T._anchor_rows(pick, nxt, fsym, lib)
        main = np.flatnonzero(A["dk"][rows] >= 0)[:len(np.r_[[r for p in pick for r in A["order"][A["sorted"] == p]]])]
        rows_m = rows[main]
        ix = np.repeat(np.arange(len(pick)), [(A["sorted"] == p).sum() for p in pick])
        am = {"nxt": j.asarray(nxt), "sym": j.asarray(fsym)}
        dcur, drx, dax, dfree, _ = jax.jit(_anchor_rows, static_argnums=5)(dev, am, ld, j.asarray(rows_m), j.asarray(ix),
                                                                         int(A["dk"].max()))
        np.testing.assert_array_equal(np.asarray(dcur), cur[main])
        np.testing.assert_array_equal(np.asarray(drx), rx[main])
        np.testing.assert_array_equal(np.asarray(dfree), free[main])
        np.testing.assert_array_equal(np.asarray(dax), A["axid"][rows_m])


@pytest.mark.parametrize("batch", [3, 64])
def test_anchored_sample_replays_reference(geom, batch):
    lib = _anchor_lib(geom)
    fx = _events(geom)
    gen, parts = _gen([1, 2, 3]), _parts([1, 2, 3])
    rr = RecordRandom(11)
    host = T.sample(fx, parts, gen, lib, geom, rr, **OPTS)
    dev = DeviceTier1b(lib, geom, batch_size=batch)
    got = dev.sample(fx, parts, gen, ReplayRandom(rr.draws), **OPTS)
    compare_frames(host[0], got[0])
    compare_frames(host[1], got[1])
    hi, di = host[2], got[2]
    nz = lambda d: {k: v for k, v in d.items() if v}   # the reference leaves out zero counts of placements not reached
    assert nz(hi["other_bar"]) == nz(di["other_bar"]), (hi["other_bar"], di["other_bar"])
    assert hi["desc_steps"] == di["desc_steps"]
    ob = hi["other_bar"]
    assert ob["anchor_forward"] > 0 and ob["anchor_unfed"] > 0 and ob["anchor_kept_entered"] > 0
    assert ob["anchor_forward_free"] > 0


def test_anchored_empty_and_refusals(geom):
    lib = _anchor_lib(geom)
    fx = _events(geom).iloc[:0]
    gen, parts = _gen([1]), _parts([])
    rr = RecordRandom(3)
    host = T.sample(fx, parts, gen, lib, geom, rr, **OPTS)
    got = DeviceTier1b(lib, geom).sample(fx, parts, gen, ReplayRandom(rr.draws), **OPTS)
    compare_frames(host[0], got[0])
    with pytest.raises(ValueError, match="separate"):
        DeviceTier1b(lib, geom).sample(fx, parts, gen, ReplayRandom(rr.draws), **dict(OPTS, hard=True))
    plain, _ = _lib3(geom)
    plain["libxing"] = plain["libxing"].assign(ctx=0)
    with pytest.raises(ValueError, match="attach_anchors"):
        DeviceTier1b(plain, geom).sample(fx, parts, gen, ReplayRandom(rr.draws), **OPTS)
