"""Opt-in position-conditioned tier-1b templates: closest approach, cells, unvisited-layer and near-miss sampling,
and inertness of every other option."""
import numpy as np
import pandas as pd
import pytest

from crvfastsim import tier1b as T

from test_tier1b_topology import _crossings, _ex, _sampling_lib


@pytest.fixture(scope="module")
def geom():
    from crvfastsim import geometry
    try:
        return geometry.from_offline_config(geometry.find_counters("Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no counters file: {e}")


def test_closest_approach_matches_brute_force(geom):
    rng = np.random.default_rng(1)
    n = 300
    pos = np.c_[rng.uniform(-9000, 2000, n), np.full(n, 6000.0), rng.uniform(19000, 27000, n)]
    u = np.c_[rng.normal(0, 0.4, n), -np.ones(n), rng.normal(0, 0.4, n)]
    g, d, _ = T.closest_approach(pos, u, geom)
    lo, hi = T.group_boxes(geom)
    ts = np.linspace(0, 8000, 8001)
    q = pos[:, None, :] + ts[None, :, None] * T._unit(u)[:, None, :]
    best = np.stack([np.linalg.norm(np.maximum(np.maximum(lo[i] - q, 0), q - hi[i]), axis=2).min(1)
                     for i in range(len(T.GROUPS))], axis=1)
    assert np.allclose(d, best.min(1), atol=1.0)
    miss = best.min(1) > 1.0
    assert np.array_equal(g[miss], best.argmin(1)[miss])
    # a vertical ray through the M sectors enters M first (top of the stack), one beside EX misses it by 96 mm
    g2, d2, _ = T.closest_approach(np.array([[-3900.0, 6000, 23000], [-7000.0, 6000, 23000]]),
                                   np.array([[0.0, -1, 0], [0.0, -1, 0]]), geom)
    assert list(g2) == [3, 0] and d2[0] == 0 and np.isclose(d2[1], 96.0, atol=0.5)


def test_nearmiss_cells_distinct():
    ke = np.array([100.0, 500, 500, 500, 2e4])
    c = T.nearmiss_cell(ke, np.array([0, 0, 0, 1, 3]), np.array([0.0, 0, 50, 50, 5000]))
    assert len(np.unique(c)) == 5 and c[0] == 0 and c[2] == 11 and c[3] == 110   # KE classes in EX only
    assert T.nearmiss_cell(np.array([100.0]), np.array([1]), np.array([50.0]))[0] == c[3]


def test_uv_cells_nested(geom):
    bars = np.array([_ex(geom, 0, 3), _ex(geom, 0, 4), _ex(geom, 0, 12), _ex(geom, 1, 3)])
    entry = geom.center[bars] + np.array([[10.0, 0, 0], [500.0, 0, 0], [10.0, 0, 0], [10.0, 0, 0]])
    c = T.uv_cells(bars, entry, np.full(4, 2000.0), np.zeros(4, bool), np.ones(4, np.int64), geom)
    assert c[0, 0] != c[1, 0]                                     # 490 mm apart along the bar: another fine cell
    assert c[0, 0] != c[2, 0]                                     # 9 bars apart across: another fine cell
    assert c[0, 2] == c[1, 2] == c[2, 2] and c[0, 2] != c[3, 2]   # same layer plane; another layer
    assert len(np.unique(c[:, 4])) == 1                           # one library group
    stop = T.uv_cells(bars[:1], entry[:1], np.array([100.0]), np.array([True]), np.ones(1, np.int64), geom)
    assert (stop != c[:1]).all()
    also_t1 = T.uv_cells(bars[:1], entry[:1], np.array([2000.0]), np.array([False]), np.array([3]), geom)
    assert (also_t1[0, :3] != c[0, :3]).all() and (also_t1[0, 3:] == c[0, 3:]).all()   # mask only at levels 0-2


def test_visit_mask(geom):
    x = pd.concat([_crossings(geom, [_ex(geom, 3, 4), geom.bar_index(1, 0, 3, 5)], event=1),
                   _crossings(geom, [_ex(geom, 3, 4)], event=2)], ignore_index=True)
    assert list(T.visit_mask(x, geom)) == [3, 3, 1]


def _template_side(geom, lib, ncensus=1):
    """Template sidecar for _sampling_lib: every library crossing (EX layer 3, a single last crossing) has its one
    descendant row in a T1 bar below, never visited; one near-miss template in the cell of a vertical 1 GeV muon
    beside EX, with ncensus no-crossing census muons in that cell."""
    lx = lib["libxing"]
    nx = len(lx)
    bar = np.full(nx, _ex(geom, 3, 4))
    entry = geom.center[bar] + np.array([100.0, geom.halflen[bar[0], 1], 0.0])
    xing = pd.DataFrame({"xid": lx.xid.to_numpy(), "bar": bar, "ex": entry[:, 0], "ey": entry[:, 1],
                         "ez": entry[:, 2], "p": 3000.0, "last": 1, "mu": lx.xid.to_numpy()})
    t1 = geom.bar_index(1, 0, 3, 5)
    uv = pd.DataFrame({"row": np.arange(len(lib["libdesc"])), "bar": t1})
    nmt = pd.DataFrame({"tid": [0], "group": [0], "dist": [96.0], "ke": [1000.0 - 1e-9], "nsteps": [2], "edep": [1.5]})
    tb = _ex(geom, 0, 0)
    c = geom.center[tb]
    nms = pd.DataFrame({"tid": 0, "sp": [5, 6], "pdg": 11, "bar": tb, "dt0": [2000.0, 2001.0], "dt1": [2000.5, 2001.5],
                        "x0": c[0], "y0": c[1], "z0": c[2], "x1": c[0], "y1": c[1] - 1, "z1": c[2], "px0": 0.0,
                        "py0": -1.0, "pz0": 0.0, "p1": 0.1, "len": 1.0, "edep": [1.0, 0.5]})
    nmc = pd.DataFrame({"group": [0] * ncensus, "dist": [96.0] * ncensus, "ke": [1000.0 - 1e-9] * ncensus})
    return {"xing": xing, "uv": uv, "nmt": nmt, "nms": nms, "nmc": nmc,
            "meta": {"generated": 4, "census_events": 4}}


def _gen(events, x=-7000.0):
    """One vertical 1 GeV-kinetic muon per event, generated above the CRV at x (beside EX by default)."""
    e = 1000.0 - 1e-9 + T.MASS[13]
    return pd.DataFrame({"run": 1400, "subrun": 1, "event": events, "idx": 0, "pdg": 13, "genid": 0, "x": x,
                         "y": 6000.0, "z": 23000.0, "px": 0.0, "py": -np.sqrt(e ** 2 - T.MASS[13] ** 2), "pz": 0.0,
                         "e": e, "t": 50.0})


def _parts(events):
    return pd.DataFrame({"run": 1400, "subrun": 1, "event": events, "key": 1, "parent": -1, "pdg": 13, "isprim": 1})


def _lib(geom, templates=True, ncensus=1):
    lib = _sampling_lib(geom, True)
    lib["libshower"] = pd.DataFrame({"shid": [0], "apdg": [22], "kebin": [3], "ke": [10.0]})
    cols = ["shid", "sp", "isroot", "pdg", "bar", "dt0", "dt1", "x0", "y0", "z0", "x1", "y1", "z1", "px0", "py0",
            "pz0", "p1", "len", "edep"]
    lib["libshowersteps"] = pd.DataFrame({c: np.zeros(0, np.int64 if c in ("shid", "sp", "isroot", "pdg", "bar")
                                                      else float) for c in cols})
    lib["libgen"] = pd.DataFrame({"apdg": [22], "kebin": [3], "ngen": [10], "nshower": [0]})
    if templates:
        lib = T.attach_templates(lib, _template_side(geom, lib, ncensus), geom)
    return lib


def test_templates_inert_when_off(geom):
    fx = _crossings(geom, [_ex(geom, 3, 9), _ex(geom, 2, 9)], event=1)
    gen = pd.concat([_gen([1], x=geom.center[_ex(geom, 3, 9), 0]), _gen([2])], ignore_index=True)
    parts = pd.concat([_parts([1]), _parts([2])], ignore_index=True)
    for opts in ({}, {"context": False, "point": True}, {"topology": True}):
        r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
        a = T.sample(fx, parts, gen, _lib(geom, templates=False), geom, r1, **opts)
        b = T.sample(fx, parts, gen, _lib(geom), geom, r2, **opts)
        pd.testing.assert_frame_equal(a[0], b[0])
        pd.testing.assert_frame_equal(a[1], b[1])
        assert r1.random() == r2.random()


def test_unvisited_rows_replaced_by_position_draw(geom):
    lib = _lib(geom)
    fx = _crossings(geom, [_ex(geom, 3, 4)], event=1)
    gen, parts = _gen([1], x=fx.x0.iloc[0]), _parts([1])
    base = T.sample(fx, parts, gen, lib, geom, np.random.default_rng(2))[0]
    new, parts2, info = T.sample(fx, parts, gen, lib, geom, np.random.default_rng(2), unvisited=True)
    t1 = geom.bar_index(1, 0, 3, 5)
    assert info["templates"]["unvisited"]["steps"] == 1
    # the main draw loses its (unvisited) row; the position draw adds it back at its library position relative to the
    # entry (entry difference 0 here): the T1 bar containing it (the test row sits 30 mm across from t1's centre)
    want = T.bar_containing(geom.center[[t1]] + np.array([[30.0, 0.5, 2.0]]), geom)[0]
    assert geom.sector[want] == 1 and T.layer_slots(geom)[0][want] == T.layer_slots(geom)[0][t1]
    assert (new.bar == want).sum() == 1 and len(new) == len(base)
    assert not (base.bar == want).any()
    row = new[new.bar == want].iloc[0]
    lo, hi = geom.center[want] - geom.halflen[want], geom.center[want] + geom.halflen[want]
    assert np.all([lo[j] <= row[f"{c}0"] <= hi[j] for j, c in enumerate("xyz")])
    assert parts2.set_index("key").loc[int(row.sp), "parent"] == 1


def test_unvisited_translation_falls_back_to_library_position(geom):
    lib = _lib(geom)
    shifted = _crossings(geom, [_ex(geom, 3, 4)], event=1)
    shifted[["x0", "x1"]] += 200.0                                  # same fine cell, 200 mm along the bar
    out, info = T._unvisited_for_crossings(shifted, lib, geom, np.random.default_rng(0))
    t1 = geom.bar_index(1, 0, 3, 5)
    assert info["translated"] == 1 and out.bar.iloc[0] != t1      # T1 bars run along z: 200 mm in x is another bar
    far = _crossings(geom, [_ex(geom, 3, 4)], event=1)
    far[["z0", "z1"]] += 5000.0                                     # translated point outside every T1 bar
    out2, info2 = T._unvisited_for_crossings(far, lib, geom, np.random.default_rng(0))
    assert info2["translated"] == 0 and out2.bar.iloc[0] == t1


def test_nearmiss_only_for_events_without_crossing(geom):
    lib = _lib(geom)
    fx = _crossings(geom, [_ex(geom, 3, 9)], event=1)
    gen = pd.concat([_gen([1], x=geom.center[_ex(geom, 3, 9), 0]), _gen([2]), _gen([3], x=-12000.0)],
                    ignore_index=True)
    parts = pd.concat([_parts([1]), _parts([2]), _parts([3])], ignore_index=True)
    steps, p, info = T.sample(fx, parts, gen, lib, geom, np.random.default_rng(4), nearmiss=True)
    nm = info["templates"]["nearmiss"]
    assert nm["candidates"] == 2 and nm["reached"] == 1             # event 3 is far from every group: no cell
    ev2 = steps[steps.event == 2]
    assert len(ev2) == 2 and np.allclose(ev2.t0, 50.0 + np.array([2000.0, 2001.0]))
    par = p.set_index(["event", "key"]).parent
    assert all(par.loc[(2, int(k))] == 1 for k in ev2.sp)
    assert not (steps.event == 3).any()


def test_nearmiss_reach_probability(geom):
    lib = _lib(geom, ncensus=4)
    r = lib["nmreach"].set_index("cell")
    cell = T.nearmiss_cell(np.array([1000.0 - 1e-9]), np.array([0]), np.array([96.0]))[0]
    assert r.loc[cell, "prob"] == pytest.approx(0.25) and r.loc[cell, "nden"] == 4
    gen = _gen(np.arange(10, 4010))
    fx = _crossings(geom, [_ex(geom, 3, 9)], event=1).iloc[:0]
    out, info = T._nearmiss_for_gen(fx, gen, lib, geom, np.random.default_rng(9))
    assert abs(info["reached"] - 1000) < 100 and info["expected_reached"] == pytest.approx(1000)


def test_attach_templates_checks_library(geom):
    lib = _lib(geom, templates=False)
    side = _template_side(geom, lib)
    T.attach_templates(lib, side, geom)
    bad = dict(side, xing=side["xing"].iloc[1:])
    with pytest.raises(ValueError):
        T.attach_templates(lib, bad, geom)
    with pytest.raises(ValueError):
        T._desc_for_crossings(_crossings(geom, [_ex(geom, 3, 9)]), lib, geom, np.random.default_rng(0),
                              drop_unvisited=True)

