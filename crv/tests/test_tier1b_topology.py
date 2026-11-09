"""Opt-in layer-topology placement of other-bar muon descendants: visits, library topology, placement, inertness."""
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


def _crossings(geom, bars, event=1, sp=1, uz=0.0):
    """One straight downward muon crossing the given EX bars in order (entry at each bar's top face)."""
    c = geom.center[bars]
    n = len(bars)
    return pd.DataFrame({"run": 1400, "subrun": 1, "event": event, "nstep": n, "bar": bars, "sp": sp,
                         "t0": 100.0 + np.arange(n), "t1": 100.5 + np.arange(n), "x0": c[:, 0] + 100.0,
                         "y0": c[:, 1] + geom.halflen[bars, 1], "z0": c[:, 2], "x1": c[:, 0] + 100.0,
                         "y1": c[:, 1] - geom.halflen[bars, 1], "z1": c[:, 2], "px0": 0.0, "py0": -3000.0,
                         "pz0": uz, "p1": 3000.0, "len": 20.0, "edep": 3.0})


def _ex(geom, layer, counter, module=1):
    return geom.bar_index(0, module, layer, counter)


def test_layer_slots_unique_and_lookup(geom):
    lid, widx = T.layer_slots(geom)
    code = lid * 10_000 + widx
    assert len(np.unique(code)) == geom.nbars
    assert np.array_equal(T.slot_bar(geom, lid, widx), np.arange(geom.nbars))
    assert T.slot_bar(geom, lid[:1], np.array([10_000]))[0] == -1
    # neighbouring modules of a sector layer continue the width ranks
    assert widx[_ex(geom, 0, 0, module=1)] == widx[_ex(geom, 0, 15, module=0)] + 1


def test_muon_visits_group_same_layer(geom):
    bars = [_ex(geom, 3, 4), _ex(geom, 3, 5), _ex(geom, 2, 5), _ex(geom, 1, 5)]
    mu, vis, v = T.muon_visits(_crossings(geom, bars), geom)
    assert list(vis) == [0, 0, 1, 2] and set(mu) == {0}
    lid, widx = T.layer_slots(geom)
    assert list(v.wmin) == [widx[bars[0]], widx[bars[2]], widx[bars[3]]]
    assert v.wmax.iloc[0] == widx[bars[1]]


def _library(geom, lib_bars, desc_bar, anchor=0, uz=0.0):
    x = _crossings(geom, lib_bars, uz=uz)
    mu, vis, visits = T.muon_visits(x, geom)
    lx = pd.DataFrame({"xid": np.arange(len(x)), "bin": 0, "ux": 0.0, "uy": -1.0, "uz": 0.0, "bar": lib_bars,
                       "mu": mu, "vis": vis})
    u = T._unit(x[["px0", "py0", "pz0"]].to_numpy())
    lx[["ux", "uy", "uz"]] = u
    xb = lib_bars[anchor]
    dc = geom.center[desc_bar] - geom.center[xb]
    loc = np.array([30.0, 1.0, 2.0])     # in-bar offset (world x, y, z) from the descendant bar centre
    ld = pd.DataFrame({"xid": [anchor], "sp": 7, "samebar": 0, "pdg": 11, "dt0": 0.1, "dt1": 0.2,
                       "dx0": 0.0, "dy0": 0.0, "dz0": 0.0, "dx1": 0.0, "dy1": 0.0, "dz1": 0.0,
                       "px0": 0.0, "py0": -1.0, "pz0": 0.0, "p1": 0.0, "len": 1.0, "edep": 0.5,
                       "dcx": dc[0], "dcy": dc[1], "dcz": dc[2], "entrylen": 0.0,
                       "lx0": loc[0], "ly0": loc[1], "lz0": loc[2], "lx1": loc[0], "ly1": loc[1] - 1, "lz1": loc[2]})
    return ld, lx, visits.sort_values(["mu", "vis"]).reset_index(drop=True)


def test_desc_topology_side_offset_and_visit(geom):
    lib_bars = [_ex(geom, 3, 4), _ex(geom, 2, 4), _ex(geom, 1, 4)]
    ld, lx, visits = _library(geom, lib_bars, _ex(geom, 2, 6))       # two bars beyond the layer-2 crossing
    t = T.desc_topology(ld, lx, visits, geom)
    assert t.tvis.iloc[0] == 1 and t.tside.iloc[0] == 2               # lean along +width is zero: sign +1
    assert np.isclose(t.tl0.iloc[0], 30.0 - 100.0)                   # along x from the layer-2 entry (x + 100)
    ld2, lx2, v2 = _library(geom, lib_bars, _ex(geom, 0, 4))         # a layer the muon never visits
    assert T.desc_topology(ld2, lx2, v2, geom).tvis.iloc[0] == T.TOPO_NONE


def test_topology_target_around_receiving_muon(geom):
    lib_bars = [_ex(geom, 3, 4), _ex(geom, 2, 4), _ex(geom, 1, 4)]
    ld, lx, visits = _library(geom, lib_bars, _ex(geom, 2, 6))
    ld = pd.concat([ld, T.desc_topology(ld, lx, visits, geom)], axis=1)
    fast = [_ex(geom, 3, 9), _ex(geom, 2, 9), _ex(geom, 2, 10), _ex(geom, 1, 10)]
    fx = _crossings(geom, fast)
    fu = T._unit(fx[["px0", "py0", "pz0"]].to_numpy())[[0]]
    hit, tb, q0, _ = T._topology_targets(fx, ld, np.array([0]), fu, geom)
    assert hit[0] and tb[0] == _ex(geom, 2, 12)                      # two beyond the two bars crossed in layer 2
    assert np.isclose(q0[0, 0], fx.x0.iloc[1] - 70.0)                # along-bar offset from that visit's entry
    lo, hi = geom.center[tb[0]] - geom.halflen[tb[0]], geom.center[tb[0]] + geom.halflen[tb[0]]
    assert np.all(q0[0] >= lo) and np.all(q0[0] <= hi)
    fxl = fx.assign(pz0=-500.0)                                       # receiving muon leans the other way
    ful = T._unit(fxl[["px0", "py0", "pz0"]].to_numpy())[[0]]
    _, tbl, _, _ = T._topology_targets(fxl, ld, np.array([0]), ful, geom)
    assert tbl[0] == _ex(geom, 2, 7)
    short = fx.iloc[:1]                                                # the receiving muon has no second visit
    hit_s, _, _, _ = T._topology_targets(short, ld, np.array([0]), fu, geom)
    assert not hit_s[0]


def _sampling_lib(geom, tb):
    nb = len(T.GROUPS) * T.N_PBINS * T.N_LBINS
    lib_bars = [_ex(geom, 3, 4), _ex(geom, 2, 4)]
    ld, lx, visits = _library(geom, lib_bars, _ex(geom, 2, 5))
    ld = pd.concat([ld, T.desc_topology(ld, lx, visits, geom)], axis=1)
    libbins = pd.DataFrame([(0, g, e) for g in range(len(T.GROUPS)) for e in (1e3, 2e3, 3e3)] +
                           [(1, g, e) for g in range(len(T.GROUPS)) for e in (5.0, 12.0, 20.0, 22.0, 25.0, 30.0)],
                           columns=["kind", "group", "edge"])
    xing = pd.DataFrame({"xid": np.arange(nb), "bin": np.arange(nb), "ux": 0.0, "uy": -1.0, "uz": 0.0})
    desc = pd.concat([ld.assign(xid=i) for i in range(nb)], ignore_index=True)
    if not tb:
        desc = desc.drop(columns=list(T.TOPO_COLS))
    return {"libbins": libbins, "libxing": xing, "libdesc": desc}


def test_default_and_other_options_ignore_topology_columns(geom):
    fx = _crossings(geom, [_ex(geom, 3, 9), _ex(geom, 2, 9)])
    for opts in ({}, {"point": True}, {"contain": True, "point": True}):
        r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
        a = T._desc_for_crossings(fx, _sampling_lib(geom, True), geom, r1, **opts)[0]
        b = T._desc_for_crossings(fx, _sampling_lib(geom, False), geom, r2, **opts)[0]
        pd.testing.assert_frame_equal(a, b)
        assert r1.random() == r2.random()


def test_topology_placement_avoids_receiving_muon_bars(geom):
    fx = _crossings(geom, [_ex(geom, 3, 9), _ex(geom, 2, 9), _ex(geom, 2, 10)])
    out, ndrop, _, why = T._desc_for_crossings(fx, _sampling_lib(geom, True), geom, np.random.default_rng(1),
                                               topology=True)
    assert ndrop == 0 and why["topology"] == 1                        # the layer-2 anchors have no next visit
    first = out[out.xi == 0]
    assert len(first) == 1 and first.bar.iloc[0] == _ex(geom, 2, 11) and not np.isin(first.bar, fx.bar).any()
    with pytest.raises(ValueError):
        T._desc_for_crossings(fx, _sampling_lib(geom, False), geom, np.random.default_rng(1), topology=True)
