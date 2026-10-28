"""Unit tests for tier 1b: rotation, and sampling bookkeeping on a synthetic library (needs the counters file)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crvfastsim import tier1b as T                                     # noqa: E402


def test_rotate_maps_direction_and_keeps_length():
    rng = np.random.default_rng(1)
    a, b = T._unit(rng.normal(size=(100, 3))), T._unit(rng.normal(size=(100, 3)))
    assert np.allclose(T.rotate(a, a, b), b, atol=1e-9)
    v = rng.normal(size=(100, 3))
    assert np.allclose(np.linalg.norm(T.rotate(v, a, b), axis=1), np.linalg.norm(v, axis=1))


def test_fit_inside_keeps_midpoint_off_faces():
    lo, hi = np.zeros((3, 3)), np.array([[10.0, 20.0, 30.0]] * 3)
    r0 = np.array([[-5.0, 1.0, 1.0], [9.0, 19.0, 29.0], [2.0, 2.0, 2.0]])
    r1 = np.array([[-1.0, 3.0, 3.0], [12.0, 25.0, 29.5], [4.0, 4.0, 4.0]])
    a, b = T.fit_inside(r0, r1, lo, hi)
    mid = 0.5 * (a + b)
    assert np.all(mid >= T.INSET_MM - 1e-9) and np.all(mid <= hi - T.INSET_MM + 1e-9)
    assert np.all((a >= lo) & (a <= hi) & (b >= lo) & (b <= hi))
    assert np.allclose(a[2], r0[2]) and np.allclose(b[2], r1[2])   # a step already inside is untouched


def test_outside_same_bar_takes_g4_transverse_position(geom):
    bar = 10
    c, h = geom.center[bar], geom.halflen[bar]
    top = c.copy()
    top[1] += h[1] - 0.1
    fs = pd.DataFrame({"run": [1400], "subrun": [1], "event": [5], "nstep": [1], "bar": [bar], "sp": [1],
                       "t0": [100.0], "t1": [100.1], "x0": [top[0]], "y0": [top[1]], "z0": [top[2]],
                       "x1": [top[0]], "y1": [top[1] - 20], "z1": [top[2]], "px0": [0.0], "py0": [-2500.0],
                       "pz0": [0.0], "p1": [2490.0], "len": [20.0], "edep": [3.0]})
    lib = _lib(geom, bar)
    lib["libdesc"]["dy0"], lib["libdesc"]["dy1"] = 5.0, 4.0          # rotated placement lands above the bar
    out, ndrop, nclamp, _ = T._desc_for_crossings(fs, lib, geom, np.random.default_rng(0))
    assert nclamp == 1 and ndrop == 0
    ax = geom.local_to_world[bar]
    loc0 = out[["x0", "y0", "z0"]].to_numpy()[0] - c
    assert np.isclose(loc0[ax[0]], [2.0, 3.0, 4.0][ax[0]]) and np.isclose(loc0[ax[1]], [2.0, 3.0, 4.0][ax[1]])


def test_nearest_bar_same_axis(geom):
    b = 10
    a = geom.local_to_world[b, 2]
    p = geom.center[b][None, :].copy()
    p[0, a] += 1e5                                            # far along the length: still this bar
    assert T.nearest_bar(p, np.array([a]), geom)[0] == b


@pytest.fixture(scope="module")
def geom():
    from crvfastsim import geometry
    try:
        return geometry.from_offline_config(geometry.find_counters("Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no counters file: {e}")


def _lib(geom, bar):
    """One library crossing per bin, each with one same-bar descendant at +1 mm along the muon; one gamma shower."""
    nb = len(T.GROUPS) * T.N_PBINS * T.N_LBINS
    c = geom.center[bar]
    return {
        "libbins": pd.DataFrame([(0, g, e) for g in range(len(T.GROUPS)) for e in (1e3, 2e3, 3e3)] +
                                [(1, g, e) for g in range(len(T.GROUPS)) for e in (5.0, 12.0, 20.0, 22.0, 25.0, 30.0)],
                                columns=["kind", "group", "edge"]),
        "libxing": pd.DataFrame({"xid": np.arange(nb), "bin": np.arange(nb), "ux": 0.0, "uy": -1.0, "uz": 0.0}),
        "libdesc": pd.DataFrame({"xid": np.arange(nb), "sp": 7, "samebar": 1, "pdg": 11, "dt0": 0.1, "dt1": 0.2,
                                 "dx0": 0.0, "dy0": -1.0, "dz0": 0.0, "dx1": 0.0, "dy1": -2.0, "dz1": 0.0,
                                 "px0": 0.0, "py0": -1.0, "pz0": 0.0, "p1": 0.0, "len": 1.0, "edep": 0.5,
                                 "dcx": 0.0, "dcy": 0.0, "dcz": 0.0, "entrylen": 0.0,
                                 "lx0": 2.0, "ly0": 3.0, "lz0": 4.0, "lx1": 2.0, "ly1": 2.0, "lz1": 4.0}),
        "libshower": pd.DataFrame({"shid": [0], "apdg": [22], "kebin": [T._ke_bin(np.array([100.0]))[0]], "ke": [100.0]}),
        "libshowersteps": pd.DataFrame({"shid": [0, 0], "sp": [1, 5], "isroot": [0, 0], "pdg": [11, -11], "bar": [bar, bar],
                                        "dt0": [5.0, 6.0], "dt1": [5.1, 6.1],
                                        **{k: [c[i % 3], c[i % 3]] for i, k in enumerate(T.STEP_COLS)},
                                        "px0": 0.0, "py0": -1.0, "pz0": 0.0, "p1": 0.0, "len": 2.0, "edep": [3.0, 4.0]}),
        "libgen": pd.DataFrame({"apdg": [22], "kebin": [T._ke_bin(np.array([100.0]))[0]], "ngen": [1], "nshower": [1]}),
    }


def test_sample_bookkeeping(geom):
    bar = 10
    c, h = geom.center[bar], geom.halflen[bar]
    top = c.copy()
    top[1] += h[1] - 0.1                                     # entry just below the bar's top face, going down
    fs = pd.DataFrame({"run": [1400], "subrun": [1], "event": [5], "nstep": [1], "bar": [bar], "sp": [1],
                       "t0": [100.0], "t1": [100.1], "x0": [top[0]], "y0": [top[1]], "z0": [top[2]],
                       "x1": [top[0]], "y1": [top[1] - 20], "z1": [top[2]], "px0": [0.0], "py0": [-2500.0],
                       "pz0": [0.0], "p1": [2490.0], "len": [20.0], "edep": [3.0]})
    fp = pd.DataFrame({"run": [1400], "subrun": [1], "event": [5], "key": [1], "parent": [-1], "pdg": [13], "isprim": [1]})
    gen = pd.DataFrame({"run": [1400, 1400], "subrun": [1, 1], "event": [5, 5], "idx": [0, 1], "pdg": [13, 22],
                        "genid": 0, "x": 0.0, "y": 0.0, "z": 0.0, "px": 0.0, "py": [-2500.0, -100.0], "pz": 0.0,
                        "e": [2502.0, 100.0], "t": [50.0, 70.0]})
    steps, parts, info = T.sample(fs, fp, gen, _lib(geom, bar), geom, np.random.default_rng(0))
    assert info["desc_dropped"] == 0 and info["showers"] == 1
    assert set(steps.sp) <= set(parts.key)                  # every step points at a particle
    assert set(parts.parent[parts.parent >= 0]) <= set(parts.key)
    d = steps[(steps.sp > 1) & (steps.sp < T.SHOWER_KEY_BASE)]
    assert len(d) == 1 and d.bar.iloc[0] == bar and np.isclose(d.t0.iloc[0], 100.1)
    assert np.isclose(d.y0.iloc[0], top[1] - 1.0)          # offset applied along the (same) muon direction
    s = steps[steps.sp >= T.SHOWER_KEY_BASE]
    assert len(s) == 2 and np.allclose(sorted(s.t0), [75.0, 76.0])
    root = parts[parts.key == T.SHOWER_KEY_BASE]
    assert len(root) == 1 and root.pdg.iloc[0] == 22 and root.isprim.iloc[0] == 1


def _xing(geom, bars, t0, sp=1):
    """Crossing rows (one muon unless sp varies) entering each bar at its centre."""
    c = geom.center[np.asarray(bars)]
    return pd.DataFrame({"run": 1400, "subrun": 1, "event": 5, "sp": sp, "bar": bars, "t0": t0,
                         "x0": c[:, 0], "y0": c[:, 1], "z0": c[:, 2]})


def test_layer_rank_top_is_zero(geom):
    r = T.layer_rank(geom)
    ta = geom.local_to_world[:, 0]
    t = geom.center[np.arange(geom.nbars), ta]
    for s, m in {(int(a), int(b)) for a, b in zip(geom.sector, geom.module)}:
        sel = (geom.sector == s) & (geom.module == m)
        assert t[sel & (r == 0)].min() > t[sel & (r > 0)].max()
        assert sorted(set(r[sel])) == list(range(len(set(geom.layer[sel]))))


def test_crossing_context_layer_and_upstream(geom):
    r = T.layer_rank(geom)
    ex = geom.sector_names.index("EX")
    t2 = geom.sector_names.index("T2")
    col = lambda s, m, lay: int(np.flatnonzero((geom.sector == s) & (geom.module == m) & (r == lay))[0])
    a, b = col(ex, 1, 0), col(ex, 1, 1)
    ctx = T.crossing_context(_xing(geom, [a, b], [1.0, 2.0]), geom)
    assert list(ctx) == [0, 2]                               # top no upstream; layer 1, the layer above is the same module
    up = col(t2, 0, 3)                                       # T2 bottom layer, then the EX bar nearest below it
    below = np.flatnonzero((geom.sector == ex) & (r == 0))
    e0 = int(below[np.argmin(np.linalg.norm(geom.center[below] - geom.center[up], axis=1))])
    assert np.linalg.norm(geom.center[e0] - geom.center[up]) < T.UPSTREAM_MM
    ctx = T.crossing_context(_xing(geom, [up, e0], [1.0, 2.0]), geom)
    assert list(ctx) == [6, 1]
    ctx = T.crossing_context(_xing(geom, [up, e0], [1.0, 2.0], sp=[1, 2]), geom)   # another muon: no upstream
    assert list(ctx) == [6, 0]


def _ctx_lib(geom, bar, ctx_rows):
    """_lib with two library crossings per bin: context ctx_rows[0] (descendant dy -1) and ctx_rows[1] (dy -3)."""
    lib = _lib(geom, bar)
    nb = len(lib["libxing"])
    lx = pd.concat([lib["libxing"].assign(ctx=ctx_rows[0]),
                    lib["libxing"].assign(xid=lib["libxing"].xid + nb, ctx=ctx_rows[1])], ignore_index=True)
    ld2 = lib["libdesc"].assign(xid=lib["libdesc"].xid + nb, dy0=-3.0, dy1=-4.0)
    lib["libxing"], lib["libdesc"] = lx, pd.concat([lib["libdesc"], ld2], ignore_index=True)
    return lib


def _down(geom, bar, below_top=0.1):
    """One fast crossing entering bar just below its top face, going straight down."""
    c, h = geom.center[bar], geom.halflen[bar]
    top = c.copy()
    top[1] += h[1] - below_top
    return pd.DataFrame({"run": [1400], "subrun": [1], "event": [5], "nstep": [1], "bar": [bar], "sp": [1],
                         "t0": [100.0], "t1": [100.1], "x0": [top[0]], "y0": [top[1]], "z0": [top[2]],
                         "x1": [top[0]], "y1": [top[1] - 20], "z1": [top[2]], "px0": [0.0], "py0": [-2500.0],
                         "pz0": [0.0], "p1": [2490.0], "len": [20.0], "edep": [3.0]}), top


def test_context_draw_and_fallback(geom, monkeypatch):
    bar = int(np.flatnonzero(T.layer_rank(geom) == 0)[0])
    fs, top = _down(geom, bar)
    fctx = int(T.crossing_context(fs, geom)[0])
    monkeypatch.setattr(T, "MIN_CTX_CROSSINGS", 1)
    for seed in range(8):                                    # the matching context only
        out, *_ = T._desc_for_crossings(fs, _ctx_lib(geom, bar, (fctx + 1, fctx)), geom, np.random.default_rng(seed),
                                        context=True)
        assert np.isclose(out.y0.iloc[0], top[1] - 3.0)
    monkeypatch.setattr(T, "MIN_CTX_CROSSINGS", 2)           # cell too small: the whole bin
    got = {round(float(T._desc_for_crossings(fs, _ctx_lib(geom, bar, (fctx + 1, fctx)), geom, np.random.default_rng(s),
                                               context=True)[0].y0.iloc[0] - top[1]), 3) for s in range(16)}
    assert got == {-1.0, -3.0}


@pytest.mark.parametrize("context", [False, True])
def test_descendant_trace_is_rng_and_payload_inert(geom, context):
    bar = 10
    fs, _ = _down(geom, bar)
    lib = _ctx_lib(geom, bar, (0, 1))
    a_rng, b_rng = np.random.default_rng(17), np.random.default_rng(17)
    a = T._desc_for_crossings(fs, lib, geom, a_rng, context=context, contain=True, point=True)
    trace = {}
    b = T._desc_for_crossings(fs, lib, geom, b_rng, context=context, contain=True, point=True, trace=trace)
    pd.testing.assert_frame_equal(a[0], b[0], check_exact=True)
    assert a[1:] == b[1:]
    np.testing.assert_array_equal(a_rng.random(20), b_rng.random(20))
    np.testing.assert_array_equal(trace["target_bar"][trace["kept"]], b[0].bar)
    np.testing.assert_array_equal(trace["source_edep"][trace["kept"]], b[0].edep)
    assert trace["source_samebar"].all()


def test_default_ignores_ctx_column(geom):
    """Options off: a library with the ctx column samples exactly as the same library without it."""
    bar = int(np.flatnonzero(T.layer_rank(geom) == 0)[0])
    fs, _ = _down(geom, bar)
    lib = _ctx_lib(geom, bar, (0, 1))
    old = {k: v for k, v in lib.items()}
    old["libxing"] = lib["libxing"].drop(columns="ctx")
    for seed in range(4):
        a = T._desc_for_crossings(fs, lib, geom, np.random.default_rng(seed))[0]
        b = T._desc_for_crossings(fs, old, geom, np.random.default_rng(seed))[0]
        pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError):
        T._desc_for_crossings(fs, old, geom, np.random.default_rng(0), context=True)


def test_same_bar_endpoints_stay_in_the_bar(geom):
    bar = 10
    c, h = geom.center[bar], geom.halflen[bar]
    fs, _ = _down(geom, bar)
    lib = _lib(geom, bar)
    lib["libdesc"]["dy0"], lib["libdesc"]["dy1"] = 3.0, -9.0     # midpoint inside, start 3 mm above the top face
    out, ndrop, nclamp, _ = T._desc_for_crossings(fs, lib, geom, np.random.default_rng(0), contain=True)
    assert nclamp == 0 and ndrop == 0
    for e in "01":
        r = out[[f"x{e}", f"y{e}", f"z{e}"]].to_numpy()[0]
        assert np.all(np.abs(r - c) <= h + 1e-9)
    out, *_ = T._desc_for_crossings(fs, lib, geom, np.random.default_rng(0))   # default: unchanged, start outside
    assert out.y0.iloc[0] > c[1] + h[1] + 1.0


def _other_axis_pair(geom):
    """A T1 top-layer bar and the EX bottom-layer bar above it (bars along z and along x)."""
    r = T.layer_rank(geom)
    names = np.asarray(geom.sector_names)[geom.sector]
    t1 = np.flatnonzero((names == "T1") & (r == 0))
    ex = np.flatnonzero((names == "EX") & (r == 3))
    b1 = int(t1[len(t1) // 2])
    c = geom.center[b1]
    e = int(ex[np.argmin(np.abs(geom.center[ex, 2] - c[2]))])        # EX bars span T1 in x
    assert geom.local_to_world[b1, 2] != geom.local_to_world[e, 2]
    return b1, e


def test_point_placement_other_axis(geom):
    b1, e = _other_axis_pair(geom)
    c1, ce = geom.center[b1], geom.center[e]
    fs = pd.DataFrame({"run": [1400], "subrun": [1], "event": [5], "nstep": [1], "bar": [b1], "sp": [1],
                       "t0": [100.0], "t1": [100.1], "x0": [c1[0]], "y0": [c1[1]], "z0": [c1[2]],
                       "x1": [c1[0]], "y1": [c1[1] - 20], "z1": [c1[2]], "px0": [0.0], "py0": [-2500.0],
                       "pz0": [0.0], "p1": [2490.0], "len": [20.0], "edep": [3.0]})
    lib = _lib(geom, b1)
    d = lib["libdesc"]
    loc = ce - geom.center[e]                                        # the step sits at the library bar's centre...
    shift_x = c1[0] - ce[0]                                          # ...below the fast T1 entry along x
    d["samebar"], d["entrylen"] = 0, 0.0
    d["dcx"], d["dcy"], d["dcz"] = ce - c1
    d["dcx"] += shift_x
    for k, v in zip(("lx0", "ly0", "lz0", "lx1", "ly1", "lz1"), (-shift_x, 1.0, 0.0, -shift_x, -1.0, 0.0)):
        d[k] = v + loc[{"x": 0, "y": 1, "z": 2}[k[1]]]
    out, *_, why = T._desc_for_crossings(fs, lib, geom, np.random.default_rng(0), point=True)
    assert why["point"] == 1 and out.bar.iloc[0] == e
    assert np.all(np.abs(out[["x0", "y0", "z0"]].to_numpy()[0] - ce) <= geom.halflen[e] + 1e-9)
    out, *_, why = T._desc_for_crossings(fs, lib, geom, np.random.default_rng(0))   # default: a bar along z
    assert "point" not in why and geom.local_to_world[out.bar.iloc[0], 2] == geom.local_to_world[b1, 2]


def test_mirror_reflects_other_bar_offsets(geom):
    r = T.layer_rank(geom)
    names = np.asarray(geom.sector_names)[geom.sector]
    wa = geom.local_to_world[:, 1]
    lay = np.flatnonzero((names == "EX") & (r == 1) & (geom.module == geom.module[np.flatnonzero(names == "EX")[0]]))
    lay = lay[np.argsort(geom.center[lay, wa[lay[0]]])]
    b, n, m = int(lay[8]), int(lay[9]), int(lay[7])                  # a bar and its two width neighbours
    w = wa[b]
    c = geom.center[b]
    lib = _lib(geom, b)
    lean = np.zeros(3)
    lean[1], lean[w] = -1.0, 0.3
    lib["libxing"][["ux", "uy", "uz"]] = T._unit(lean[None, :])[0]
    d = lib["libdesc"]
    d["samebar"], d["entrylen"] = 0, 0.0
    d["dcx"], d["dcy"], d["dcz"] = geom.center[n] - c
    for k in ("lx0", "ly0", "lz0", "lx1", "ly1", "lz1"):
        d[k] = 0.0
    d["ly1"] = -1.0
    fu = -lean.copy()
    fu[1] = -1.0                                                     # fast muon leans the other way across the bar
    fs = pd.DataFrame({"run": [1400], "subrun": [1], "event": [5], "nstep": [1], "bar": [b], "sp": [1],
                       "t0": [100.0], "t1": [100.1], "x0": [c[0]], "y0": [c[1]], "z0": [c[2]],
                       "x1": [c[0]], "y1": [c[1] - 20], "z1": [c[2]], "px0": [2500 * fu[0]], "py0": [2500 * fu[1]],
                       "pz0": [2500 * fu[2]], "p1": [2490.0], "len": [20.0], "edep": [3.0]})
    out, *_, why = T._desc_for_crossings(fs, lib, geom, np.random.default_rng(0), mirror=True)
    assert why["mirror_w"] == 1 and out.bar.iloc[0] == m
    out, *_ = T._desc_for_crossings(fs, lib, geom, np.random.default_rng(0))
    assert out.bar.iloc[0] == n
    fs[["px0", "pz0"]] *= -1                                         # same lean: no reflection
    out, *_, why = T._desc_for_crossings(fs, lib, geom, np.random.default_rng(0), mirror=True)
    assert why["mirror_w"] == 0 and out.bar.iloc[0] == n


def test_translate_keeps_offset_from_the_entry(geom):
    r = T.layer_rank(geom)
    names = np.asarray(geom.sector_names)[geom.sector]
    wa = geom.local_to_world[:, 1]
    lay = np.flatnonzero((names == "EX") & (r == 1) & (geom.module == geom.module[np.flatnonzero(names == "EX")[0]]))
    lay = lay[np.argsort(geom.center[lay, wa[lay[0]]])]
    b, n, nn = int(lay[8]), int(lay[9]), int(lay[10])
    w, c, h = wa[b], geom.center[b], geom.halflen[b]
    ew = np.zeros(3)
    ew[w] = 1.0
    lib = _lib(geom, b)
    d = lib["libdesc"]
    d["samebar"], d["entrylen"] = 0, 0.0
    dc = geom.center[n] - c
    d["dcx"], d["dcy"], d["dcz"] = dc
    for k in ("lx0", "ly0", "lz0", "lx1", "ly1", "lz1"):
        d[k] = 0.0
    lib_entry = c - 20 * ew + np.array([0, h[1], 0])               # library muon entered 20 mm to one side
    off = geom.center[n] - lib_entry
    d["dx0"], d["dy0"], d["dz0"] = off
    d["dx1"], d["dy1"], d["dz1"] = off
    e = c + 20 * ew + np.array([0, h[1], 0])                        # fast muon entered 20 mm to the other side
    fs = pd.DataFrame({"run": [1400], "subrun": [1], "event": [5], "nstep": [1], "bar": [b], "sp": [1],
                       "t0": [100.0], "t1": [100.1], "x0": [e[0]], "y0": [e[1]], "z0": [e[2]],
                       "x1": [e[0]], "y1": [e[1] - 20], "z1": [e[2]], "px0": [0.0], "py0": [-2500.0],
                       "pz0": [0.0], "p1": [2490.0], "len": [20.0], "edep": [3.0]})
    out, *_, why = T._desc_for_crossings(fs, lib, geom, np.random.default_rng(0), translate=True)
    assert why["translate"] == 1 and out.bar.iloc[0] == nn
    assert np.allclose(out[["x0", "y0", "z0"]].to_numpy()[0], e + off)
    out, *_, why = T._desc_for_crossings(fs, lib, geom, np.random.default_rng(0))   # default: the bar-structure offset
    assert "translate" not in why and out.bar.iloc[0] == n
