"""Unit tests: the cosmic pre-filter (mu2efastsim.cosmic_prefilter) and its scan (cosmic_prefilter_scan), synthetic
geometry and events only. Geometry is checked against brute-force sampling in float64 and float32."""
import gzip
import json
import sqlite3

import numpy as np
import pytest

from mu2efastsim import cosmic_prefilter as cp
from mu2efastsim import cosmic_prefilter_scan as scan
from mu2efastsim import materials

DTYPES = (np.float64, np.float32)


def _lines(rng, n, scale=3000.0):
    o = np.column_stack([rng.uniform(-scale, scale, n), np.full(n, 5000.0), rng.uniform(-scale, scale, n)])
    d = np.column_stack([rng.normal(0, 0.5, n), -np.ones(n), rng.normal(0, 0.5, n)])
    return o, d / np.linalg.norm(d, axis=1)[:, None]


def _brute(fun, o, d, t_max=20000.0, n=200001):
    t = np.linspace(0, t_max, n)
    return np.array([fun(o[i] + t[:, None] * d[i]).min() for i in range(len(o))])


def _box_sd(lo, hi):
    c, h = 0.5 * (np.asarray(lo) + hi), 0.5 * (np.asarray(hi) - lo)
    return lambda x: (np.abs(x - c) - h).max(1)


@pytest.mark.parametrize("dtype", DTYPES)
def test_box_margin_matches_brute_force(dtype):
    rng = np.random.default_rng(1)
    lo, hi = np.array([-1000.0, -200.0, -500.0]), np.array([1500.0, 300.0, 2500.0])
    o, d = _lines(rng, 60)
    m = cp.box_required_margin(o.astype(dtype), d.astype(dtype), lo, hi, t_max=20000.0)
    assert m.dtype == dtype
    ref = _brute(_box_sd(lo, hi), o, d)
    tol = 0.2 if dtype == np.float64 else 1.0          # brute-force grid step 0.1 mm (times |d| <= 1)
    assert np.allclose(m, ref, atol=tol)


def test_box_margin_simple_cases():
    lo, hi = np.array([-1.0, -2.0, -3.0]), np.array([1.0, 2.0, 3.0])
    o = np.array([[0.0, 10, 0], [4.0, 10, 0], [0.0, -10, 0], [0.0, 10, 0]])
    d = np.array([[0.0, -1, 0], [0.0, -1, 0], [0.0, -1, 0], [0.0, 1, 0]])
    m = cp.box_required_margin(o, d, lo, hi)
    # through the centre: the box can shrink by its smallest half-width; 3 mm off a face: 3; a forward ray moving
    # away from the box is closest at its start, 8 mm beyond the y face, in both directions
    assert np.allclose(m, [-1.0, 3.0, 8.0, 8.0])


def test_box_entry_threshold():
    rng = np.random.default_rng(2)
    lo, hi = np.array([-500.0, -500.0, -500.0]), np.array([500.0, 500.0, 500.0])
    o, d = _lines(rng, 200, 2000.0)
    m = cp.box_required_margin(o, d, lo, hi)
    assert np.all(np.isfinite(cp.box_entry(o, d, lo, hi, m + 1e-6)))
    assert np.all(np.isinf(cp.box_entry(o, d, lo, hi, m - 1e-3)))
    t = cp.box_entry(o, d, lo, hi, np.maximum(m, 0) + 10.0)
    p = o + t[:, None] * d
    grown = np.maximum(m, 0) + 10.0
    assert np.all(np.abs(p).max(1) <= 500.0 + grown + 1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("axis", (0, 2))
def test_cylinder_margin_and_entry_match_brute_force(dtype, axis):
    rng = np.random.default_rng(3 + axis)
    env = cp.Envelope.cylinder("c", "trk", (100.0, -300.0, 200.0), axis, 850.0, 1600.0)
    o, d = _lines(rng, 40)
    c = np.asarray(env.center)
    tr = [k for k in range(3) if k != axis]

    def sd(x):
        w = x - c
        return np.maximum(np.sqrt((w[:, tr] ** 2).sum(1)) - env.radius, np.abs(w[:, axis]) - env.half)

    m = cp.required_margin(o.astype(dtype), d.astype(dtype), env, t_max=20000.0)
    ref = _brute(sd, o, d)
    assert np.allclose(m, ref, atol=0.2 if dtype == np.float64 else 1.0)
    t = cp.entry(o.astype(dtype), d.astype(dtype), env, 50.0, t_max=20000.0)
    tt = np.linspace(0, 20000.0, 200001)
    for i in range(len(o)):
        inside = sd(o[i] + tt[:, None] * d[i]) <= 50.0
        if inside.any():
            assert abs(t[i] - tt[np.argmax(inside)]) < (0.2 if dtype == np.float64 else 2.0)
        else:
            assert np.isinf(t[i])


GDML = """<?xml version="1.0"?>
<gdml>
 <materials>
  <material name="Dense0x9" state="solid"><D unit="g/cm3" value="7.0"/><fraction n="1" ref="Fe0x8"/></material>
 </materials>
 <solids>
  <box name="WorldBox" x="100000" y="100000" z="100000" lunit="mm"/>
  <tube name="TrkTube" rmin="380" rmax="850" z="3200" deltaphi="360" aunit="deg" lunit="mm"/>
  <box name="Bar" x="10" y="20" z="4000" lunit="mm"/>
  <box name="SecBox" x="20" y="40" z="4000" lunit="mm"/>
  <union name="SecUnion"><first ref="SecBox"/><second ref="SecBox"/></union>
 </solids>
 <structure>
  <volume name="Bar0x1"><materialref ref="G4_POLYSTYRENE0x2"/><solidref ref="Bar"/></volume>
  <volume name="CRSmother_CRV_A0x3"><materialref ref="G4_AIR0x4"/><solidref ref="SecUnion"/>
   <physvol><volumeref ref="Bar0x1"/><position name="p1" x="-500" y="0" z="0" unit="mm"/></physvol>
   <physvol><volumeref ref="Bar0x1"/><position name="p2" x="700" y="30" z="100" unit="mm"/></physvol>
  </volume>
  <volume name="TrackerMother0x5"><materialref ref="G4_AIR0x4"/><solidref ref="TrkTube"/></volume>
  <volume name="World"><materialref ref="G4_AIR0x4"/><solidref ref="WorldBox"/>
   <physvol><volumeref ref="TrackerMother0x5"/><position name="pt" x="-3904" y="0" z="24175" unit="mm"/></physvol>
   <physvol><volumeref ref="CRSmother_CRV_A0x3"/><position name="pc" x="0" y="4000" z="23000" unit="mm"/></physvol>
  </volume>
 </structure>
 <setup name="Default" version="1.0"><world ref="World"/></setup>
</gdml>
"""


def test_envelopes_from_gdml(tmp_path):
    g = tmp_path / "w.gdml"
    g.write_text(GDML)
    envs = cp.envelopes_from_gdml(g, {"trk": (r"TrackerMother",), "crv": (r"CRSmother_CRV_\w+",)})
    by = {e.term: e for e in envs}
    t = by["trk"]
    assert t.axis == 2 and np.allclose(t.center, (-3904, 0, 24175)) and t.radius == 850 and t.half == 1600
    c = by["crv"]
    assert c.axis == -1
    # union of the mother's first box (20 x 40 x 4000 at the origin) and two bars, placed at (0, 4000, 23000)
    assert np.allclose(c.lo, (-505, 3980, 21000)) and np.allclose(c.hi, (705, 4040, 25100))
    assert cp.gdml_densities(g) == {"Dense": 7.0}
    with pytest.raises(ValueError, match="no envelope matched"):
        cp.envelopes_from_gdml(g, {"calo": (r"CaloDisk_\d+",)})


def test_range_table():
    rt = cp.RangeTable()
    tk = np.array([10.0, 100.0, 1000.0, 10000.0])
    assert np.allclose(rt.ke_after(tk, 0.0), tk, rtol=1e-6)
    r1 = rt.range_of(1000.0)
    assert 1500.0 < r1 < 2600.0                          # ~1 GeV muon: about 2 m of concrete (2.35 g/cm3)
    assert np.all(np.diff(rt.range_of(np.geomspace(1, 1e6, 50))) > 0)
    left = rt.ke_after(np.array([1000.0]), np.array([500.0]))[0]
    assert np.isclose(rt.range_of(left), r1 - 500.0, rtol=1e-4)
    assert rt.ke_after(np.array([100.0]), np.array([1e5]))[0] == -1.0
    assert rt.scale("CONCRETE_MARS") == 1.0 and 0 < rt.scale("G4_AIR") < 1e-2
    assert rt.scale("NotAMaterial") == 1.0 and "NotAMaterial" in rt.unknown
    rt = cp.RangeTable(densities={"Heavy": 4.7})
    assert np.isclose(rt.scale("Heavy"), 2.0) and "Heavy" in rt.unknown


def _slab_world():
    """A concrete slab 1 m thick (y in [2000, 3000]) over a 2 m cube envelope at the origin."""
    cube = np.array([[1, 0, 0, 5000], [-1, 0, 0, 5000], [0, 1, 0, 3000], [0, -1, 0, -2000], [0, 0, 1, 5000],
                     [0, 0, -1, 5000]], float)
    return [("CONCRETE_MARS", cube, np.array([-5000.0, 2000, -5000]), np.array([5000.0, 3000, 5000]))]


def _prims(rows):
    cols = ("event", "pdg", "x", "y", "z", "px", "py", "pz", "e")
    a = np.array(rows, float)
    out = {c: a[:, i] for i, c in enumerate(cols)}
    out["event"] = out["event"].astype(np.int64)
    out["pdg"] = out["pdg"].astype(np.int32)
    return out


def _mu(event, x, tk, pdg=13):
    m = cp.MASS[abs(pdg)]
    e = tk + m
    p = np.sqrt(e * e - m * m)
    return (event, pdg, x, 4000.0, 0.0, 0.0, -p, 0.0, e)


@pytest.mark.parametrize("dtype", DTYPES)
def test_decisions_end_to_end(dtype):
    env = cp.Envelope.box("cube", "crv", (-1000, -1000, -1000), (1000, 1000, 1000))
    pf = cp.Prefilter([env], _slab_world(), dtype=dtype)
    stop_tk = float(np.interp(1000.0 + 1000.0 * pf.range.scale("G4_AIR"), pf.range.r, pf.range.tk))
    prims = _prims([_mu(1, 0.0, 5000.0),            # through the slab, plenty of energy: kept
                    _mu(2, 0.0, 0.5 * stop_tk),      # ranges out in the slab: range-check reject
                    _mu(3, 1300.0, 5000.0),          # misses the cube by 300 mm
                    _mu(4, 5000.0, 50.0, pdg=22),    # a gamma far away
                    _mu(4, 6000.0, 5000.0, pdg=-13)])  # ... with a muon missing by 5 m in the same event
    f = pf.features(prims, [100.0, 500.0])
    assert f.dtype == np.dtype(dtype).name
    assert np.allclose(f.margin_star[:3, 2], [-1000, -1000, 300], atol=1e-3)
    assert np.isinf(f.margin_star[:, 0]).all()        # no trk envelope
    p = cp.Params(margin_mm=100.0, min_muon_ke_mev=0.0, nonmuon="always")
    assert cp.decide_primaries(f, p).tolist() == [True, False, False, True, False]
    p = cp.Params(margin_mm=500.0, min_muon_ke_mev=None, nonmuon="threshold", nonmuon_min_ke_mev=100.0)
    assert cp.decide_primaries(f, p).tolist() == [True, True, True, False, False]
    p = cp.Params(margin_mm=500.0, min_muon_ke_mev=0.0, nonmuon="threshold", nonmuon_min_ke_mev=0.0,
                  nonmuon_margin_mm=100.0)
    assert cp.decide_primaries(f, p).tolist() == [True, False, True, False, False]
    keep, ev, kev = pf.decide(prims, cp.Params(margin_mm=100.0))
    assert ev.tolist() == [1, 2, 3, 4] and kev.tolist() == [True, False, False, True]
    # energy at the cube grown by 100 mm: the slab and 1.9 m of air
    left = f.tk_env[0, 2, 0]
    expect = pf.range.ke_after(np.array([5000.0]), np.array([1000.0 + 1900.0 * pf.range.scale("G4_AIR")]))[0]
    assert np.isclose(left, expect, rtol=1e-4)
    with pytest.raises(ValueError, match="not in the features"):
        cp.decide_primaries(f, cp.Params(margin_mm=250.0))


def test_float32_and_float64_agree_on_random_primaries():
    rng = np.random.default_rng(7)
    envs = [cp.Envelope.box("b", "crv", (-2000, 3000, -1500), (2000, 3500, 4000)),
            cp.Envelope.cylinder("t", "trk", (0, 0, 1000), 2, 850, 1600)]
    n = 4000
    o, d = _lines(rng, n, 8000.0)
    tk = rng.lognormal(7, 1.2, n)
    e = tk + cp.MASS[13]
    p = np.sqrt(e * e - cp.MASS[13] ** 2)
    prims = {"event": np.arange(n), "pdg": np.full(n, 13, np.int32), "x": o[:, 0], "y": o[:, 1] + 10000,
             "z": o[:, 2], "px": p * d[:, 0], "py": p * d[:, 1], "pz": p * d[:, 2], "e": e}
    out = {}
    for dt in DTYPES:
        f = cp.Prefilter(envs, _slab_world(), dtype=dt).features(prims, [0.0, 500.0])
        out[dt] = (f, [cp.decide_primaries(f, cp.Params(margin_mm=m, min_muon_ke_mev=10.0)) for m in (0.0, 500.0)])
    f64, k64 = out[np.float64]
    f32, k32 = out[np.float32]
    fin = np.isfinite(f64.margin_star)
    assert np.allclose(f32.margin_star[fin], f64.margin_star[fin], atol=0.5)
    for a, b in zip(k64, k32):
        assert (a != b).mean() < 2e-3                    # only lines within float32 rounding of a boundary flip
        assert a.sum() > 50


def test_params_validation_and_record():
    with pytest.raises(ValueError):
        cp.Params(nonmuon="never")
    with pytest.raises(ValueError):
        cp.Params(terms=("dqm",))
    with pytest.raises(ValueError):
        cp.Params(margin_mm=2e4)
    with pytest.raises(ValueError):
        cp.Params(min_muon_ke_mev=-1.0)
    rec = cp.Params(terms=("crv",)).record()
    assert rec["terms"] == ["crv"] and rec["nonmuon"] == "always"
    with pytest.raises(ValueError, match="unknown CRY particle"):
        cp.kinetic_energy(np.array([999]), np.array([1.0]))


PRINT = """
 ###############  PrintModule Run/Subrun/Event      1400     9001        1

ProductPrint generate_x
GenParticleCollection has 1 particles
 ind   pdg        x        y        z        px       py       pz      t     tau  gen
   0    13    -3904.0  15365.4  24171.0       0.0  -1000.0      0.0  50000.0     0.0    CosmicCRY

 ###############  PrintModule Run/Subrun/Event      1400     9001        2

ProductPrint generate_x
GenParticleCollection has 2 particles
   0    22   -121938.9  15365.4 146447.7   -730.8    -584.0    287.9      1.6     0.0    CosmicCRY
   1   -13    100.0  15365.4  24171.0       10.0  -2000.0      0.0  50000.0     0.0    CosmicCRY
"""


def test_read_printgen(tmp_path):
    g = tmp_path / "gen.txt.gz"
    with gzip.open(g, "wt") as fh:
        fh.write(PRINT)
    prim, events = scan.read_printgen(g)
    assert events.tolist() == [scan.event_key(9001, 1), scan.event_key(9001, 2)]
    assert prim["pdg"].tolist() == [13, 22, -13] and prim["idx"].tolist() == [0, 0, 1]
    assert np.isclose(prim["x"][1], -121938.9) and np.isclose(prim["pz"][1], 287.9)
    assert np.isclose(prim["e"][0], np.sqrt(1000.0 ** 2 + cp.MASS[13] ** 2))


def _write_tables(tmp_path):
    uproot = pytest.importorskip("uproot")
    i = lambda *v: np.array(v, np.int32)        # noqa: E731
    fl = lambda *v: np.array(v, np.float32)     # noqa: E731
    # event 1: muon primary (key 1) with 2 good CRV steps -> crv; event 2: gamma primary (key 1) whose electron
    # (key 5) has 12 good straw steps -> trk; event 3: accepted with no steps -> calo; event 4: a muon and a gamma,
    # only the gamma's descendant (key 7, parent 2 = gen 1) makes the CRV term; a soft step (p 40) does not count
    gen = {"subrun": i(5, 5, 5, 5, 5), "event": i(1, 2, 3, 4, 4), "idx": i(0, 0, 0, 0, 1), "pdg": i(13, 22, -13, 13, 22)}
    parts = {"subrun": i(5, 5, 5, 5, 5, 5, 5), "event": i(1, 2, 2, 3, 4, 4, 4), "key": i(1, 1, 5, 1, 1, 2, 7),
             "parent": i(-1, -1, 1, -1, -1, -1, 2), "genidx": i(0, 0, -1, 0, 0, 1, -1)}
    steps = {"subrun": i(5, 5, 5, 5), "event": i(1, 1, 4, 4), "sp": i(1, 1, 7, 1), "edep": fl(0.5, 0.3, 0.2, 0.5),
             "px0": fl(0, 0, 0, 0), "py0": fl(-900, -900, -60, -40), "pz0": fl(0, 0, 0, 0)}
    sgs = {"subrun": np.full(12, 5, np.int32), "event": np.full(12, 2, np.int32), "sp": np.full(12, 5, np.int32),
           "eion": np.full(12, 1e-3, np.float32), "px": np.zeros(12, np.float32), "py": np.full(12, -70, np.float32),
           "pz": np.zeros(12, np.float32)}
    c, t = tmp_path / "crvsteps.root", tmp_path / "sgsteps.root"
    with uproot.recreate(c) as f:
        f["dump/gen"] = gen
        f["dump/particles"] = parts
        f["dump/steps"] = steps
    with uproot.recreate(t) as f:
        f["dump/gen"] = gen
        f["dump/particles"] = parts
        f["dump/sgsteps"] = sgs
    return c, t


def test_census(tmp_path):
    c, t = _write_tables(tmp_path)
    cen = scan.census(c, t)
    k = scan.event_key(5, np.array([1, 2, 3, 4]))
    assert cen["event"].tolist() == k.tolist()
    assert cen["crv"].tolist() == [True, False, False, True]
    assert cen["trk"].tolist() == [False, True, False, False]
    assert cen["calo"].tolist() == [False, False, True, False]
    assert cen["crv_mask"].tolist() == [1, 0, 0, 2] and cen["trk_mask"].tolist() == [0, 1, 0, 0]
    assert cen["mu_mask"].tolist() == [1, 0, 1, 1]


def test_timing(tmp_path):
    db = tmp_path / "timing.db"
    with sqlite3.connect(db) as con:
        con.execute("create table TimeModule (Run, SubRun, Event, Path, ModuleLabel, ModuleType, Time)")
        rows = [(1400, 9001, e, "p", lab, "x", 0.01 * e) for e in range(1, 31) for lab in ("g4run", "generate")]
        con.executemany("insert into TimeModule values (?,?,?,?,?,?,?)", rows)
    gen = tmp_path / "gen.txt"
    gen.write_text("".join(f" ###  PrintModule Run/Subrun/Event 1400 9001 {e}\n" for e in range(1, 31)))
    dts = tmp_path / "dts.txt"
    dts.write_text("".join(f" ###  PrintModule Run/Subrun/Event 1400 9001 {e}\n" for e in (3, 25, 30)))
    t = scan.timing(db, gen, dts, skip_first=20)
    assert t["event"].tolist() == scan.event_key(9001, np.arange(21, 31)).tolist()
    assert np.allclose(t["g4_s"], 0.01 * np.arange(21, 31))
    assert t["accepted"].tolist() == [e in (25, 30) for e in range(21, 31)]


def test_curve_components(tmp_path):
    """Loss components on a hand-built sample: features of four accepted events and one rejected one."""
    margins = np.array([0.0, 500.0])
    # event, idx, pdg, tk, margin_star (trk, calo, crv), tk_env at the crv term per margin
    rows = [(1, 0, 13, 5000.0, (np.inf, np.inf, -100.0), (4000.0, 4000.0)),   # accepted crv, clean muon
            (2, 0, 13, 5000.0, (np.inf, np.inf, 300.0), (np.nan, 3900.0)),    # accepted crv, near miss
            (3, 0, 13, 200.0, (np.inf, np.inf, -50.0), (-1.0, -1.0)),         # accepted crv, ranges out
            (4, 0, 22, 80.0, (np.inf, np.inf, 9000.0), (np.nan, np.nan)),     # accepted crv via a gamma
            (5, 0, 13, 5000.0, (np.inf, np.inf, 9000.0), (np.nan, np.nan))]   # rejected by G4 and the filter
    n = len(rows)
    tk_env = np.full((n, 3, 2), np.nan)
    for k, r in enumerate(rows):
        tk_env[k, 2] = r[5]
    f = cp.Features(np.array([r[0] for r in rows], np.int64), np.array([r[2] for r in rows], np.int32),
                    np.array([r[3] for r in rows]), np.array([r[4] for r in rows]), margins, tk_env, "float64")
    fp = tmp_path / "f.npz"
    np.savez_compressed(fp, event=f.event, pdg=f.pdg, tk=f.tk, margin_star=f.margin_star, margins=f.margins,
                        tk_env=f.tk_env, dtype=np.array(f.dtype), unknown_materials=np.array([], dtype=str),
                        idx=np.array([r[1] for r in rows], np.int32))
    cen = {"event": np.array([1, 2, 3, 4], np.int64), "crv": np.ones(4, bool), "trk": np.zeros(4, bool),
           "calo": np.zeros(4, bool), "crv_mask": np.ones(4, np.int64), "trk_mask": np.zeros(4, np.int64),
           "mu_mask": np.array([1, 1, 1, 0], np.int64)}
    cfile = tmp_path / "c.npz"
    np.savez_compressed(cfile, **cen)
    tim = {"event": np.array([1, 2, 3, 4, 5], np.int64), "g4_s": np.array([0.03, 0.03, 0.03, 0.03, 0.88]),
           "accepted": np.array([True, True, True, True, False])}
    tfile = tmp_path / "t.npz"
    np.savez_compressed(tfile, **tim)
    grid = [cp.Params(margin_mm=0.0, min_muon_ke_mev=0.0, nonmuon="threshold", nonmuon_min_ke_mev=100.0),
            cp.Params(margin_mm=500.0, min_muon_ke_mev=None, nonmuon="always")]
    rows_out, _ = scan.curve([fp], [cfile], [fp], [tfile], grid)
    a, b = rows_out
    assert (a["lost"], a["lost_muon_near_miss"], a["lost_muon_range"], a["lost_nonmuon"]) == (3, 1, 1, 1)
    assert np.isclose(a["loss"], 0.75) and a["loss_crv"] == 0.75 and a["simulated"] == 1
    assert np.isclose(a["g4_saved_fraction"], 0.97) and np.isclose(a["timing_loss"], 0.75)
    assert b["lost"] == 0 and np.isclose(b["g4_saved_fraction"], 0.88)
    assert np.isclose(b["g4_s_per_accepted_kept"], 0.03) and np.isclose(b["g4_s_per_accepted_stock"], 0.25)
    assert [r["params"]["margin_mm"] for r in scan.pareto(rows_out)] == [500.0, 0.0]


def test_param_grid_is_valid():
    grid = scan.param_grid()
    assert len(grid) > 500 and len({json.dumps(p.record(), sort_keys=True) for p in grid}) == len(grid)
    assert all(isinstance(p, cp.Params) for p in grid)
    assert materials.CONCRETE_MARS is cp.REF_MATERIAL


def test_column_eq_subsegments_match_one_segment():
    """column_eq splits long paths into SUBSEGMENT_MM pieces; the total must equal the unsplit column."""
    rng = np.random.default_rng(11)
    world = [("CONCRETE_MARS", q[1], q[2], q[3]) for q in _slab_world()]
    world.append(("G4_Al", np.array([[1, 0, 0, 700], [-1, 0, 0, 300], [0, 1, 0, 1500], [0, -1, 0, -1400],
                                     [0, 0, 1, 900], [0, 0, -1, 100]], float),
                  np.array([300.0, 1400, -100]), np.array([700.0, 1500, 900])))
    pf = cp.Prefilter([cp.Envelope.box("b", "crv", (-1, -1, -1), (1, 1, 1))], world)
    o, d = _lines(rng, 300, 6000.0)
    o[:, 1] += 3000.0
    t = rng.uniform(100.0, 12000.0, len(o))
    eq = pf.column_eq(o, d, t)
    ref = np.zeros(len(o))
    solid = np.zeros(len(o))
    for name, length in pf.world.column(o, d, t).items():
        ref += pf.range.scale(name) * length
        solid += length
    ref += pf.range.scale("G4_AIR") * np.clip(t - solid, 0, None)
    assert np.allclose(eq, ref, rtol=1e-9, atol=1e-6) and (solid > 0).sum() > 50
