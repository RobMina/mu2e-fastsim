"""crvfastsim.design: parametric module geometry, tilted crossings, scaled optics, P(miss), deadtime and cost model.
Synthetic counters files and lookup tables only (no Mu2e-internal inputs); the nominal-layout check on a stock
counters file runs where that file can be found."""
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crvfastsim import geometry as G                                    # noqa: E402
from crvfastsim.design import deadtime as DT                             # noqa: E402
from crvfastsim.design import fom as F                                   # noqa: E402
from crvfastsim.design import module as M                                # noqa: E402
from crvfastsim.design import optics as O                                # noqa: E402
from crvfastsim.design import veto as V                                  # noqa: E402
from crvfastsim.reco import coinc as C                                   # noqa: E402
from crvfastsim.response import photons as PH                            # noqa: E402

COUNTERS = """
vector<string> crs.sectorNames = {"A", "B"};
int crs.nLayers = 4;
double crs.scintillatorBarThickness = 19.80;
double crs.scintillatorBarWidth = 51.34;
double crs.layerOffset = 42;
double crs.gapLarge = 0.2;
double crs.gapSmall = 0.0;
double crs.gapBetweenModules = 3.66;
vector<double> crs.gapBetweenLayers = {9.525, 9.525, 9.525};
string crs.absorberMaterialName = "G4_Al";
vector<double> crs.layerDirectionA = {0, 1, 0};
vector<double> crs.gapDirectionA = {0, 0, 1};
vector<double> crs.offsetDirectionA = {0, 0, -1};
double crs.scintillatorBarLengthA = 3000;
int crs.nModulesA = 3;
int crs.nCountersPerModuleA = 16;
vector<double> crs.firstCounterA = {0, 3000, 0};
vector<double> crs.layerDirectionB = {-1, 0, 0};
vector<double> crs.gapDirectionB = {0, 0, 1};
vector<double> crs.offsetDirectionB = {0, 0, -1};
double crs.scintillatorBarLengthB = 2000;
int crs.nModulesB = 2;
int crs.nCountersPerModuleB = 16;
vector<double> crs.firstCounterB = {-2000, 1500, 0};
"""


@pytest.fixture
def counters(tmp_path):
    p = tmp_path / "crv_counters_test.txt"
    p.write_text(COUNTERS)
    return str(p)


def _table(name="3000_0", reflector=0, length=3000.0):
    ex = np.linspace(0, 9.64, 5)
    ey = np.linspace(-25.42, 25.42, 9)
    ez = np.linspace(-length / 2, length / 2, 31)
    zc = 0.5 * (ez[1:] + ez[:-1]) + length / 2
    prob = np.ones((4, 8, 30)) * 0.0005 * np.exp(-zc / 4000.0)[None, None, :]
    return O.TableModel(name, prob, ex, ey, ez, (9.64, 25.42, length / 2), 4000.0, reflector)


def _ctx(counters):
    sec = {"A": PH.SectorConfig("3000_0", 0, 30482.0), "B": PH.SectorConfig("3000_1", 1, 39794.0)}
    cuts = {n: C.SectorCuts(8, 10, 20, -6, 6, 3, 3, 0, 250, 0) for n in sec}
    ctx = F.Context(counters=counters, sectors=sec, cuts=cuts)
    ctx.tables = {"3000_0": _table(), "3000_1": _table("3000_1", 1)}
    return ctx


def _muons(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    o = np.stack([rng.uniform(-2500, 500, n), np.full(n, 6000.0), rng.uniform(-200, 5000, n)], 1)
    d = np.stack([rng.normal(0, 0.3, n), -np.ones(n), rng.normal(0, 0.3, n)], 1)
    d /= np.linalg.norm(d, axis=1)[:, None]
    return dict(origin=o, direction=d, p=rng.uniform(500, 5000, n))


def _same_as_stock(path):
    geo = G.from_offline_config(path)
    dg = M.build(M.Design(), path)
    got = M.to_crv_geometry(dg)
    assert got.nbars == geo.nbars                                   # same creation order: sector, module, layer
    assert np.array_equal(got.sector, geo.sector) and np.array_equal(got.layer, geo.layer)
    assert np.array_equal(got.module, geo.module)
    np.testing.assert_allclose(got.center, geo.center, atol=1e-6)
    np.testing.assert_allclose(got.halflen, geo.halflen, atol=1e-9)
    assert np.array_equal(got.local_to_world, geo.local_to_world)


def test_nominal_design_reproduces_stock_bars(counters):
    _same_as_stock(counters)


@pytest.mark.parametrize("name", ["Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt", M.RUN2_COUNTERS])
def test_nominal_design_reproduces_stock_offline_file(name, monkeypatch):
    monkeypatch.setenv("CRVFS_NO_FETCH", "1")
    try:
        path = G.find_counters(name)
    except FileNotFoundError:
        pytest.skip(f"{name} not on MU2E_SEARCH_PATH")
    _same_as_stock(path)


def test_layers_absorber_and_segments(counters):
    dg = M.build(M.Design(n_layers=5, absorber=20.0, max_length=1100.0, absorber_material="MildSteel"), counters)
    a = dg.sector == 0
    assert int(dg.layer[a].max()) == 4
    y = np.unique(np.round(dg.center[a & (dg.segment == 0) & (dg.counter == 0), 1], 6))
    np.testing.assert_allclose(np.diff(y), 19.80 + 20.0)
    assert dg.segment[a].max() == 2 and np.allclose(dg.halflen[a, 2], 1000.0 / 2)
    nmod = dg.module[a].max() + 1
    assert (dg.abs_sector == 0).sum() == nmod * 4 and np.allclose(2 * dg.abs_halflen[dg.abs_sector == 0, 0], 20.0)
    assert dg.absorber_material == "MildSteel"


@pytest.mark.parametrize("tilt", [5.0, 20.0, 45.0])
def test_tilt_rotates_modules_without_overlap(counters, tilt):
    dg = M.build(M.Design(tilt_deg=tilt), counters)
    dg0 = M.build(M.Design(), counters)
    for s in range(2):
        A = dg.axes[s]
        np.testing.assert_allclose(A @ A.T, np.eye(3), atol=1e-12)
        assert math.isclose(abs(A[0] @ dg0.axes[s][0]), math.cos(math.radians(tilt)), rel_tol=1e-12)
        b = dg.sector == s
        loc = dg.center[b] @ A.T
        lo, hi = loc - dg.halflen[b], loc + dg.halflen[b]
        mods = dg.module[b]
        for m in range(mods.max()):
            assert not M._overlap(lo[mods == m], hi[mods == m], lo[mods == m + 1], hi[mods == m + 1])
        louver = 822.84 * math.cos(math.radians(tilt)) + 3.66
        assert dg.module_pitch[s] >= louver - 1e-9
        if tilt >= 20.0:
            assert math.isclose(dg.module_pitch[s], louver, rel_tol=1e-12)


def test_crossing_path_lengths(counters):
    o = np.array([[-500.0, 6000.0, 1000.0]])
    d = np.array([[0.0, -1.0, 0.0]])
    dg = M.build(M.Design(), counters)
    idx, pos = V.sector_crossings(dg, o, d)
    a = dg.sector[idx[:, 1]] == 0
    path = np.linalg.norm(pos[a, 3:] - pos[a, :3], axis=1)
    assert sorted(dg.layer[idx[a, 1]].tolist()) == [0, 1, 2, 3] and np.allclose(path, 19.80, rtol=1e-12)
    dg = M.build(M.Design(tilt_deg=30.0), counters)
    rng = np.random.default_rng(2)
    o = np.stack([rng.uniform(-1400, 1400, 300), np.full(300, 6000.0), rng.uniform(300, 2000, 300)], 1)
    idx, pos = V.sector_crossings(dg, o, np.tile(d, (300, 1)))
    assert np.all(np.abs(pos[:, :3]) <= dg.halflen[idx[:, 1]] + 1e-6)
    assert np.all(np.abs(pos[:, 3:]) <= dg.halflen[idx[:, 1]] + 1e-6)
    assert len(np.unique(idx[dg.sector[idx[:, 1]] == 0, 0])) == 300          # louvers: every vertical line meets one


def test_tilt_enters_only_through_the_module_frame(counters):
    """Tilt is module-local: a tilted module seen by rigidly rotated muons gives the untilted module's local crossings
    and PE, because crossings and response use the counter-local entrance state only (no global-layout binning)."""
    d0, d1 = M.build(M.Design(), counters), M.build(M.Design(tilt_deg=20.0), counters)
    s = 0
    m0, m1 = (d0.sector == s) & (d0.module == 0), (d1.sector == s) & (d1.module == 0)
    assert np.array_equal(np.flatnonzero(m0), np.flatnonzero(m1))
    piv = d0.center[m0].mean(0)
    np.testing.assert_allclose(d1.center[m1].mean(0), piv, atol=1e-9)
    R = d0.axes[s].T @ d1.axes[s]                        # row vectors: v -> v @ R (untilted -> tilted module)
    mu = _muons(3000, seed=5)
    i0, p0 = V.sector_crossings(d0, mu["origin"], mu["direction"])
    i1, p1 = V.sector_crossings(d1, piv + (mu["origin"] - piv) @ R, mu["direction"] @ R)
    k0, k1 = m0[i0[:, 1]], m1[i1[:, 1]]
    i0, p0, i1, p1 = i0[k0], p0[k0], i1[k1], p1[k1]
    o0, o1 = np.lexsort((i0[:, 1], i0[:, 0])), np.lexsort((i1[:, 1], i1[:, 0]))
    assert len(o0) > 100 and np.array_equal(i0[o0], i1[o1])
    np.testing.assert_allclose(p1[o1], p0[o0], atol=1e-6)
    tab = _table()
    pe = [O.expected_pe(p[o, :3], p[o, 3:], d.halflen[i[o, 1]], mu["p"][i[o, 0]], tab, 30482.0, 1.4, 1.4, (0, 1))
          for d, i, p, o in ((d0, i0, p0, o0), (d1, i1, p1, o1))]
    np.testing.assert_allclose(pe[1], pe[0], rtol=1e-9, atol=1e-9)


def test_coincidence_rule_and_threshold_knobs(counters):
    mu = _muons()
    ctx = _ctx(counters)
    cal = {"source": "synthetic", "total": 0.05}
    base = F.evaluate(M.Design(), mu, ctx, calibration=cal)
    assert set(base["coincidence"].values()) == {"3 of 4, 8 PE"}
    same = F.evaluate(M.Design(coincidence_layers=3), mu, _ctx(counters), calibration=cal)
    assert same["inefficiency"] == base["inefficiency"] and same["deadtime"] == base["deadtime"]
    strict = F.evaluate(M.Design(coincidence_layers=4), mu, _ctx(counters), calibration=cal)
    assert strict["inefficiency"] > base["inefficiency"] and strict["deadtime"] < base["deadtime"]
    loose = F.evaluate(M.Design(n_layers=5, coincidence_layers=2), mu, _ctx(counters), calibration=cal)
    assert loose["inefficiency"] <= base["inefficiency"] and loose["deadtime"] > base["deadtime"]
    assert set(loose["coincidence"].values()) == {"2 of 5, 8 PE"}
    high = F.evaluate(M.Design(pe_threshold=30.0), mu, _ctx(counters), calibration=cal)
    assert high["inefficiency"] > base["inefficiency"] and high["deadtime"] < base["deadtime"]
    spare = F.evaluate(M.Design(n_layers=5, spare_layers=3), mu, _ctx(counters), calibration=cal)
    assert spare["inefficiency"] == loose["inefficiency"] and spare["deadtime"] == loose["deadtime"]
    with pytest.raises(ValueError):
        M.Design(n_layers=3, spare_layers=3)
    with pytest.raises(ValueError):
        M.Design(coincidence_layers=3, spare_layers=1)
    with pytest.raises(ValueError):
        M.Design(n_layers=3, coincidence_layers=4)
    with pytest.raises(ValueError):
        F.evaluate(M.Design(coincidence_layers=5), mu, _ctx(counters))       # stock 4 layers
    with pytest.raises(ValueError):
        M.Design(pe_threshold=0.0)


def test_absorber_mass_in_tonnes(counters):
    from mu2efastsim import materials
    dg = M.build(M.Design(), counters)
    vol = float(np.prod(2 * dg.abs_halflen, axis=1).sum()) * 1e-9          # m^3
    ct = M.cost_terms(dg, {"A": 0, "B": 1}, materials.get("G4_Al").rho)
    assert ct["absorber_t"] == pytest.approx(vol * materials.get("G4_Al").rho) and 0.1 < ct["absorber_t"] < 10


def test_p_coincidence_matches_enumeration():
    rng = np.random.default_rng(3)
    mu = rng.uniform(0, 15, (50, 5, 2))
    thr = rng.choice([4.0, 8.0, 12.0], 50)
    k = rng.integers(1, 6, 50)
    got = V.p_coincidence(mu, thr, k)
    from scipy.stats import poisson
    for g in range(50):
        p = 1 - poisson.cdf(thr[g] - 1, mu[g]).prod(1)
        tot = sum(np.prod([p[j] if h else 1 - p[j] for j, h in enumerate(hits)])
                  for hits in itertools.product([0, 1], repeat=5) if sum(hits) >= k[g])
        assert math.isclose(got[g], tot, rel_tol=1e-10, abs_tol=1e-14)


def test_expected_pe_scaling():
    tab = _table()
    h = np.array([[9.9, 25.67, 1500.0]])
    a, b = np.array([[-9.9, 3.0, 0.0]]), np.array([[9.9, 3.0, 0.0]])
    p = np.array([1000.0])
    pe = O.expected_pe(a, b, h, p, tab, 30482.0, 1.4, 1.4, (0, 1))
    pe2 = O.expected_pe(a * 0.5, b * 0.5, h, p, tab, 30482.0, 1.4, 1.4, (0, 1))
    np.testing.assert_allclose(pe2, pe / 2, rtol=1e-12)
    pe_f = O.expected_pe(a, b, h, p, tab, 30482.0, 1.8, 1.4, (0, 1))
    np.testing.assert_allclose(pe_f / pe, 39794.0 / 30482.0, rtol=1e-12)
    one = O.expected_pe(a, b, h, p, tab, 30482.0, 1.4, 1.4, (0,))
    assert one[0, 1] == 0 and one[0, 0] == pe[0, 0]
    far = O.expected_pe(a + [0, 0, 1400.0], b + [0, 0, 1400.0], h, p, tab, 30482.0, 1.4, 1.4, (0, 1))
    assert far[0, 0] < pe[0, 0] < far[0, 1]
    with pytest.raises(ValueError):
        O.capture_factor(0.4, 50.0, 1.4, 1.4, tab, O.OpticsConfig())


def test_evaluate_nominal_and_dtypes(counters):
    mu = _muons()
    r64 = F.evaluate(M.Design(), mu, _ctx(counters), dtype=np.float64)
    r32 = F.evaluate(M.Design(), mu, _ctx(counters), dtype=np.float32)
    assert 0 <= r64["inefficiency"] <= 1 and r64["n_crossings"] > 0
    assert math.isclose(r64["cost"], 1.0) and math.isclose(r64["deadtime_rate_ratio"], 1.0, rel_tol=1e-9)
    assert math.isclose(r64["ly_margin"], 0.94 ** 5 / 0.86, rel_tol=1e-9)
    assert abs(r32["inefficiency"] - r64["inefficiency"]) <= 1e-4 + 1e-3 * r64["inefficiency"]
    assert r64["inefficiency"] >= r64["inefficiency_covered"]
    json.dumps(r64, default=float)


def test_evaluate_design_moves_metrics(counters):
    mu = _muons()
    base = F.evaluate(M.Design(), mu, _ctx(counters))
    thick = F.evaluate(M.Design(thickness=30.0), mu, _ctx(counters))
    assert thick["cost_rel"]["scint_m3"] > 1.4 and thick["ly_margin"] > 1.2 * base["ly_margin"]
    five = F.evaluate(M.Design(n_layers=5), mu, _ctx(counters))
    assert five["cost_rel"]["channels"] == pytest.approx(1.25)
    weak = F.evaluate(M.Design(fiber_diameter=0.5), mu, _ctx(counters))
    assert weak["ly_margin"] < base["ly_margin"] and weak["inefficiency_covered"] > base["inefficiency_covered"]


def test_deadtime_calibration_and_absorber(counters):
    ctx = _ctx(counters)
    cal = {"source": "synthetic", "total": 0.10}
    nom = F.evaluate(M.Design(), _muons(200), ctx, calibration=cal)
    assert math.isclose(nom["deadtime"], 0.10, rel_tol=1e-9)
    heavy = F.evaluate(M.Design(absorber=30.0), _muons(200), ctx, calibration=cal)
    assert heavy["deadtime"] < nom["deadtime"]
    cfg = F.FomConfig(deadtime=DT.DeadtimeConfig(kappa=3.0))
    hot = F.evaluate(M.Design(), _muons(200), ctx, cfg, calibration=cal)
    assert hot["deadtime"] > 0.25 and hot["deadtime_rate_ratio"] > 3.0


def test_cli_geometry_and_evaluate(counters, tmp_path, monkeypatch):
    from crvfastsim.design import __main__ as CLI
    out = tmp_path / "g.json"
    assert CLI.main(["geometry", "--counters", counters, "--design", '{"tilt_deg": 10}', "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["n_bars"] > 0 and rep["design"]["tilt_deg"] == 10
    with pytest.raises(SystemExit):
        CLI.main(["evaluate", "--muons", "x.npz"])            # --dtype is required
    with pytest.raises(KeyError):
        M.Design.from_dict({"layers": 3})


def test_proxy_selection():
    o = np.array([[-3904.0, 5000.0, 8000.0], [5000.0, 5000.0, 8000.0], [-3904.0, 5000.0, 20000.0]])
    d = np.array([[0.0, -1.0, 0.0]] * 3)
    assert F.proxy_hits(o, d).tolist() == [True, False, False]


def test_closure_against_full_chain_where_tables_are(tmp_path):
    import os
    lk = os.environ.get("CRVFS_LOOKUP_DIR", "")
    if not all(os.path.exists(os.path.join(lk, f)) for f in ("LookupTable_6000_0", "photonMap.root")):
        pytest.skip("Mu2e-internal lookup tables not available (CRVFS_LOOKUP_DIR)")
    from crvfastsim.design import closure
    p = tmp_path / "c.txt"
    p.write_text(COUNTERS.replace("BarLengthA = 3000", "BarLengthA = 6000").replace("BarLengthB = 2000",
                                                                                     "BarLengthB = 3200"))
    sec = {"A": PH.SectorConfig("6000_0", 0, 39794.0), "B": PH.SectorConfig("3200_0", 0, 28610.0)}
    cuts = {n: C.SectorCuts(8, 10, 20, -6, 6, 3, 3, 0, 250, 0) for n in sec}
    ctx = F.Context(counters=str(p), sectors=sec, cuts=cuts, lookup_dir=lk)
    out = closure.run(M.Design(), _muons(1500), ctx, photon_map=os.path.join(lk, "photonMap.root"))
    assert out["skipped_sectors"] == [] and 0.5 < out["pe_scale_fit"] < 2.0
    for s in out["sectors"].values():
        assert s["crossings"] > 100 and abs(s["pe_ratio"] - out["pe_scale_fit"]) < 0.2
