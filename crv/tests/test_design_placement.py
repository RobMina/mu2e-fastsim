"""crvfastsim.design layout, envelope and placement on synthetic inputs (no Mu2e-internal files): orientation codes,
convex pieces and sweeps, the layout of the stock design, the solver's shifts, sheet pushes and remedies against
analytic answers, and the placed figures of merit in both dtypes."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crvfastsim import geometry as G                                    # noqa: E402
from crvfastsim.design import envelope as E                              # noqa: E402
from crvfastsim.design import fom as F                                   # noqa: E402
from crvfastsim.design import kit_task as KT                             # noqa: E402
from crvfastsim.design import layout as L                                # noqa: E402
from crvfastsim.design import module as M                                # noqa: E402
from crvfastsim.design import optics as O                                # noqa: E402
from crvfastsim.design import placement as P                             # noqa: E402
from crvfastsim.reco import coinc as C                                   # noqa: E402
from crvfastsim.response import photons as PH                            # noqa: E402

# one horizontal sector A (layers along +y, modules along +z, counters along x): stock stack from y = -22.6
# (strongback) to 101.05 (cover), z 0..418.4
COUNTERS = """
vector<string> crs.sectorNames = {"A"};
int crs.nSectors = 1;
int crs.nLayers = 4;
double crs.scintillatorBarThickness = 19.80;
double crs.scintillatorBarWidth = 51.34;
double crs.layerOffset = 0;
double crs.gapLarge = 0.2;
double crs.gapSmall = 0.0;
double crs.gapBetweenModules = 3.66;
vector<double> crs.gapBetweenLayers = {9.525, 9.525, 9.525};
string crs.absorberMaterialName = "G4_Al";
double crs.aluminumSheetThickness = 3.175;
double crs.strongBackThickness = 12.7;
double crs.CMBOffset = 15.0;
double crs.CMBHalfThickness = 5.0;
vector<double> crs.layerDirectionA = {0, 1, 0};
vector<double> crs.gapDirectionA = {0, 0, 1};
vector<double> crs.offsetDirectionA = {0, 0, 1};
double crs.scintillatorBarLengthA = 1800;
int crs.nModulesA = 2;
int crs.nCountersPerModuleA = 4;
vector<double> crs.firstCounterA = {0, 0, 25.67};
bool crs.sipmsAtSide0A = true;
bool crs.sipmsAtSide1A = true;
"""
# a concrete block above A (y 260..860, z 0..500) and a 1-inch borated-poly sheet between (y 137.3..162.7)
UPSTREAM = """
int ExtShieldUpstream.numberOfBoxTypes = 2;
int ExtShieldUpstream.nBoxType1 = 1;   // Block label  TEST-1
int ExtShieldUpstream.nBoxType2 = 1;   // Borated poly: test sheet
vector<double> ExtShieldUpstream.dimsType1 = {2400, 600, 500};
vector<double> ExtShieldUpstream.dimsType2 = {2400, 500, 25.4};
vector<double> ExtShieldUpstream.tolsType1 = {0, 0, 0};
vector<double> ExtShieldUpstream.tolsType2 = {0, 0, 0};
string ExtShieldUpstream.materialType1 = "CONCRETE_MARS";
string ExtShieldUpstream.materialType2 = "Polyethylene092B050d095";
vector<double> ExtShieldUpstream.centerType1Box1 = {0, 560, 250};
vector<double> ExtShieldUpstream.centerType2Box1 = {0, 150, 250};
string ExtShieldUpstream.orientationType2Box1 = "100";
double ExtShieldUpstream.OpenDistance = 0.0;
double ExtShieldUpstream.FrameGap = 0.0;
"""
# one T-block far away (exercises the extrusion reader)
DOWNSTREAM = """
int ExtShieldDownstream.numberOfBoxTypes = 1;
int ExtShieldDownstream.nBoxType1 = 1;
vector<double> ExtShieldDownstream.outlineType1UVerts = {680.8,680.8,223.6,223.6,-223.6,-223.6,-680.8,-680.8};
vector<double> ExtShieldDownstream.outlineType1VVerts = {-452.2,-5,-5,452.2,452.2,-5,-5,-452.2};
double ExtShieldDownstream.lengthType1 = 3851;
vector<double> ExtShieldDownstream.tolsType1 = {5.0,5.0,9.5};
string ExtShieldDownstream.materialType1 = "CONCRETE_MARS";
vector<double> ExtShieldDownstream.centerType1Box1 = {0, 0, 20000};
string ExtShieldDownstream.orientationType1Box1 = "103";
"""
HALL = "double yOfFloorSurface.below.mu2eOrigin = -100;\n"
# 6 x 30 mm counters with 30 mm absorbers: the stack grows from 123.65 to 345.875 mm
THICK = M.Design(n_layers=6, thickness=30.0, absorber=30.0)


@pytest.fixture
def world(tmp_path):
    paths = {}
    for name, text in (("counters", COUNTERS), ("up", UPSTREAM), ("down", DOWNSTREAM), ("hall", HALL)):
        p = tmp_path / f"{name}.txt"
        p.write_text(text)
        paths[name] = str(p)
    cfg = P.PlacementConfig(upstream=paths["up"], downstream=paths["down"], hall=paths["hall"], stagger_mm=(0.0,))
    return paths, cfg


def test_orientation_codes_are_rotations():
    assert np.array_equal(E.orientation_matrix("000"), np.eye(3))
    R = E.orientation_matrix("100")                 # rotateX(-90) as the frame rotation: local (u, v, w) -> (u, -w, v)
    assert np.allclose(R @ [0, 0, 1], [0, -1, 0]) and np.allclose(R @ [0, 1, 0], [0, 0, 1])
    for code in ("001", "010", "103", "101", "012", "030", "300", "002", "321"):
        R = E.orientation_matrix(code)
        assert np.allclose(R @ R.T, np.eye(3)) and math.isclose(np.linalg.det(R), 1.0)
        assert np.allclose(R, np.rint(R))
    assert not np.allclose(E.orientation_matrix("040"), np.rint(E.orientation_matrix("040")))   # 45 degrees


def test_triangulation_and_downstream_tolerances(world):
    uv = np.array([[680.8, -452.2], [680.8, -5], [223.6, -5], [223.6, 452.2], [-223.6, 452.2], [-223.6, -5],
                   [-680.8, -5], [-680.8, -452.2]])
    tris = E.triangulate(uv)
    area = sum(abs(E._area2(uv[list(t)])) / 2 for t in tris)
    assert len(tris) == 6 and math.isclose(area, 1361.6 * 447.2 + 447.2 * 457.2, rel_tol=1e-9)
    xs, ys = E._tol_signs(uv)                        # every vertex moves outward
    assert np.array_equal(np.sign(uv[:, 0]), xs)
    assert np.array_equal(ys[[0, 7]], [-1, -1]) and np.array_equal(ys[[3, 4]], [1, 1])
    paths, _ = world
    pieces = E.read_downstream(paths["down"])
    lo, hi = np.min([p.verts.min(0) for p in pieces], 0), np.max([p.verts.max(0) for p in pieces], 0)
    ext = np.sort(hi - lo)                           # rotated, grown by the tolerances
    assert np.allclose(ext, np.sort([1361.6 + 5, 904.4 + 5, 3851 + 9.5]))


def test_sweep_and_separation_match_boxes():
    ps = E.PieceSet.of([E.box_piece("b", [0, 0, 0], [10, 10, 10])])
    t0, t1 = E.sweep([-5, 30, -5], [5, 40, 5], ps, [0, -1, 0], 2.0)      # moving down onto the box
    assert math.isclose(t0[0, 0], 30 - 10 - 2) and math.isclose(t1[0, 0], 40 + 10 + 2)
    assert np.isnan(E.sweep([30, 30, 30], [40, 40, 40], ps, [0, -1, 0], 0.0)[0][0, 0])    # never meets it
    assert math.isclose(E.separation([-5, 30, -5], [5, 40, 5], ps)[0, 0], 20.0)
    assert math.isclose(E.separation([-5, 5, -5], [5, 15, 5], ps)[0, 0], -5.0)
    tri = E.prism_pieces("t", np.array([[0, 0], [10, 0], [0, 10]]), 5.0, [0, 0, 0], np.eye(3))[0]
    pt = E.PieceSet.of([tri])
    assert E.separation([6, 6, -1], [8, 8, 1], pt)[0, 0] > 0            # beyond the hypotenuse, inside the AABB
    assert E.separation([1, 1, -1], [2, 2, 1], pt)[0, 0] < 0


def test_layout_of_the_stock_design_is_stock(world):
    paths, _ = world
    text = L.counters_text(M.Design(), L.Placement(), paths["counters"])
    a, b = M.build(M.Design(), paths["counters"]), L.build(M.Design(), text)
    assert a.nbars == b.nbars and np.array_equal(a.center, b.center) and np.array_equal(a.halflen, b.halflen)
    assert "AF" not in b.sector_names


def test_layout_shift_move_filler_and_stagger(world, tmp_path):
    paths, _ = world
    d = M.Design(width=40.0, n_layers=6, thickness=30.0, absorber=30.0, max_length=1000.0)
    with pytest.raises(ValueError, match="layerOffset"):           # the stagger rides on the layer offset
        L.counters_text(d, L.Placement(), paths["counters"], L.LayoutConfig(stagger_mm=20.0, split=True, close_up=False))
    cpath = tmp_path / "offset.txt"
    cpath.write_text(COUNTERS.replace("crs.layerOffset = 0", "crs.layerOffset = 42"))
    t = L.counters_text(d, L.Placement(shifts={"A": 10.0}, moves={"A": [0, 0, 5.0]}), str(cpath),
                        L.LayoutConfig(stagger_mm=20.0, split=True, close_up=False))
    dg = L.build(d, t, None, L.LayoutConfig(stagger_mm=20.0, split=True, close_up=False))
    c = G.parse_simpleconfig(cpath.read_text())
    k, n, cov = L.tiling(c, "A", 40.0, True)
    kf = L.filler_counters(c, L.stock_extent(c, "A") - n * L.module_pitch(c, k, 40.0), 40.0)
    assert dg.sector_names == (["A", "AF"] if kf >= 2 else ["A"]) and cov <= 1.0 + 1e-12
    a = dg.sector == 0
    assert int(dg.module[a].max()) + 1 == n and (int(dg.counter[a].max()) + 1) // n == k
    assert math.isclose(dg.center[a & (dg.layer == 0), 1].min(), -10.0)          # shifted against +y
    assert math.isclose(dg.center[a & (dg.layer == 0), 2].min(), 25.67 + 5.0, abs_tol=1e-9)    # moved 5 mm in z
    off = L.layer_offset(c, d)
    assert math.isclose(off, min(42 * 40 / 51.34, 42 * 3 / 5), rel_tol=1e-3)
    z0 = [dg.center[a & (dg.layer == j), 2].min() for j in range(6)]
    assert np.allclose(np.diff(z0), off)                       # the layer offset along offsetDirection (+z)
    assert int(dg.segment[a].max()) == 1                       # split, with the 42 mm joint gap
    x0 = [dg.center[a & (dg.layer == j) & (dg.segment == 0), 0].min() for j in range(6)]
    assert np.allclose(np.diff(x0), 20.0)                      # the joints step 20 mm per layer


def test_stock_places_without_moving(world):
    paths, cfg = world
    r = P.place(M.Design(), paths["counters"], cfg)
    assert r[0].feasible and r[0].placement.shifts == {} and r[0].sheets["ExtShieldUpstreamBox_2"]["moved_mm"] == 0


def test_shift_rests_on_the_floor_and_sheet_pushed(world, tmp_path):
    paths, cfg = world
    hall = tmp_path / "hall_low.txt"
    hall.write_text("double yOfFloorSurface.below.mu2eOrigin = -500;\n")
    cfg2 = P.PlacementConfig(upstream=cfg.upstream, downstream=cfg.downstream, hall=str(hall), stagger_mm=(0.0,))
    p = P.place(THICK, paths["counters"], cfg2)[0]
    # least sheet displacement, then the smallest shift: A moves down until its cover (318.175) clears the stock
    # sheet (137.3) by 3 mm
    assert p.feasible and math.isclose(p.placement.shifts["A"], 318.175 - 134.3, abs_tol=0.01)
    assert p.sheets["ExtShieldUpstreamBox_2"]["moved_mm"] == 0


def test_remedies_thin_and_move(world):
    paths, cfg = world
    rem = (P.Remedy("thin", sheets=("test sheet",), min_thickness=5.0), P.Remedy("move", sectors=("A",),
                                                                              direction=(0, 0, 1), max_mm=600.0,
                                                                              step=50.0))
    cfg = P.PlacementConfig(upstream=cfg.upstream, downstream=cfg.downstream, hall=cfg.hall, stagger_mm=(0.0,),
                            remedies=rem, policy="each")
    out = P.place(THICK, paths["counters"], cfg)
    base, thin, move = out
    assert not base.feasible and base.violations or base.unsolved
    # thin: A rests on the floor (bottom at -97), its cover at 248.875 pushes the sheet to 251.875; the concrete at
    # 260 (1 mm) leaves 7.125 mm of sheet
    assert thin.feasible and 6.5 <= thin.sheets["ExtShieldUpstreamBox_2"]["thickness_mm"] <= 7.125 + 1e-6
    assert math.isclose(thin.placement.shifts["A"], 69.3, abs_tol=0.01)
    assert math.isclose(thin.sheets["ExtShieldUpstreamBox_2"]["moved_mm"], 251.875 - 137.3, abs_tol=0.01)
    assert thin.shield_modified and thin.summary()["deadtime_shield_uncalibrated"]
    # move: A leaves the block's z range (z >= 500 + 3), then needs no shift
    assert move.feasible and 503.0 - 1e-6 <= move.remedy["amount_mm"] <= 503.5 and move.placement.shifts == {}
    assert not move.shield_modified
    with pytest.raises(ValueError):
        P.Remedy("bad", sectors=("A",), max_mm=0.0)
    with pytest.raises(ValueError):
        P.Remedy("none")
    assert P.parse_remedies("move:top:T1,T2:0,0,1:50;thin:u:Outer CRV-U|type16:10;"
                            "movethin:b:T1:0,0,1:20:type15:5") == (
        P.Remedy("top", sectors=("T1", "T2"), direction=(0.0, 0.0, 1.0), max_mm=50.0),
        P.Remedy("u", sheets=("Outer CRV-U", "type16"), min_thickness=10.0),
        P.Remedy("b", sectors=("T1",), max_mm=20.0, sheets=("type15",), min_thickness=5.0))


def test_check_reports_a_hand_placement(world):
    paths, cfg = world
    bad = P.check(THICK, L.Placement(), paths["counters"], cfg)
    kinds = {v["kind"] for v in bad["violations"]}
    assert "sheet-envelope" in kinds                              # the pushed sheet runs into the concrete
    good = P.check(THICK, L.Placement(shifts={"A": 69.3}), paths["counters"], cfg, thickness={"type2": 7.0})
    assert good["violations"] == []


def _ctx(counters):
    sec = {"A": PH.SectorConfig("3000_0", 0, 30482.0)}
    cuts = {"A": C.SectorCuts(8, 10, 20, -6, 6, 3, 3, 0, 250, 0)}
    ctx = F.Context(counters=counters, sectors=sec, cuts=cuts)
    ex, ey = np.linspace(0, 9.64, 5), np.linspace(-25.42, 25.42, 9)
    ez = np.linspace(-900, 900, 31)
    prob = np.ones((4, 8, 30)) * 0.0005
    ctx.tables = {"3000_0": O.TableModel("3000_0", prob, ex, ey, ez, (9.64, 25.42, 900.0), 4000.0, 0)}
    return ctx


def _muons(n=3000, seed=2):
    rng = np.random.default_rng(seed)
    o = np.stack([rng.uniform(-800, 800, n), np.full(n, 3000.0), rng.uniform(10, 450, n)], 1)
    d = np.stack([rng.normal(0, 0.05, n), -np.ones(n), rng.normal(0, 0.05, n)], 1)
    d /= np.linalg.norm(d, axis=1)[:, None]
    return dict(origin=o, direction=d, p=rng.uniform(1000, 5000, n))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_evaluate_placed_variants_both_dtypes(world, dtype):
    paths, cfg = world
    rem = (P.Remedy("thin", sheets="type2", min_thickness=5.0),
           P.Remedy("move", sectors=("A",), direction=(0, 0, 1), max_mm=600.0, step=50.0))
    cfg = P.PlacementConfig(upstream=cfg.upstream, downstream=cfg.downstream, hall=cfg.hall, stagger_mm=(0.0,),
                            remedies=rem)
    fc = F.FomConfig(placement=cfg, finder_cuts=False, stack_range=False)
    r = F.evaluate(THICK, _muons(), _ctx(paths["counters"]), fc, dtype=dtype)
    assert r["dtype"] == np.dtype(dtype).name and r["placement"]["feasible"]
    v = {x["variant"]: x for x in r["placement_variants"]}
    assert set(v) == {"base", "thin", "move"} and not v["base"]["feasible"] and v["thin"]["scored"]
    assert v["move"]["uncovered_fraction"] > v["thin"]["uncovered_fraction"]   # moved out of the muon band
    assert r["placement"]["variant"] == "thin" and r["deadtime_shield_uncalibrated"]
    assert r["inefficiency"] == min(x["inefficiency"] for x in v.values() if x["feasible"])
    m = KT.metrics_of(r)
    assert m["placement_feasible"] == 1.0 and m["placement_shield_modified"] == 1.0 and m["placement_n_variants"] == 3
    json.dumps(r, default=float)
    plain = F.evaluate(THICK, _muons(), _ctx(paths["counters"]), F.FomConfig(finder_cuts=False, stack_range=False),
                       dtype=dtype)
    assert "placement" not in plain and not set(KT.PLACEMENT_METRICS) & set(KT.metrics_of(plain))


def test_kit_placement_params():
    assert KT.PLACE_POLICIES == P.POLICIES
    assert KT.placement_config("", 60, 3, 120, 500, "") is None
    c = KT.placement_config("each", 60, 3, 120, 500, "thin:u:Outer CRV-U:10")
    assert c.stagger_mm == (60.0, 50.0, 40.0, 30.0, 20.0, 10.0, 0.0) and c.remedies[0].min_thickness == 10.0
    assert KT.placement_config("first", 0, 3, 120, 500, "").stagger_mm == (0.0,)
    assert KT.remedy_words("move:top:T1,T2,T3:0,0,1:50; thin:u:Outer CRV-U|Inner CRV-U:10;"
                           "movethin:b:T1,E1:0,0,1:60:Outer CRV-U:10") == [
        "move:top:T1,T2,T3:0,0,1:50", "thin:u:Outer CRV-U|Inner CRV-U:10", "movethin:b:T1,E1:0,0,1:60:Outer CRV-U:10"]
    for bad in ("move:top:T1:0,0:50", "shift:x:T1:1", "thin:u:x:10;thin:u:y:5"):
        with pytest.raises(ValueError):
            KT.remedy_words(bad)
