"""crvfastsim.design.placement on the Run 2 envelope (public Offline files: crv_counters_v10, ExtShieldUpstream_v07,
ExtShieldDownstream_v06, mu2eHall_v03; skipped where they cannot be found) against the hand-fitted G1 placements.

Ground truth: the Geant4 surface check (one event, crsVeto and the external shielding) of the hand placements. It
was clean for crvd03 R41 / R28 / R51 (it11), crvd04 P39 / P30 (it3) and the second fit of crvd04 C with a 10 mm outer
CRV-U sheet (P45T2: E1 / E2 moved 46 mm downstream), and reported 28 overlaps for each of the first C variants P45S
(T1-T3 moved 15 mm downstream) and P45T (outer CRV-U sheet 10 mm): T1-T3 against the R1 / R2 / L1 upper segments by
1.1 mm, U against E1 / E2, and the inner CRV-U sheet against the inner CRV-TS sheet and the downstream shielding. The
checker must agree with every verdict; the solver must place the five clean designs as the hand fits did, and C with
the thinner sheet without hand tuning."""
import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crvfastsim import geometry as G                                    # noqa: E402
from crvfastsim.design import envelope as E                              # noqa: E402
from crvfastsim.design import layout as L                                # noqa: E402
from crvfastsim.design import module as M                                # noqa: E402
from crvfastsim.design import placement as P                             # noqa: E402

FILES = (M.RUN2_COUNTERS, E.UPSTREAM, E.DOWNSTREAM, E.HALL)

# the designs (6 layers, 5 of 6 for the coincidence) and their hand placements (shifts along -layerDirection [mm])
DESIGNS = {
    "R41": dict(thickness=24.71, width=25.0, absorber=10.0, fiber_diameter=2.0, max_length=4648.0),
    "R28": dict(thickness=21.99, width=35.93, absorber=22.76, fiber_diameter=2.0, max_length=4600.0),
    "R51": dict(thickness=10.0, width=25.0, absorber=30.0, fiber_diameter=2.0, max_length=3586.0),
    "P39": dict(thickness=30.0, width=42.38, absorber=9.79, fiber_diameter=1.548, max_length=5984.0),
    "P30": dict(thickness=30.0, width=44.92, absorber=22.28, fiber_diameter=1.672, max_length=5517.0),
    "P45": dict(thickness=30.0, width=25.0, absorber=30.0, fiber_diameter=1.0, max_length=4258.0),
}


def _hand(t, e, u, d, stagger, l1=None, move=None, thickness=None):
    sh = {**{s: t for s in ("T1", "T2", "T3")}, "E1": e, "E2": e, "U": u, **{s: d for s in ("D1", "D2", "D3", "D4")}}
    if l1 is not None:
        sh["L1"] = l1
    return dict(placement=L.Placement(shifts=sh, moves=move or {}, no_feb=("L1",)), stagger=stagger,
                thickness=thickness or {})


HAND = {
    "R41": _hand(63.3, 70.8, 98.6, 64.0, 60.0),
    "R28": _hand(64.6, 24.1, 147.4, 113.0, 60.0),
    "R51": _hand(50.8, 46.3, 117.6, 83.0, 50.0),
    "P39": _hand(55.0, 32.0, 131.9, 96.9, 60.0),
    "P30": _hand(55.0, -30.0, 170.4, 159.4, 60.0),
    "P45S": _hand(45.0, -79.0, 155.0, 198.0, 50.0, l1=34.4, move={s: [0.0, 0.0, 15.0] for s in ("T1", "T2", "T3")}),
    "P45T": _hand(45.0, -79.0, 170.0, 198.0, 50.0, l1=34.4, thickness={"Outer CRV-U": 10.0}),
    # the second C-thinpoly fit (it2): E1 / E2 moved 46 mm downstream, U further upstream; Geant4: 0 overlaps
    "P45T2": _hand(41.0, -83.0, 185.0, 198.0, 50.0, l1=34.4, move={s: [0.0, 0.0, 46.0] for s in ("E1", "E2")},
                   thickness={"Outer CRV-U": 10.0}),
}
CLEAN = ("R41", "R28", "R51", "P39", "P30", "P45T2")
FIVE = CLEAN[:5]                                   # the designs that place without a remedy


def design(tag):
    return M.Design(n_layers=6, spare_layers=1, **DESIGNS[tag[:3]])


@pytest.fixture(scope="module")
def env():
    try:
        for f in FILES:
            G.find_counters(f)
    except FileNotFoundError as e:
        pytest.skip(str(e))
    return E.Envelope.load(G.parse_simpleconfig(L.read_text(M.RUN2_COUNTERS)))


# the hand placements and their Geant4 references used the G1 option-B v3 rules: length splits at max_length, each
# sector at its stock firstCounter (end strips open), every sector shifted alone. LEGACY reproduces them; the
# defaults (stock counter lengths, close-up, tied chains) are tested below.
LEGACY_LAYOUT = L.LayoutConfig(split=True, close_up=False)
LEGACY = P.PlacementConfig(layout=LEGACY_LAYOUT, tie_chains=False)
TOUCH = P.PlacementConfig(layout=LEGACY_LAYOUT, tie_chains=False, clearance=0.0, sheet_clearance=0.0)   # overlaps
#                                                                                       only, as the Geant4 check


def test_envelope_reads_the_run2_shielding(env):
    names = [(s.name, s.type, s.label) for s in env.sheets]
    assert [n[1] for n in names] == [15, 16, 17, 18, 19] and names[0][0] == "ExtShieldUpstreamBox_29"
    assert all(n[2].startswith("Borated poly") for n in names)
    concrete = [i for i, v in enumerate(env.fixed.volume) if v == "ExtShieldUpstreamBox_11"]   # BRW-175, type 6
    assert abs(env.fixed.hi[concrete, 2].max() - (-3087.0 + 911.9 / 2 + 5.0 / 4)) < 1e-9          # dims/2 + tols/4
    stock = P.check(M.Design(), L.Placement(), cfg=TOUCH, env=env)
    assert stock["violations"] == [] and stock["feb_conflicts"] == []


@pytest.mark.parametrize("tag", CLEAN)
def test_hand_placements_clean_where_geant4_is(env, tag):
    h = HAND[tag]
    r = P.check(design(tag), h["placement"], cfg=TOUCH, stagger_mm=h["stagger"], thickness=h["thickness"], env=env)
    assert r["violations"] == []


@pytest.mark.parametrize("tag", ("P45S", "P45T"))
def test_c_variants_overlap_where_geant4_reports(env, tag):
    h = HAND[tag]
    r = P.check(design(tag), h["placement"], cfg=TOUCH, stagger_mm=h["stagger"], thickness=h["thickness"], env=env)
    v = r["violations"]

    def has(a, b, kind, lo=0.0, hi=1e9):
        return any(x["kind"] == kind and x["a"].startswith(a) and x["b"].startswith(b) and lo <= x["depth_mm"] <= hi
                   for x in v)
    for side in ("R1s1", "R2s1", "L1s1"):                  # Geant4: T vs the side sectors' upper segments, 1.1 mm
        assert any(x["kind"] == "sector-sector" and side in (x["a"], x["b"]) and abs(x["depth_mm"] - 1.1) < 0.05
                   for x in v), side
    assert has("E1", "Us1", "sector-sector") and has("E2", "Us1", "sector-sector")
    assert has("ExtShieldUpstreamBox_30", "ExtShieldDownstreamBox_", "sheet-envelope")
    assert has("ExtShieldUpstreamBox_30", "ExtShieldUpstreamBox_31", "sheet-sheet")
    r3 = P.check(design(tag), h["placement"], cfg=LEGACY, stagger_mm=h["stagger"], thickness=h["thickness"], env=env)
    trims = {v["label"]: v["trimmed_mm"].get("-z", 0.0) for v in r3["sheets"].values()}
    want = (42.1, 53.3) if tag == "P45S" else (27.1, 38.3)  # the hand fit's trims against U (3 mm clearance)
    assert [trims["Borated poly: Outer CRV-TS"], trims["Borated poly: Outer CRV-TS-Ext"]] == pytest.approx(want, abs=0.06)


@pytest.mark.parametrize("tag", FIVE)
def test_hand_sheet_moves_reproduced(env, tag):
    h = HAND[tag]
    r = P.check(design(tag), h["placement"], cfg=LEGACY, stagger_mm=h["stagger"], env=env)
    got = {v["label"]: v["moved_mm"] for v in r["sheets"].values()}
    want = {"R41": (14.4, 24.4, 17.2), "R28": (61.8, 72.0, 112.8), "R51": (26.1, 56.0, 60.8)}.get(tag)
    if want:                                               # the hand fit's own sheet moves (outer U, TS, TS-Ext)
        assert [got["Borated poly: Outer CRV-U"], got["Borated poly: Outer CRV-TS"],
                got["Borated poly: Outer CRV-TS-Ext"]] == pytest.approx(want, abs=0.1)
    assert got["Borated poly: Inner CRV-U"] == 0 and got["Borated poly: Inner CRV-TS"] == 0


def _hand_sheets(env, tag):
    h = HAND[tag]
    r = P.check(design(tag), h["placement"], cfg=LEGACY, stagger_mm=h["stagger"], thickness=h["thickness"], env=env)
    return {v["label"]: v["moved_mm"] for v in r["sheets"].values()}


@pytest.mark.parametrize("tag", FIVE)
def test_auto_placement_matches_the_hand_fit(env, tag):
    """The solver places the five Geant4-clean designs as the hand fits did: the same sectors shifted (or fewer),
    the same joint stagger, FEB boxes dropped only where they hit something (the hand fits dropped L1's in every
    design), every shift within 2.5 mm of the hand value for crvd03 (the hand values are Geant4 depth + 5 mm, the
    solver keeps 3 mm), within 7.5 mm for crvd04 except U, which the hand fit kept up to 26.3 mm further upstream
    than needed, and no sheet moved more than by the hand fit (+ 1 mm)."""
    h = HAND[tag]
    p = P.place(design(tag), cfg=LEGACY, env=env)[0]
    assert p.feasible and p.remedy is None and p.stagger_mm == h["stagger"]
    assert set(p.placement.no_feb) <= set(h["placement"].no_feb)
    assert set(p.placement.shifts) <= set(h["placement"].shifts)
    tol = 2.5 if tag[0] == "R" else 7.5
    for sec, s in p.placement.shifts.items():
        want = h["placement"].shifts[sec]
        if sec == "U" and tag[0] == "P":
            assert want - 26.5 <= s <= want + 0.5, (sec, s, want)
        else:
            assert abs(s - want) <= tol, (sec, s, want)
    hand = _hand_sheets(env, tag)
    for v in p.sheets.values():
        assert v["thickness_mm"] == v["stock_thickness_mm"] and v["moved_mm"] <= hand[v["label"]] + 1.0
        assert sum(v["trimmed_mm"].values()) <= 2.0             # at most a sliver off the TS-Ext sheet end
    again = P.check(design(tag), p.placement, cfg=TOUCH, stagger_mm=p.stagger_mm, env=env)
    assert again["violations"] == []


@pytest.mark.slow
def test_c_needs_both_remedies_together(env):
    """crvd04 C (P45, a 330 mm stack): neither the top-sector move (P45S) nor the thinner outer CRV-U sheet (P45T)
    places it on its own, as the Geant4 check of the first hand variants found: U also meets the upstream ends of E1 / E2 and is
    held between the concrete and the inner CRV-U sheet, which lies against the downstream shielding. Moving the top
    and E sectors downstream with a thinner outer CRV-U sheet places it."""
    rem = tuple(P.Remedy(**kw, step=10.0) for kw in (
        dict(name="topmove", sectors=("T1", "T2", "T3"), max_mm=60.0),
        dict(name="thinU", sheets=("Outer CRV-U",), min_thickness=5.0),
        dict(name="top+thinU", sectors=("T1", "T2", "T3", "T4", "T5", "E1", "E2"), max_mm=150.0,
             sheets=("Outer CRV-U",), min_thickness=5.0)))
    out = {p.label: p for p in P.place(design("P45"), cfg=dataclasses.replace(LEGACY, remedies=rem), env=env)}
    assert not out["base"].feasible and "U" in out["base"].unsolved and out["base"].stagger_mm == 50.0
    assert not out["topmove"].feasible and not out["thinU"].feasible
    both = out["top+thinU"]
    assert both.feasible and both.shield_modified and both.summary()["deadtime_shield_uncalibrated"]
    assert 0 < both.remedy["move_mm"] <= 60.0 and 5.0 <= both.remedy["thickness_mm"]["ExtShieldUpstreamBox_29"] < 25.4
    assert both.placement.no_feb == ("L1",)
    again = P.check(design("P45"), both.placement, cfg=TOUCH, stagger_mm=both.stagger_mm,
                    thickness={"Outer CRV-U": both.remedy["thickness_mm"]["ExtShieldUpstreamBox_29"]}, env=env)
    assert again["violations"] == []


def _synthetic_tables(ctx):
    import numpy as np
    from crvfastsim.design import optics as O
    for sc in ctx.sectors.values():
        length = float(sc.table.split("_")[0])
        ex, ey = np.linspace(0, 9.9, 5), np.linspace(-25.67, 25.67, 9)
        ez = np.linspace(-length / 2, length / 2, 31)
        zc = 0.5 * (ez[1:] + ez[:-1]) + length / 2
        prob = np.ones((4, 8, 30)) * 0.0005 * np.exp(-zc / 4000.0)[None, None, :]
        ctx.tables[sc.table] = O.TableModel(sc.table, prob, ex, ey, ez, (9.9, 25.67, length / 2), 4000.0,
                                            int(sc.table.endswith("_1")))
    return ctx


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_placed_run2_design_scores_in_both_dtypes(env, dtype):
    """A placed Run 2 design (fillers, segments, staggered joints) through fom.evaluate: fillers take their parent's
    response and deadtime terms, the per-sector deadtime is per stock sector, and float32 agrees with float64."""
    import numpy as np
    from crvfastsim.design import fom as F
    rng = np.random.default_rng(5)
    n = 3000
    o = np.stack([rng.uniform(-7000, -800, n), np.full(n, 6000.0), rng.uniform(-2000, 18000, n)], 1)
    d = np.stack([rng.normal(0, 0.2, n), -np.ones(n), rng.normal(0, 0.2, n)], 1)
    d /= np.linalg.norm(d, axis=1)[:, None]
    mu = dict(origin=o, direction=d, p=rng.uniform(1000, 10000, n))
    cfg = F.FomConfig(placement=LEGACY, stack_range=False)
    out = {}
    for dt in ("float64", dtype):
        ctx = _synthetic_tables(F.Context())
        out[dt] = F.evaluate(design("R41"), mu, ctx, cfg, dtype=np.dtype(dt).type)
    r = out[dtype]
    assert r["placement"]["feasible"] and r["placement"]["stagger_mm"] == 60.0
    assert any(k.endswith("F") for k in r["inefficiency_sector"])           # fillers are scored
    assert set(r["coincidence"]) == set(r["inefficiency_sector"])
    assert 0 < r["inefficiency"] < 1 and np.isfinite(r["deadtime_rate_ratio"]) and r["n_bars"] > 0
    assert r["deadtime_shield_uncalibrated"] is False
    # test_design's dtype tolerance plus one track: on this sample float32 flips the finder's slope cut for one
    # muon in D1 (with finder_cuts off the dtypes agree to 3e-7)
    r64 = out["float64"]["inefficiency"]
    assert abs(r["inefficiency"] - r64) <= 1e-4 + 1e-3 * r64 + 1.0 / r["n_muons"]


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_placed_run2_design_with_sector_type_groups(env, dtype):
    """The kit's coinc_groups=sector_type on a placed Run 2 design: fillers join their parent's sector-type group,
    grouping only adds coincidence chances (inefficiency not above the ungrouped one), float32 agrees with float64."""
    import numpy as np
    from crvfastsim.design import fom as F
    from crvfastsim.design import kit_task as KT
    rng = np.random.default_rng(7)
    n = 3000
    o = np.stack([rng.uniform(-7000, -800, n), np.full(n, 6000.0), rng.uniform(-2000, 18000, n)], 1)
    d = np.stack([rng.normal(0, 0.2, n), -np.ones(n), rng.normal(0, 0.2, n)], 1)
    d /= np.linalg.norm(d, axis=1)[:, None]
    mu = dict(origin=o, direction=d, p=rng.uniform(1000, 10000, n))
    ctx0 = F.Context()
    groups = KT.coincidence_groups("sector_type", ctx0.counters)
    out = {}
    for g in ((), groups):
        for dt in ("float64", dtype):
            cfg = F.FomConfig(placement=LEGACY, stack_range=False, coincidence_groups=g)
            out[bool(g), dt] = F.evaluate(design("R41"), mu, _synthetic_tables(F.Context()), cfg,
                                          dtype=np.dtype(dt).type)
    grouped, alone = out[True, dtype], out[False, dtype]
    assert grouped["placement"]["feasible"] and any(k.endswith("F") for k in grouped["inefficiency_sector"])
    assert grouped["inefficiency"] <= alone["inefficiency"] + 1.0 / n
    # dtypes: with the finder cuts on, float32 flips the slope-window cut for a couple of edge muons on this sample
    # (test_placed_run2_design_scores_in_both_dtypes); with them off the grouped placed score agrees tightly
    nocut = [F.evaluate(design("R41"), mu, _synthetic_tables(F.Context()),
                        F.FomConfig(placement=LEGACY, stack_range=False, finder_cuts=False,
                                    coincidence_groups=groups), dtype=np.dtype(dt).type)["inefficiency"]
             for dt in ("float64", dtype)]
    assert abs(nocut[1] - nocut[0]) <= 1e-6       # float32 sums over 3000 muons (#346: 3e-7)
    # fillers in their parent's group: the grouped placed score equals one with the filler pairs spelled out
    names = list(grouped["inefficiency_sector"])
    stock = dict(groups)
    full = groups + tuple((s, stock[s[:-1]]) for s in names if s.endswith("F") and s[:-1] in stock)
    cfg = F.FomConfig(placement=LEGACY, stack_range=False, coincidence_groups=full)
    spelled = F.evaluate(design("R41"), mu, _synthetic_tables(F.Context()), cfg, dtype=np.dtype(dtype).type)
    assert spelled["inefficiency"] == grouped["inefficiency"]


@pytest.mark.slow
def test_c_thinpoly_placed_without_hand_tuning(env):
    """crvd04 C with a thinner outer CRV-U sheet: the automatic fit, given the E1 / E2 downstream move and the sheet
    thinning (down to 10 mm) as one remedy, reaches a clean placement that matches the Geant4-clean hand fit P45T2
    (E1 / E2 +46 mm, T 41, E -83, U 185, sheet 10 mm); the thinning or the move alone does not place."""
    rem = (P.Remedy("thinU", sheets=("Outer CRV-U",), min_thickness=10.0, step=10.0),
           P.Remedy("E", sectors=("E1", "E2"), max_mm=80.0, step=10.0),
           P.Remedy("E+thinU", sectors=("E1", "E2"), max_mm=80.0, sheets=("Outer CRV-U",), min_thickness=10.0,
                    step=10.0))
    out = {p.label: p for p in P.place(design("P45"), cfg=dataclasses.replace(LEGACY, remedies=rem), env=env)}
    assert not out["base"].feasible and not out["thinU"].feasible and not out["E"].feasible
    p, h = out["E+thinU"], HAND["P45T2"]
    assert p.feasible and p.stagger_mm == h["stagger"] and p.placement.no_feb == ("L1",)
    assert abs(p.remedy["move_mm"] - 46.0) <= 5.0
    t = p.remedy["thickness_mm"]["ExtShieldUpstreamBox_29"]
    assert 10.0 <= t < 25.4 and p.shield_modified and p.summary()["deadtime_shield_uncalibrated"]
    for sec, tol in (("T1", 1.0), ("T2", 1.0), ("T3", 1.0), ("E2", 1.0), ("U", 3.0), ("L1", 2.5)):
        assert abs(p.placement.shifts[sec] - h["placement"].shifts[sec]) <= tol, sec
    again = P.check(design("P45"), p.placement, cfg=TOUCH, stagger_mm=p.stagger_mm, thickness={"Outer CRV-U": t},
                    env=env)
    assert again["violations"] == [] and again["feb_conflicts"] == []


def test_close_up_chains_and_stock_unchanged(env):
    """Stock-adjacent sectors are found as chains; the stock design needs no close-up (its tiling covers each sector
    exactly), so its layout text keeps every stock firstCounter."""
    c = G.parse_simpleconfig(L.read_text(M.RUN2_COUNTERS))
    ch = L.sector_chains(c)
    assert {"T2": "T1", "T3": "T2", "T4": "T3", "T5": "T4"}.items() <= ch.items()
    assert L.close_up_offsets(c, float(c["crs.scintillatorBarWidth"])) == {}
    t = G.parse_simpleconfig(L.counters_text(M.Design(), L.Placement()))
    for nm in c["crs.sectorNames"]:
        assert t[f"crs.firstCounter{nm}"] == pytest.approx(c[f"crs.firstCounter{nm}"])


def test_close_up_removes_the_top_boundary_cracks(env):
    """crvd04 A (42.38 mm counters): without close-up the no-overshoot tiling leaves each top sector's end strip
    open in every layer (28-54 mm at T2|T3, T3|T4, T4|T5); with it every boundary is the stock module gap."""
    from crvfastsim.design import tiling_plot as TP
    gap0 = float(G.parse_simpleconfig(L.read_text(M.RUN2_COUNTERS))["crs.gapBetweenModules"])
    for close, check in ((False, lambda g: max(g) > 25.0), (True, lambda g: max(abs(x) for x in g) <= gap0 + 0.5)):
        cfg = L.LayoutConfig(close_up=close)
        dg = L.build(design("P39"), L.counters_text(design("P39"), L.Placement(), cfg=cfg), None, cfg)
        rows, nl = TP.top_boxes(dg)
        bnd = {(b["prev"], b["next"]): b["gap_mm_per_layer"] for b in TP.boundaries(rows, nl)}
        for pair in (("T2F", "T3"), ("T3F", "T4"), ("T4", "T5")):
            assert check(bnd[pair]), (close, pair, bnd[pair])


def test_tiling_plot_nominal(env, tmp_path):
    from crvfastsim.design import tiling_plot as TP
    out = tmp_path / "run2.png"
    assert TP.main(["--out", str(out)]) == 0
    s = __import__("json").load(open(tmp_path / "run2.json"))
    assert out.stat().st_size > 10000 and s["n_layers"] == 4
    assert all(max(abs(x) for x in b["gap_mm_per_layer"]) < 5.0 for b in s["boundaries"])


@pytest.mark.parametrize("tag", ["P39", "P30"])
def test_default_layout_places_flat_top_with_stock_lengths(env, tag):
    """The default rules on crvd04 A / B: stock counter lengths (no joints), closed-up boundaries and one shift for
    the top chain. The base placement cannot lower the whole top plane; the E1 / E2 downstream move places the
    design with the top sectors at their stock height (no step)."""
    rem = P.parse_remedies("move:E:E1,E2:0,0,1:80")
    vs = {v.label: v for v in P.place(design(tag), cfg=P.PlacementConfig(remedies=rem), env=env)}
    v = vs["E"]
    assert v.feasible and v.violations == []
    top = {s: v.placement.shifts.get(s, 0.0) for s in ("T1", "T2", "T3", "T4", "T5")}
    assert len(set(round(x, 6) for x in top.values())) == 1
    dg = L.build(design(tag), v.text, None, v.layout)
    assert int(dg.segment.max()) == 0
