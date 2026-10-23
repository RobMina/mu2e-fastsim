"""For muon events with CRV activity in G4 but none in the fast steps: rerun the fast transport (stage C) on just
those primaries and report, per muon, whether the fast line enters the CRV bounding box, crosses any bar, how close
(perpendicular, in the bar-normal plane) it passes to the bars G4 hit, and the G4 entry angle vs the generated one.
Decides whether in-CRV scattering (kicks at bar crossings) can recover them: only muons whose fast line crosses at
least one bar can be kicked into others.

    python -m valhist.missprobe --g4 crvsteps...root ... --fast fast1b...root ... --gen gen...root ...
"""
import argparse
import json

import numpy as np

from crvfastsim import geometry, tables, transport as TR
from crvfastsim import crossings as CX
from crvfastsim.analyses import v2compare as V2
from crvfastsim.analyses.transport import default_gdml
from mu2efastsim import world as world_mod


def _key(df):
    return (df.subrun.to_numpy().astype(np.int64) << 20) | df.event.to_numpy().astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g4", nargs="+", required=True)
    ap.add_argument("--fast", nargs="+", required=True)
    ap.add_argument("--gen", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    geom = geometry.from_offline_config(geometry.find_counters(V2.DEFAULT_COUNTERS))
    gs = tables.read_many(a.g4, "steps", tables.KEY + ["bar", "sp", "edep"])
    gp = tables.read_many(a.g4, "particles", tables.KEY + ["key", "pdg", "isprim"])
    fs = tables.read_many(a.fast, "steps", tables.KEY + ["edep"])
    gen = tables.read_many(a.gen, "gen")
    for x in (gs, gp, fs, gen):
        x["k"] = _key(x)
    mu = tables.primary_muons(gen)
    mu["k"] = _key(mu)
    gs = gs.merge(gp[["k", "key", "pdg", "isprim"]].rename(columns={"key": "sp"}), on=["k", "sp"], how="left")
    pm = gs[(gs.isprim == 1) & (gs.pdg.abs() == 13)]
    g_only = sorted(set(pm.k) - set(fs.k))
    sub = gen[gen.k.isin(g_only)].drop(columns="k")
    world = [p for p in world_mod.load(default_gdml(), y_min=TR.Y_MSC_PLANE) if p[0] != "G4_POLYSTYRENE"]
    out = {"g4_only_primary_muon_events": len(g_only)}
    # rerun the stage-C transport several times (different MSC draws) on these primaries: how often does the fast
    # muon cross a bar at all?
    hits = []
    for r in range(5):
        st, _ = TR.transport(sub, geom, np.random.default_rng([a.seed, r]), stage="C", world=world)
        hits.append(len(set(_key(st)) & set(g_only)))
    out["fast_rerun_events_with_steps_of_5_draws"] = hits
    # straight generated line: enters the CRV box? distance to the nearest bar G4 hit (box distance)
    m = mu[mu.k.isin(g_only)].set_index("k").loc[g_only]
    o = m[["x", "y", "z"]].to_numpy()
    d = m[["px", "py", "pz"]].to_numpy()
    d /= np.linalg.norm(d, axis=1)[:, None]
    lo, hi = CX.bar_bounds(geom)
    lo_all, hi_all = lo.min(0), hi.max(0)
    b_in, b_out = CX.slab_crossings(o, d, lo_all[None, :], hi_all[None, :])
    enters_box = b_out[:, 0] > np.maximum(b_in[:, 0], 0)
    xs = CX.crossings(o, d, geom)
    crosses_bar = np.zeros(len(o), bool)
    crosses_bar[xs[:, 0].astype(int)] = True
    out["straight_line_enters_crv_box_frac"] = float(enters_box.mean())
    out["straight_line_crosses_a_bar_frac"] = float(crosses_bar.mean())
    # minimum distance of the straight line to the boxes of the bars G4's muon hit
    hitbars = pm[pm.k.isin(g_only)].groupby("k").bar.apply(lambda b: np.unique(b.to_numpy()))
    dmin = []
    for i, k in enumerate(g_only):
        bars = hitbars[k]
        ts = np.linspace(0, 60000, 60001)
        best = np.inf
        # closest approach to each box: sample the line near the box's y range
        for b in bars:
            c, h = geom.center[b], geom.halflen[b]
            if abs(d[i, 1]) < 1e-9:
                continue
            t0 = (c[1] - o[i, 1]) / d[i, 1]
            tt = t0 + np.linspace(-3000, 3000, 6001)
            p = o[i] + d[i] * tt[:, None]
            q = np.maximum(np.abs(p - c) - h, 0)
            best = min(best, float(np.sqrt((q ** 2).sum(1)).min()))
        dmin.append(best)
    dmin = np.array(dmin)
    out["straight_line_to_g4_bars_mm_quartiles"] = np.percentile(dmin, [10, 25, 50, 75, 90]).round(1).tolist()
    out["frac_within_10mm"] = float((dmin < 10).mean())
    out["frac_within_50mm"] = float((dmin < 50).mean())
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
