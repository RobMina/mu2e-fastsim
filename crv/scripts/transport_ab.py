"""Stage-C transport A/B on one gen table: two option sets, same seed; reports time per arm and step-level equality.

    python3 scripts/transport_ab.py gen.root --gdml mu2e_extracted_v04.gdml [--a unbanded] [--b banded] [--max-events N]
Arms: banded (default world), unbanded (piece-by-piece columns). Equal = same rows and every column within 1e-9 relative.
"""
import argparse
import json
import time

import numpy as np

from crvfastsim import geometry, tables
from crvfastsim import transport as tr

ARMS = ("banded", "unbanded")


def run_arm(arm, gen, geom, gdml, seed):
    t0 = time.perf_counter()
    world, ab = tr.load_world(gdml, banded=(arm == "banded"))
    t1 = time.perf_counter()
    st, pa = tr.transport(gen, geom, np.random.default_rng([seed, 0]), stage="C", world=world, absorbers=ab)
    return st, pa, {"load_s": t1 - t0, "transport_s": time.perf_counter() - t1}


def compare_columns(gen, geom, gdml):
    """Per height band, the world columns of both modes for the near-CRV muons (no randomness): largest |difference|
    and the count of muon-band pairs that are zero in one mode and not in the other."""
    from mu2efastsim import world as world_mod
    mu = tr.all_primary_muons(gen)
    p0 = np.sqrt(mu.px ** 2 + mu.py ** 2 + mu.pz ** 2).to_numpy()
    o, d = mu[["x", "y", "z"]].to_numpy(), mu[["px", "py", "pz"]].to_numpy() / p0[:, None]
    down = tr.near_crv(o, d, geom)
    o, d = o[down], d[down]
    wb, _ = tr.load_world(gdml, banded=True)
    wu, _ = tr.load_world(gdml, banded=False)
    out = []
    for k in range(len(tr.BAND_EDGES_Y) - 1):
        y_top, y_bot = tr.BAND_EDGES_Y[k], tr.BAND_EDGES_Y[k + 1]
        s_top = (o[:, 1] - y_top) / -d[:, 1]
        seg = (o[:, 1] - y_bot) / -d[:, 1] - s_top
        ok = np.flatnonzero((s_top >= 0) & (seg > 0))
        pu = [q for q in wu if q[3][1] >= y_bot and q[2][1] <= y_top]
        cu = world_mod.column(pu, (o + d * s_top[:, None])[ok], d[ok], seg[ok]) if pu else {}
        cb = world_mod.column(wb.band(k), (o + d * s_top[:, None])[ok], d[ok], seg[ok]) if len(wb.band(k)) else {}
        for m in sorted(set(cu) | set(cb)):
            a, b = cu.get(m, np.zeros(len(ok))), cb.get(m, np.zeros(len(ok)))
            out.append({"band": k, "mat": m, "max_abs_diff_mm": float(np.abs(a - b).max()) if len(ok) else 0.0,
                        "zero_mismatch": int(((a > 0) != (b > 0)).sum()), "n": int(len(ok))})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gen")
    ap.add_argument("--gdml", required=True)
    ap.add_argument("--counters", default="Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt")
    ap.add_argument("--a", default="unbanded", choices=ARMS)
    ap.add_argument("--b", default="banded", choices=ARMS)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--columns", action="store_true", help="compare the world columns per band instead (no transport)")
    a = ap.parse_args()
    geom = geometry.from_offline_config(geometry.find_counters(a.counters))
    gen = tables.read(a.gen, "gen")
    if a.max_events:
        gen = gen[gen.event.isin(np.sort(gen.event.unique())[:a.max_events])].reset_index(drop=True)
    if a.columns:
        print(json.dumps([r for r in compare_columns(gen, geom, a.gdml) if r["max_abs_diff_mm"] > 1e-9 or r["zero_mismatch"]],
                         indent=1))
        return
    sa, _, ta = run_arm(a.a, gen, geom, a.gdml, a.seed)
    sb, _, tb = run_arm(a.b, gen, geom, a.gdml, a.seed)
    res = {"a": a.a, "b": a.b, "time_a": ta, "time_b": tb, "steps_a": len(sa), "steps_b": len(sb)}
    if len(sa) == len(sb):
        rel = {c: float(np.max(np.abs(sa[c].to_numpy(float) - sb[c].to_numpy(float))
                               / np.maximum(np.abs(sa[c].to_numpy(float)), 1e-12))) if len(sa) else 0.0
               for c in sa.columns}
        res["max_rel_diff"] = rel
        res["equal"] = bool(all(v <= 1e-9 for v in rel.values()))
    else:
        res["equal"] = False
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
