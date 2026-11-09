"""Synthetic CrvStep tables without Mu2e input: vertical-ish muons through the CRV bars, one step per bar crossing.

For response tests and benches where G4 tables cannot be shipped (Polaris). Each event is one muon (cos(theta) ~ 0.85,
uniform over the CRV footprint) traced with the bar crossings of crvfastsim.crossings; visible energy per crossing
= 0.18 MeV/mm x path x a Landau-like factor. Columns follow tables.py (steps, particles).

    python3 -m crvfastsim.response.synthsteps --events N --out steps.root
"""
import argparse

import numpy as np
import pandas as pd

from .. import crossings, geometry, tables


def make(n_events, seed=1, counters="Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"):
    g = geometry.from_offline_config(geometry.find_counters(counters))
    rng = np.random.default_rng(seed)
    lo, hi = (g.center - g.halflen).min(0), (g.center + g.halflen).max(0)
    o = np.c_[rng.uniform(lo[0], hi[0], n_events), np.full(n_events, hi[1] + 100.0), rng.uniform(lo[2], hi[2], n_events)]
    ct = rng.uniform(0.7, 1.0, n_events)
    ph = rng.uniform(0, 2 * np.pi, n_events)
    st = np.sqrt(1 - ct ** 2)
    d = np.c_[st * np.cos(ph), -ct, st * np.sin(ph)]
    xs = crossings.crossings(o, d, g)
    it, bar, t_in, t_out, path = xs[:, 0].astype(int), xs[:, 1].astype(int), xs[:, 2], xs[:, 3], xs[:, 4]
    p = rng.lognormal(np.log(3000), 0.8, n_events)
    ev = it + 1
    t0 = rng.uniform(0, 9e4, n_events)[it] + t_in / 299.79
    steps = pd.DataFrame({"run": np.uint32(1400), "subrun": np.uint32(1), "event": ev.astype(np.uint32),
                          "nstep": np.ones(len(it), np.int64), "bar": bar.astype(np.int64), "sp": np.ones(len(it), np.int64),
                          "t0": t0, "t1": t0 + path / 299.79})
    for j, c in enumerate("xyz"):
        steps[f"{c}0"] = o[it, j] + d[it, j] * t_in
        steps[f"{c}1"] = o[it, j] + d[it, j] * t_out
        steps[f"p{c}0"] = d[it, j] * p[it]
    steps["p1"] = p[it]
    steps["len"] = path
    steps["edep"] = 0.18 * path * rng.gumbel(1.0, 0.15, len(it)).clip(0.5, 5)
    parts = pd.DataFrame({"run": np.uint32(1400), "subrun": np.uint32(1), "event": np.arange(1, n_events + 1).astype(np.uint32),
                          "key": np.ones(n_events, np.int64), "parent": -np.ones(n_events, np.int64),
                          "pdg": np.full(n_events, 13, np.int32), "isprim": np.ones(n_events, np.int64)})
    return steps, parts


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--events", type=int, default=20000)
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--out", required=True)
    x = a.parse_args()
    s, p = make(x.events, x.seed)
    tables.write(x.out, s, p, as_ttree=True)
    print(f"{len(s)} steps, {x.events} events -> {x.out}")
