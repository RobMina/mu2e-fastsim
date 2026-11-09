"""Check the light-propagation knobs on real CrvSteps (GPVM: v6_0 tables on cvmfs, a crvsteps table):
  1. default identity: no knob gives the same photons as a reference package (--ref-path, e.g. a git archive of HEAD)
  2. table values: table_propagation per v6_0 table (Offline crv-fiber-atten prints 166.464/6618.02 EX,
     164.23/7864.6 T1, 163.686/5168.96 T2)
  3. effects: speed_EX 1.08 shifts EX photon times by inv_v (1/1.08 - 1) d; atten_T1 0.58 adds (1/0.58 - 1)/L_T1 to the
     T1 log-slope of photons per step along z.

    python3 scripts/check_propagation_knobs.py CRVSTEPS.root [--ref-path DIR] [--nsteps 200000]
"""
import argparse
import importlib
import sys

import numpy as np


def photons(mod_path, steps, pdg, knobs, seed=1):
    if mod_path:
        sys.path.insert(0, mod_path)
    for m in [k for k in sys.modules if k.startswith("crvfastsim")]:
        del sys.modules[m]
    chain = importlib.import_module("crvfastsim.chain")
    geometry = importlib.import_module("crvfastsim.geometry")
    P = importlib.import_module("crvfastsim.response.photons")
    r3 = importlib.import_module("crvfastsim.reco.r3")
    if mod_path:
        sys.path.remove(mod_path)
    geom = geometry.from_offline_config(geometry.find_counters(r3.DEFAULT_COUNTERS))
    cfg = chain.apply_knobs(chain.ChainConfig(), knobs).response
    tabs = P.load_tables(cfg)
    dev = np.zeros(geom.nbars * 4)
    out = P.make_photons(steps, pdg, geom, cfg, tabs, dev, np.random.default_rng(seed))
    return out, P, geom, tabs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps")
    ap.add_argument("--ref-path", default=None, help="directory holding a reference crvfastsim package")
    ap.add_argument("--nsteps", type=int, default=200000)
    a = ap.parse_args()
    from crvfastsim.reco import r3
    c = r3.load_cache([a.steps], {})[0]
    steps, pdg = c["steps"].iloc[:a.nsteps].reset_index(drop=True), c["pdg"][:a.nsteps]

    (s0, c0, t0), P, geom, tabs = photons(None, steps, pdg, {})
    if a.ref_path:
        (s1, c1, t1), *_ = photons(a.ref_path, steps, pdg, {})
        same = len(s0) == len(s1) and np.array_equal(s0, s1) and np.array_equal(c0, c1) and np.array_equal(t0, t1)
        print(f"1. default identity vs {a.ref_path}: {'IDENTICAL' if same else 'DIFFERENT'} ({len(s0):,} photons)")
    (s0, c0, t0), P, geom, tabs = photons(None, steps, pdg, {})

    print("2. table propagation:")
    prop = {}
    for name, L in tabs.items():
        inv_v, att = P.table_propagation(L)
        prop[name] = (inv_v, att)
        print(f"   {name}: {1 / inv_v:.3f} mm/ns, {att:.2f} mm")

    bar = steps.bar.to_numpy().astype(int)
    sec = geom.sector[bar]
    (sv, cv, tv), *_ = photons(None, steps, pdg, {"speed_EX": 1.08})
    ex = geom.sector[cv // 4] == 0
    ex0 = geom.sector[c0 // 4] == 0
    print(f"3a. speed_EX 1.08: EX photons {ex0.sum():,} -> {ex.sum():,}; "
          f"mean EX time {t0[ex0].mean():.3f} -> {tv[ex].mean():.3f} ns; non-EX times identical: "
          f"{np.array_equal(t0[~ex0], tv[~ex])}")

    (sa, ca, ta), *_ = photons(None, steps, pdg, {"atten_T1": 0.58})
    zl = steps.z0.to_numpy() * 0.5 + steps.z1.to_numpy() * 0.5          # T1 bars along Mu2e z (extracted)
    t1 = sec == 1
    zb = np.linspace(zl[t1].min(), zl[t1].max(), 13)

    def logslope(s):
        n = np.histogram(zl[s][t1[s]], zb)[0]
        m = np.histogram(zl[t1], zb, weights=steps.edep.to_numpy()[t1])[0]
        zc = 0.5 * (zb[1:] + zb[:-1])
        g = (n > 0) & (m > 0)
        return np.polyfit(zc[g], np.log(n[g] / m[g]), 1)[0]

    d = logslope(sa) - logslope(s0)
    exp = (1 / 0.58 - 1) / prop["2370_1"][1]
    print(f"3b. atten_T1 0.58: T1 photons {np.sum(t1[s0]):,} -> {np.sum(t1[sa]):,}; "
          f"|d log-slope| {abs(d) * 1e3:.3f} per m "
          f"(expected {exp * 1e3:.3f} per m, sign set by the T1 readout end)")


if __name__ == "__main__":
    main()
