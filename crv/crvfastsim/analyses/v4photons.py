"""v4photons -- V4 at the photon stage: fast photons (response/photons.py) vs Offline's MakeCrvPhotons on the same steps.

Input: CrvStep tables (CrvStepTableDumper; G4 reference or fast). Both arms get the same steps, the same local
coordinates, beta, charge and per-channel yield deviation (--deviation: the CRVPhoton table as CSV "channel,value";
default all zero). The stock arm runs through PyROOT (needs a muse/Offline setup). Per sector group:
  * photons per SiPM per event: mean ratio fast/stock (V4: within 0.5 %, 2 sigma allowed). The arms share the steps,
    so the error is paired: from the spread of the per-(step, SiPM) differences, not of the counts themselves
  * arrival time minus step start time at the 10/50/90 % levels (V4: within 0.2 ns), as the horizontal distance
    between the two empirical CDFs at the reference quantile, (F_fast - F_ref)(q) / f(q) with the density f averaged
    over +-1 ns. The delays in the tables are whole ns, so the distribution is a staircase and a raw quantile can jump
    across a plateau (0.84 ns at a median on identical laws); the raw quantile differences are reported too.
Optionally --max-steps to bound the stock loop. --jax: the fast arm is the JAX twin (response/photons_jax.py);
--no-stock: skip the stock arm and compare --jax against the numpy port instead (no Offline needed, e.g. on Polaris).

    v4photons crvsteps.root [...] [--deviation crvphoton.csv] [--max-steps N] --outdir DIR
"""
import time

import numpy as np

from mu2efastsim import plots

from .. import geometry, tables
from ..response import lookup, photons, stockphotons
from .census import SECTOR_GROUP

GROUPS = ("EX", "T1", "T2", "M")
THRESH = {"pe_per_sipm_rel": 0.005, "time_quantile_ns": 0.2}
DEFAULT_COUNTERS = "Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"


def _deviation(path, n):
    dev = np.zeros(n)
    if path:
        a = np.loadtxt(path, delimiter=",", comments="#")
        dev[a[:, 0].astype(int)] = a[:, 1]
    return dev


def _summary(step_ev, step_bar, out, nsteps, t0, geom):
    s, c, t = out
    grp = SECTOR_GROUP[geom.sector[c // 4]]
    return {"n": len(s), "per_step_chan": np.bincount(s * 4 + c % 4, minlength=nsteps * 4), "grp": grp,
            "dt": t - t0[s], "ev": step_ev[s], "chan": c}


def run(inputs, outdir, deviation=None, max_steps=None, seed=1, counters=None, jax=False, no_stock=False, **kw):
    geom = geometry.from_offline_config(geometry.find_counters(counters or DEFAULT_COUNTERS))
    steps = tables.read_many(tables.expand(inputs), "steps")
    parts = tables.read_many(tables.expand(inputs), "particles", tables.KEY + ["key", "pdg"])
    if max_steps:
        steps = steps.iloc[:int(max_steps)].reset_index(drop=True)
    pdg = steps.merge(parts.rename(columns={"key": "sp"}), on=tables.KEY + ["sp"], how="left").pdg.fillna(0).to_numpy()
    cfg = photons.ResponseConfig()
    dev = _deviation(deviation, geom.nbars * 4)
    s_loc, e_loc, beta, charge, known = photons.step_inputs(steps, pdg, geom)
    bar = steps.bar.to_numpy().astype(np.int64)
    names = np.array(geom.sector_names)[geom.sector[bar]]
    t0 = steps.t0.to_numpy()
    ev = (steps.subrun.to_numpy().astype(np.int64) << 32) | steps.event.to_numpy().astype(np.int64)

    tabs = photons.load_tables(cfg)
    if jax:
        import jax as _jax
        from ..response import photons_jax
        dtabs = {k: photons_jax.DeviceTable(v) for k, v in tabs.items()}
        photons_jax.make_photons(steps.iloc[:200], pdg[:200], geom, cfg, dtabs, dev, np.random.default_rng(0),
                                 _jax.random.PRNGKey(0))  # compile outside the timing
        a = time.perf_counter()
        fast = photons_jax.make_photons(steps, pdg, geom, cfg, dtabs, dev, np.random.default_rng(int(seed)),
                                        _jax.random.PRNGKey(int(seed)))
        t_fast = time.perf_counter() - a
    else:
        a = time.process_time()
        fast = photons.make_photons(steps, pdg, geom, cfg, tabs, dev, np.random.default_rng(int(seed)))
        t_fast = time.process_time() - a
    a = time.process_time()
    if no_stock:
        stock = photons.make_photons(steps, pdg, geom, cfg, tabs, dev, np.random.default_rng(int(seed) + 1))
    else:
        st = stockphotons.StockPhotons(cfg, lookup.lookup_dir(), seed=int(seed))
        stock = st.run(s_loc, e_loc, t0, steps.t1.to_numpy(), beta, charge, steps.edep.to_numpy(), steps.len.to_numpy(),
                       bar, names, dev)
    t_stock = time.process_time() - a

    F, S = _summary(ev, bar, fast, len(steps), t0, geom), _summary(ev, bar, stock, len(steps), t0, geom)
    out = {"steps": len(steps), "unknown_pdg_steps": int((~known).sum()), "photons_fast": F["n"], "photons_stock": S["n"],
           "fast_arm": "jax" if jax else "numpy", "reference_arm": "numpy" if no_stock else "stock",
           "time_s_fast": t_fast, "cpu_s_reference": t_stock, "thresholds": THRESH, "groups": {}}
    step_grp = SECTOR_GROUP[geom.sector[bar]]
    for g in GROUPS:
        ms = np.repeat(step_grp == g, 4)
        pf, ps = F["per_step_chan"][ms].astype(float), S["per_step_chan"][ms].astype(float)
        if ps.sum() == 0:
            continue
        # per SiPM per event: group photons by (event, channel)
        def per_sipm(D):
            m = D["grp"] == g
            k = D["ev"][m] * 4096 + D["chan"][m]
            return np.unique(k, return_counts=True)
        kf, nf = per_sipm(F)
        ks, ns = per_sipm(S)
        allk = np.union1d(kf, ks)
        vf = np.zeros(len(allk)); vf[np.searchsorted(allk, kf)] = nf
        vs = np.zeros(len(allk)); vs[np.searchsorted(allk, ks)] = ns
        rel = vf.mean() / vs.mean() - 1
        err = np.sqrt(len(pf) * np.var(pf - ps)) / ps.sum()  # paired: same steps in both arms
        df, ds = np.sort(F["dt"][F["grp"] == g]), np.sort(S["dt"][S["grp"] == g])
        qf, qs = np.quantile(df, (0.1, 0.5, 0.9)), np.quantile(ds, (0.1, 0.5, 0.9))

        def cdf(arr, x):
            return np.searchsorted(arr, x, side="right") / len(arr)
        dens = (cdf(ds, qs + 1.0) - cdf(ds, qs - 1.0)) / 2.0
        shift = -(cdf(df, qs) - cdf(ds, qs)) / np.maximum(dens, 1e-12)
        res = {"sipm_events": len(allk), "pe_per_sipm_fast": vf.mean(), "pe_per_sipm_stock": vs.mean(),
               "pe_per_sipm_rel": rel, "pe_per_sipm_rel_err": err, "photons_per_step_rel": pf.sum() / ps.sum() - 1,
               "dt_q10_50_90_fast": qf, "dt_q10_50_90_stock": qs, "dt_q_diff_raw": qf - qs, "dt_cdf_shift_ns": shift}
        res["checks"] = {"pe_per_sipm": max(abs(rel) - 2 * err, 0) <= THRESH["pe_per_sipm_rel"],
                         "time_quantiles": bool(np.all(np.abs(shift) <= THRESH["time_quantile_ns"]))}
        res["pass"] = all(res["checks"].values())
        out["groups"][g] = res
    out["pass"] = all(r["pass"] for r in out["groups"].values())
    plots.write_json(outdir, "v4photons_stats.json", out)
    print(f"steps {len(steps):,} (unknown pdg {out['unknown_pdg_steps']}); photons fast {F['n']:,} stock {S['n']:,}; "
          f"{out['fast_arm']} {t_fast:.2f} s, {out['reference_arm']} {t_stock:.1f} s cpu ({t_stock / max(t_fast, 1e-9):.1f}x)")
    for g, r in out["groups"].items():
        flags = " ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in r["checks"].items())
        print(f"{g:3s} PE/SiPM/event {r['pe_per_sipm_stock']:.2f} -> {r['pe_per_sipm_fast']:.2f} "
              f"({r['pe_per_sipm_rel']:+.4f} +- {r['pe_per_sipm_rel_err']:.4f}); per step {r['photons_per_step_rel']:+.4f}; "
              f"dt shift q10/50/90 {np.round(r['dt_cdf_shift_ns'], 3)} ns (raw {np.round(r['dt_q_diff_raw'], 2)}) | {flags}")
    print(f"V4-photons {'PASS' if out['pass'] else 'FAIL'}")
    return out
