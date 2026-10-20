"""v4sipm -- V4 at the SiPM-charge stage: fast charges (response/sipm.py) vs Offline's MakeCrvSiPMCharges, same photons.

Photons are made once from the CrvSteps with the validated numpy photon port (response/photons.py) and fed to both arms
(the stock arm through PyROOT, response/stocksipm.py). Per sector group:
  * PE per SiPM per event (sum of charge in PE): mean ratio fast/stock (V4: within 0.5 %, 2 sigma allowed)
  * charge time relative to the step start at the 10/50/90 % levels: CDF distance (V4: within 0.2 ns)
  * diagnostics: charges per photon, fraction of charges below 0.99 PE (recovering pixels), cross-talk share
Needs a muse/Offline setup for the stock arm, as v4photons. --jax: the fast arm is the JAX twin (response/sipm_jax.py);
--no-stock: the reference is the numpy port (no Offline: Polaris).

    v4sipm crvsteps.root [...] [--deviation crvphoton.csv] [--max-steps N] --outdir DIR
"""
import time

import numpy as np

from mu2efastsim import plots

from .. import geometry, tables
from ..response import photons, sipm, stocksipm
from .census import SECTOR_GROUP
from .v4photons import DEFAULT_COUNTERS, GROUPS, THRESH, _deviation


def _cdf_shift(df, ds, levels=(0.1, 0.5, 0.9)):
    df, ds = np.sort(df), np.sort(ds)
    qs = np.quantile(ds, levels)

    def cdf(a, x):
        return np.searchsorted(a, x, side="right") / len(a)
    dens = (cdf(ds, qs + 1.0) - cdf(ds, qs - 1.0)) / 2.0
    return -(cdf(df, qs) - cdf(ds, qs)) / np.maximum(dens, 1e-12)


def run(inputs, outdir, deviation=None, max_steps=None, seed=1, counters=None, jax=False, no_stock=False,
        device=False, calib=None, **kw):
    geom = geometry.from_offline_config(geometry.find_counters(counters or DEFAULT_COUNTERS))
    steps = tables.read_many(tables.expand(inputs), "steps")
    parts = tables.read_many(tables.expand(inputs), "particles", tables.KEY + ["key", "pdg"])
    if max_steps:
        steps = steps.iloc[:int(max_steps)].reset_index(drop=True)
    pdg = steps.merge(parts.rename(columns={"key": "sp"}), on=tables.KEY + ["sp"], how="left").pdg.fillna(0).to_numpy()
    cfg = photons.ResponseConfig()
    dev = _deviation(deviation, geom.nbars * 4)
    s, c, t = photons.make_photons(steps, pdg, geom, cfg, photons.load_tables(cfg), dev, np.random.default_rng(int(seed)))
    ev = ((steps.subrun.to_numpy().astype(np.int64) << 32) | steps.event.to_numpy().astype(np.int64))[s]
    # one simulated SiPM per (event, channel): the event goes into the channel key
    key = ev * 4096 + c
    t_rel = t - steps.t0.to_numpy()[s]
    scfg = sipm.SiPMConfig()
    pmap = sipm.photon_map()
    if device:  # the fused photons -> SiPM path (response/device.py) on the same steps, against numpy photons + SiPM
        import jax as _jax
        from .. import chain as CH
        from ..response import device as DV, photons_jax
        cond = CH.Conditions.from_csv(calib, deviation)
        DT = {k: photons_jax.DeviceTable(v) for k, v in photons.load_tables(cfg).items()}
        a = time.perf_counter()
        rs = DV.response_batched(steps, pdg, geom, cfg, scfg, DT, cond.deviation, cond.status,
                                 ((steps.subrun.to_numpy().astype(np.int64) << 32) | steps.event.to_numpy().astype(np.int64)),
                                 np.random.default_rng(int(seed) + 7), _jax.random.PRNGKey(int(seed)), pmap)
        F = DV.to_host_batches(rs)
        F["chan"] = F["chan"]  # labels are event key * 4096 + channel, as `key` below
        t_fast = time.perf_counter() - a
        out_over = any(bool(r["overflow"]) for r in rs)
        keep = (cond.status[c] & 0b101) == 0
        s, c, t, key = s[keep], c[keep], t[keep], key[keep]
    elif jax:
        import jax as _jax
        from ..response import sipm_jax
        sipm_jax.simulate(key[:20000], t[:20000], s[:20000], pmap, scfg, np.random.default_rng(0), _jax.random.PRNGKey(0))
        a = time.perf_counter()
        F = sipm_jax.simulate(key, t, s, pmap, scfg, np.random.default_rng(int(seed) + 7), _jax.random.PRNGKey(int(seed)))
        t_fast = time.perf_counter() - a
    else:
        a = time.process_time()
        F = sipm.simulate(key, t, s, pmap, scfg, np.random.default_rng(int(seed) + 7))
        t_fast = time.process_time() - a
    a = time.process_time()
    if no_stock:
        S = sipm.simulate(key, t, s, pmap, scfg, np.random.default_rng(int(seed) + 8))
    else:
        S = stocksipm.run(key, t, s, scfg, stocksipm.photon_map_path(), seed=int(seed))
    t_stock = time.process_time() - a
    out = {"steps": len(steps), "photons": len(s), "charges_fast": len(F["pe"]), "charges_stock": len(S["pe"]),
           "fast_arm": "device" if device else ("jax" if jax else "numpy"),
           "device_overflow": out_over if device else None, "reference_arm": "numpy" if no_stock else "stock",
           "time_s_fast": t_fast, "cpu_s_reference": t_stock, "thresholds": THRESH, "groups": {}}
    grp_key = lambda k: SECTOR_GROUP[geom.sector[(k % 4096) // 4]]
    for g in GROUPS:
        mf, ms = grp_key(F["chan"]) == g, grp_key(S["chan"]) == g
        if not ms.any():
            continue
        kf, vf = np.unique(F["chan"][mf], return_inverse=True)
        ks, vs = np.unique(S["chan"][ms], return_inverse=True)
        pf = np.bincount(vf.ravel(), weights=F["pe"][mf])
        ps = np.bincount(vs.ravel(), weights=S["pe"][ms])
        allk = np.union1d(kf, ks)
        xf = np.zeros(len(allk)); xf[np.searchsorted(allk, kf)] = pf
        xs = np.zeros(len(allk)); xs[np.searchsorted(allk, ks)] = ps
        rel = xf.sum() / xs.sum() - 1
        err = np.sqrt(len(allk) * np.var(xf - xs)) / xs.sum()  # paired: same photons per SiPM in both arms
        tf = F["time"][mf] - steps.t0.to_numpy()[F["photon"][mf]]
        ts = S["time"][ms] - steps.t0.to_numpy()[S["photon"][ms]]
        shift = _cdf_shift(tf, ts)
        nph = int(np.isin(key, allk).sum())
        res = {"sipm_events": len(allk), "pe_per_sipm_fast": xf.mean(), "pe_per_sipm_stock": xs.mean(),
               "pe_per_sipm_rel": rel, "pe_per_sipm_rel_err": err, "time_cdf_shift_ns": shift,
               "charges_per_photon_fast": mf.sum() / max(nph, 1), "charges_per_photon_stock": ms.sum() / max(nph, 1),
               "frac_partial_fast": float((F["pe"][mf] < 0.99).mean()), "frac_partial_stock": float((S["pe"][ms] < 0.99).mean())}
        res["checks"] = {"pe_per_sipm": max(abs(rel) - 2 * err, 0) <= THRESH["pe_per_sipm_rel"],
                         "time_quantiles": bool(np.all(np.abs(shift) <= THRESH["time_quantile_ns"]))}
        res["pass"] = all(res["checks"].values())
        out["groups"][g] = res
    out["pass"] = all(r["pass"] for r in out["groups"].values())
    plots.write_json(outdir, "v4sipm_stats.json", out)
    print(f"steps {len(steps):,}; photons {len(s):,}; charges fast {len(F['pe']):,} stock {len(S['pe']):,}; "
          f"{out['fast_arm']} {t_fast:.2f} s, {out['reference_arm']} {t_stock:.1f} s")
    for g, r in out["groups"].items():
        flags = " ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in r["checks"].items())
        print(f"{g:3s} PE/SiPM/event {r['pe_per_sipm_stock']:.2f} -> {r['pe_per_sipm_fast']:.2f} "
              f"({r['pe_per_sipm_rel']:+.4f} +- {r['pe_per_sipm_rel_err']:.4f}); charges/photon "
              f"{r['charges_per_photon_stock']:.4f} -> {r['charges_per_photon_fast']:.4f}; partial "
              f"{r['frac_partial_stock']:.4f} -> {r['frac_partial_fast']:.4f}; t shift {np.round(r['time_cdf_shift_ns'], 3)} | {flags}")
    print(f"V4-sipm {'PASS' if out['pass'] else 'FAIL'}")
    return out
