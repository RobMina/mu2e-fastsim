"""Throughput of the photon substep kernel (numpy port and JAX twin) on synthetic tables and rows: no Mu2e-internal input.

The synthetic table has the shape of a 3200 mm v6 table (10 x 41 x 56 position bins, 5 beta bins, 150 delay bins) with
random weights and a uniform arrival probability (--prob); rows are muon-like bar crossings: 20 mm long through the
thickness at random width and length, photons per step = yield x edep (public Offline values by default). Photon counts,
and so the cost, scale with prob x yield x edep. Results per (step, simulated SiPM) row; a muon crossing in a
double-ended bar is 4 rows.
"""
import json
import os
import time

import numpy as np

from . import lookup, photons


def synthetic_table(rng, prob=0.0015, D=150, E=8):
    edges = {"x": np.linspace(0, 9.89, 11), "y": np.linspace(-25.67, 25.67, 42), "z": np.linspace(-1600, 1600, 57),
             "beta": np.linspace(0.455, 1.0, 6), "theta": np.linspace(0, np.pi, 12), "phi": np.linspace(0, np.pi, 6),
             "r": np.linspace(0, 0.616, 6)}
    nb = {k: len(v) - 1 for k, v in edges.items()}

    def tab(n):
        d = rng.integers(0, 4, (n, D)).astype(np.uint8)
        e = rng.integers(0, 40, (n, E)).astype(np.uint8)
        return lookup.Table(np.full(n, prob, np.float32), d, d.sum(1, dtype=np.int32), e, e.sum(1, dtype=np.int32))

    ns = nb["x"] * nb["y"] * nb["z"]
    T = {"scint": tab(ns), "ckscint": tab(ns * nb["beta"]),
         "ckfiber": tab(nb["beta"] * nb["theta"] * nb["phi"] * nb["r"] * nb["z"])}
    ck = np.array([[0.474, 0.0], [1.0, 266.0]])
    return lookup.LookupTable("LookupTable_synth", (6, 0), 0, 9.64, 25.42, 1600.0, 26.0, 2.0, 1.0, 0.7, 1.75, 0.126,
                              0.0, ck, ck, edges, T)


def synthetic_rows(rng, n, yield_per_mev=28610.0, edep=3.4, length=20.0):
    a = np.c_[np.full(n, -length / 2), rng.uniform(-24, 24, n), rng.uniform(-1500, 1500, n)]
    d = np.c_[np.full(n, length), rng.normal(0, 2, n), rng.normal(0, 2, n)]
    nsub = np.maximum((np.linalg.norm(d, axis=1) / photons.PRECISION_MM).astype(np.int64), 1)
    t0 = rng.uniform(0, 1e5, n)
    beta = np.full(n, 0.999)
    ck = np.array([[0.474, 0.0], [1.0, 266.0]])
    return {"s": a, "d": d, "t0": t0, "t1": t0 + 0.07, "nsub": nsub, "beta": beta,
            "theta": np.arccos(np.clip(d[:, 2] / np.linalg.norm(d, axis=1), -1, 1)), "dev": np.full(n, 0.94),
            "step": np.arange(n), "chan": np.arange(n) * 4,
            "n_scint": photons._per_substep(np.full(n, yield_per_mev * edep), nsub, rng),
            "n_ck_s": photons._per_substep(photons.cerenkov_per_mm(beta, np.ones(n), ck) * length, nsub, rng),
            "n_ck_f": photons._per_substep(photons.cerenkov_per_mm(beta, np.ones(n), ck) * length, nsub, rng)}


def run_sipm(n_sipms=200_000, mean_pe=130.0, repeat=3, seed=1, jax=True, numpy_sipms=20_000, photon_map_path=None):
    """SiPM-charge stage on synthetic photons: n_sipms SiPMs with Poisson(mean_pe / 0.407) photons in a 5 ns spread,
    a synthetic photon map (random, peaked). ~130 photons per (step, SiPM) row as in the photon bench."""
    from . import sipm
    rng = np.random.default_rng(seed)
    cfg = sipm.SiPMConfig()
    pmap = sipm.photon_map(photon_map_path) if photon_map_path else rng.random((40, 40)) ** 4
    n = rng.poisson(mean_pe, n_sipms)
    chan = np.repeat(np.arange(n_sipms), n)
    t = np.repeat(rng.uniform(0, 1e5, n_sipms), n) + rng.exponential(5.0, len(chan))
    out = {"sipms": n_sipms, "photons": int(len(chan)), "photon_map": photon_map_path or "synthetic"}
    if numpy_sipms:
        m = chan < numpy_sipms
        a = time.process_time()
        F = sipm.simulate(chan[m], t[m], np.arange(m.sum()), pmap, cfg, np.random.default_rng(seed))
        dt = time.process_time() - a
        out.update({"numpy_sipms_per_s": numpy_sipms / dt, "numpy_charges_per_photon": len(F["pe"]) / m.sum()})
    if jax:
        import jax as _jax
        from . import sipm_jax
        key = _jax.random.PRNGKey(seed)
        a = time.perf_counter()
        sipm_jax.simulate(chan, t, np.arange(len(chan)), pmap, cfg, np.random.default_rng(seed), key)
        out["jax_first_call_s"] = time.perf_counter() - a
        best = 1e9
        for r in range(repeat):
            a = time.perf_counter()
            J = sipm_jax.simulate(chan, t, np.arange(len(chan)), pmap, cfg, np.random.default_rng(seed + r + 1),
                                  _jax.random.fold_in(key, r + 1))
            best = min(best, time.perf_counter() - a)
        out.update({"device": str(_jax.devices()[0]), "jax_s": best, "jax_sipms_per_s": n_sipms / best,
                    "jax_charges_per_photon": len(J["pe"]) / len(chan)})
        if numpy_sipms:
            out["speedup_vs_numpy_core"] = out["jax_sipms_per_s"] / out["numpy_sipms_per_s"]
    print(json.dumps(out))
    return out


def run(n_rows=200_000, prob=0.0015, repeat=3, seed=1, jax=True, numpy_rows=20_000, outdir=None, table=None):
    """table: a real lookup-table name (e.g. '3200_0', read from $CRVFS_LOOKUP_DIR) instead of the synthetic one."""
    rng = np.random.default_rng(seed)
    L = lookup.load(table) if table else synthetic_table(rng, prob)
    rows = synthetic_rows(rng, n_rows)
    out = {"rows": n_rows, "table": table or f"synthetic p={prob}", "substeps": int(rows["nsub"].sum())}
    if numpy_rows:
        sub = {k: v[:numpy_rows] for k, v in rows.items()}
        a = time.process_time()
        s, _, _ = photons._substep_photons(L, sub, np.random.default_rng(seed))
        dt = time.process_time() - a
        out.update({"numpy_rows": numpy_rows, "numpy_cpu_s": dt, "numpy_rows_per_s": numpy_rows / dt,
                    "photons_per_row": len(s) / numpy_rows})
    if jax:
        import jax as _jax
        from . import photons_jax
        dt_ = photons_jax.DeviceTable(L)
        key = _jax.random.PRNGKey(seed)
        a = time.perf_counter()
        photons_jax.substep_photons(dt_, rows, key)
        out["jax_first_call_s"] = time.perf_counter() - a
        best = 1e9
        for r in range(repeat):
            a = time.perf_counter()
            s, _, t = photons_jax.substep_photons(dt_, rows, _jax.random.fold_in(key, r + 1))
            best = min(best, time.perf_counter() - a)
        out.update({"device": str(_jax.devices()[0]), "jax_s": best, "jax_rows_per_s": n_rows / best,
                    "jax_photons_per_row": len(t) / n_rows})
        if numpy_rows:
            out["speedup_vs_numpy_core"] = out["jax_rows_per_s"] / out["numpy_rows_per_s"]
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "photon_bench.json"), "w") as fh:
            json.dump(out, fh, indent=2)
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=200_000)
    p.add_argument("--prob", type=float, default=0.0015)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--numpy-rows", type=int, default=20_000)
    p.add_argument("--no-jax", action="store_true")
    p.add_argument("--outdir", default=None)
    p.add_argument("--stage", default="photons", choices=["photons", "sipm"])
    p.add_argument("--sipms", type=int, default=200_000)
    p.add_argument("--table", default=None, help="real lookup table name (from $CRVFS_LOOKUP_DIR), default synthetic")
    p.add_argument("--photon-map", default=None, help="real photonMap.root for --stage sipm, default synthetic")
    p.add_argument("--numpy-sipms", type=int, default=20_000)
    a = p.parse_args()
    if a.stage == "sipm":
        r = run_sipm(a.sipms, repeat=a.repeat, seed=a.seed, jax=not a.no_jax, numpy_sipms=a.numpy_sipms,
                     photon_map_path=a.photon_map)
        if a.outdir:
            os.makedirs(a.outdir, exist_ok=True)
            json.dump(r, open(os.path.join(a.outdir, "sipm_bench.json"), "w"), indent=2)
    else:
        run(a.rows, a.prob, a.repeat, a.seed, not a.no_jax, a.numpy_rows, a.outdir, a.table)
