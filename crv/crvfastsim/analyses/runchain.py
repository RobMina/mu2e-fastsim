"""runchain -- the whole fast CRV chain on CRY gen tables, no art: transport -> tier 1b -> photons -> SiPM charges ->
digis -> pulses -> clusters (crvfastsim/chain.py), with the cost of every stage. With --compare (CrvRecoTableDumper
files of the stock chain on the same shards) it reports clusters per generated event per sector type, fast vs stock.

    runchain gen.<shard>.root [...] --library lib.root --calib crvcalib.csv [--deviation crvphoton.csv] [--jax]
             [--reco-cpu] [--max-events N] [--compare ref.<shard>.root ...] [--emit shard.json] --outdir DIR
    runchain-merge shard*.json --outdir DIR      (node totals: events, clusters per sector type, rates)
"""
import json
import os
import time

import numpy as np

from mu2efastsim import plots, world as world_mod

from .. import chain, geometry, tables, tier1b
from .. import transport as tr
from ..response import photons as PH
from ..response import sipm as SI

DEFAULT_COUNTERS = "Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"
SECTOR_TYPES = {1: "EX", 2: "T1", 3: "T2", 4: "M"}


def run(inputs, outdir, library=None, calib=None, deviation=None, gdml=None, jax=False, max_events=None, seed=1,
        compare=None, counters=None, reco_cpu=False, emit=None, msc_tail=False, tail_k=None, sliced_msc=False,
        first_muon_only=False, no_absorbers=False, **kw):
    if not (library and calib):
        raise SystemExit("runchain needs --library and --calib")
    from .transport import default_gdml
    cpath = geometry.find_counters(counters or DEFAULT_COUNTERS)
    geom = geometry.from_offline_config(cpath)
    ccfg = geometry.parse_simpleconfig(open(cpath).read())
    cond = chain.Conditions.from_csv(calib, deviation)
    cfg = chain.ChainConfig(jax=bool(jax))
    T = {}
    t0 = time.perf_counter()
    world = [p for p in world_mod.load(gdml or default_gdml(), y_min=tr.Y_MSC_PLANE) if p[0] != "G4_POLYSTYRENE"]
    ab = None if no_absorbers else tr.absorber_slabs(gdml or default_gdml())
    lib = tier1b.read_library(library)
    tabs = PH.load_tables(cfg.response)
    if jax:
        from ..response import photons_jax
        tabs = {k: photons_jax.DeviceTable(v) for k, v in tabs.items()}
    pmap = SI.photon_map()
    T["setup_s"] = time.perf_counter() - t0
    out_cl, n_gen = [], 0
    for i, f in enumerate(tables.expand(inputs)):
        rng = np.random.default_rng([int(seed), i])
        gen = tables.read(f, "gen")
        if max_events:
            keep = gen.event.isin(np.sort(gen.event.unique())[:int(max_events)])
            gen = gen[keep].reset_index(drop=True)
        n_gen += int(gen[tables.KEY].drop_duplicates().shape[0])
        a = time.perf_counter()
        st, pa = tr.transport(gen, geom, rng, stage="C", world=world, msc_tail=msc_tail, tail_k=tail_k,
                              sliced_msc=sliced_msc, all_muons=not first_muon_only, absorbers=ab)
        T["transport_s"] = T.get("transport_s", 0.0) + time.perf_counter() - a
        a = time.perf_counter()
        st, pa, _ = tier1b.sample(st, pa, gen, lib, geom, rng)
        T["tier1b_s"] = T.get("tier1b_s", 0.0) + time.perf_counter() - a
        T["steps"] = T.get("steps", 0) + len(st)
        T["events_with_steps"] = T.get("events_with_steps", 0) + int(st[tables.KEY].drop_duplicates().shape[0])
        Q = chain.charges(st, pa, geom, cond, cfg, rng, tabs=tabs, pmap=pmap, timers=T)
        D = chain.digis(Q, cond, cfg, rng, timers=T)
        W = chain.reco_waveforms(D, cond)
        a = time.perf_counter()
        if reco_cpu and jax:   # response on the accelerator, reco with the JAX CPU backend (oneAPI: reco kernels slow)
            import jax as _jax
            with _jax.default_device(_jax.devices("cpu")[0]):
                outs, _ = chain.clusters(W, cond, geom, ccfg, cfg, timers=T)
        else:
            outs, _ = chain.clusters(W, cond, geom, ccfg, cfg, timers=T)
        T["reco_s"] = T.get("reco_s", 0.0) + time.perf_counter() - a
        out_cl += outs
    cl_type = np.concatenate([o["cl_type"] for o in out_cl]) if out_cl else np.zeros(0, int)
    res = {"generated_events": n_gen, "timers": T,
           "clusters_per_generated": {SECTOR_TYPES.get(int(k), str(k)): float((cl_type == k).sum() / n_gen)
                                      for k in np.unique(cl_type)}}
    nev = max(T.get("events_with_steps", 1), 1)
    res["ms_per_event_with_steps"] = {k[:-2]: 1e3 * v / nev for k, v in T.items() if k.endswith("_s") and k != "setup_s"}
    if compare:
        from ..reco import reftables
        c = reftables.read(tables.expand(compare), "clusters", ["sectortype"])
        import awkward as ak
        st_ref = ak.to_numpy(c.sectortype)
        res["compare_generated_events"] = n_gen
        res["clusters_per_generated_ref"] = {SECTOR_TYPES.get(int(k), str(k)): float((st_ref == k).sum() / n_gen)
                                             for k in np.unique(st_ref)}
    res["wall_s"] = time.perf_counter() - t0
    if jax:
        import jax as _jax
        res["device"] = str(_jax.devices()[0])
        res["reco_device"] = "cpu" if reco_cpu else res["device"]
    if emit:   # per-shard accumulators for a merge (node jobs: one process per tile)
        with open(emit, "w") as fh:
            json.dump({"generated_events": n_gen, "timers": T, "wall_s": res["wall_s"],
                       "cl_type_counts": {int(k): int((cl_type == k).sum()) for k in np.unique(cl_type)},
                       "device": res.get("device"), "reco_device": res.get("reco_device"),
                       "inputs": list(tables.expand(inputs))}, fh)
    plots.write_json(outdir, "runchain_stats.json", res)
    print(json.dumps({k: res[k] for k in res if k != "timers"}, indent=1))
    print("timers: " + ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in T.items()))
    return res


def merge(inputs, outdir, wall=None, **kw):
    """Combine runchain --emit files of one node job (one process per accelerator tile): totals, clusters per
    generated event per sector type, accepted events per node-hour from the job wall time (--wall, seconds) or, if not
    given, from the slowest shard."""
    J = [json.load(open(f)) for f in tables.expand(inputs)]
    n_gen = sum(j["generated_events"] for j in J)
    acc = sum(j["timers"].get("events_with_steps", 0) for j in J)
    cl = {}
    for j in J:
        for k, v in j["cl_type_counts"].items():
            cl[int(k)] = cl.get(int(k), 0) + v
    T = {}
    for j in J:
        for k, v in j["timers"].items():
            if isinstance(v, (int, float)):
                T[k] = T.get(k, 0) + v
    slowest = max(j["wall_s"] for j in J)
    wall_s = float(wall) if wall else slowest
    res = {"shards": len(J), "generated_events": n_gen, "events_with_steps": acc,
           "clusters_per_generated": {SECTOR_TYPES.get(k, str(k)): v / max(n_gen, 1) for k, v in sorted(cl.items())},
           "wall_s": wall_s, "slowest_shard_wall_s": slowest,
           "accepted_events_per_node_hour": 3600.0 * acc / wall_s, "generated_events_per_node_hour": 3600.0 * n_gen / wall_s,
           "stage_s_summed_over_shards": {k: v for k, v in T.items() if k.endswith("_s")},
           "ms_per_event_with_steps_per_process": {k[:-2]: 1e3 * v / max(acc, 1) for k, v in T.items()
                                                   if k.endswith("_s") and k != "setup_s"},
           "devices": sorted({str(j.get("device")) for j in J}), "reco_devices": sorted({str(j.get("reco_device")) for j in J})}
    plots.write_json(outdir, "runchain_merge_stats.json", res)
    print(json.dumps(res, indent=1))
    return res
