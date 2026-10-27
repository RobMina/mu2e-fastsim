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

import copy
import dataclasses
from pathlib import Path
from types import MappingProxyType

from .. import chain, geometry, runconfig, tables, tier1b
from .. import transport as tr
from ..response import photons as PH
from ..response import sipm as SI

DEFAULT_COUNTERS = "Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"
SECTOR_TYPES = {1: "EX", 2: "T1", 3: "T2", 4: "M"}


def _resolve(library, calib, deviation, gdml, jax, counters, run_config, window, fused_response=False):
    """Resolve setup arguments without loading large assets or drawing random numbers."""
    if not (library and calib):
        raise SystemExit("runchain needs --library and --calib")
    from .transport import default_gdml
    rc = runconfig.get(run_config, window if window or not run_config else "offspill")
    if counters:
        rc = dataclasses.replace(rc, counters=counters)
    cfg = rc.chain_config(jax=bool(jax)) if rc.name != runconfig.DEFAULT or window else chain.ChainConfig(jax=bool(jax))
    if fused_response:
        cfg = dataclasses.replace(cfg, fused_response=True)
    gdml_path = gdml or (rc.gdml_path() if rc.name != runconfig.DEFAULT else default_gdml())
    return rc, cfg, gdml_path


def _signature(rc, cfg, gdml, library, calib, deviation, no_absorbers, unbanded_world, no_below_world,
               below_margin=None):
    canonical = lambda p: str(Path(p).resolve()) if p and p != "nominal" else p
    return {"run_config": dataclasses.asdict(rc), "chain_config": dataclasses.asdict(cfg),
            "paths": [canonical(p) for p in (gdml, library, calib, deviation)],
            "no_absorbers": bool(no_absorbers), "unbanded_world": bool(unbanded_world),
            "no_below_world": bool(no_below_world),
            "below_margin": None if below_margin is None else float(below_margin),
            "asset_environment": {k: os.environ.get(k) for k in
                                  ("CRVFS_DATA", "CRVFS_LOOKUP_DIR", "CRVFS_OFFLINE_CACHE", "MU2E_SEARCH_PATH")}}


@dataclasses.dataclass(frozen=True)
class PreparedContext:
    """Single-owner setup snapshot; no RNG, inputs, event outputs or executor are retained."""
    signature: dict
    payload: dict
    propagation: dict
    template: object
    template_args: tuple
    fingerprint: str
    device: object
    preparation_timers: dict

    def audit(self):
        from .context_audit import content_fingerprint
        if content_fingerprint((self.signature, dict(self.payload), self.propagation, self.template, self.template_args)) != self.fingerprint:
            raise ValueError("prepared context mutated in memory")
        if PH._TABLE_PROP != self.propagation:
            raise ValueError("prepared optical propagation cache changed")
        from ..reco import waveform as WF
        if WF.single_pe_template(*self.template_args) is not self.template:
            raise ValueError("prepared waveform template cache changed")
        if self.device is not None:
            import jax.numpy as jnp
            if jnp.zeros(1).block_until_ready().devices() != {self.device}:
                raise ValueError("prepared context device changed")


def prepare(library=None, calib=None, deviation=None, gdml=None, jax=False, counters=None, no_absorbers=False,
            run_config=None, window=None, benchmark=None, fused_response=False, unbanded_world=False,
            no_below_world=False, below_margin=None):
    """Load and seal reusable setup; callers audit before/after each sequential invocation."""
    if fused_response:
        raise ValueError("prepared context does not support memory-dependent fused response batching")
    from .context_audit import freeze_arrays, content_fingerprint
    from ..reco import waveform as WF
    rc, cfg, gdml_path = _resolve(library, calib, deviation, gdml, jax, counters, run_config, window, fused_response)
    rc, cfg = copy.deepcopy((rc, cfg))
    sig = _signature(rc, cfg, gdml_path, library, calib, deviation, no_absorbers, unbanded_world, no_below_world,
                     below_margin)
    if benchmark is not None:
        benchmark.setup(rc, cfg, gdml_path, library, calib, deviation)
    payload, timers = _load_setup(rc, cfg, gdml_path, gdml, library, calib, deviation, jax,
                                  no_absorbers, unbanded_world, no_below_world, below_margin)
    a = time.perf_counter()
    if hasattr(payload["world"], "pieces"):
        payload["world"].pieces()
    PH._TABLE_PROP.clear()
    for table in payload["tabs"].values():
        PH.table_propagation(table.L if jax else table)
    propagation = copy.deepcopy(PH._TABLE_PROP)
    wcfg = cfg.waveform or WF.WaveformConfig()
    template_args = (wcfg.single_pe_file, wcfg.single_pe_precision, wcfg.single_pe_stretch, wcfg.single_pe_max_time)
    WF.single_pe_template.cache_clear()
    template = WF.single_pe_template(*template_args)
    device = None
    if jax:
        import jax.numpy as jnp
        device = next(iter(jnp.zeros(1).block_until_ready().devices()))
        for table in payload["tabs"].values():
            for array in (table.cum, table.prob, table.dsum, *table.edges.values()):
                array.block_until_ready()
                if array.devices() != {device}:
                    raise ValueError("prepared optical table on a different device")
    state = (sig, payload, propagation, template, template_args)
    freeze_arrays(state)
    fingerprint = content_fingerprint(state)
    timers["seal_and_cache_s"] = time.perf_counter() - a
    return PreparedContext(sig, MappingProxyType(payload), propagation, template, template_args, fingerprint, device, timers)


def _load_setup(rc, cfg, gdml_path, explicit_gdml, library, calib, deviation, jax,
                no_absorbers, unbanded_world, no_below_world, below_margin=None):
    geometry_start = time.perf_counter()
    geom = rc.geometry()
    ccfg = rc.counters_cfg()
    cond = runconfig.nominal_conditions(geom) if calib == "nominal" else chain.Conditions.from_csv(calib, deviation)
    if len(cond.status) < 4 * geom.nbars:
        raise SystemExit(f"runchain: conditions cover {len(cond.status)} channels, {rc.name} has {4 * geom.nbars}; "
                         "pass the configuration's CRVCalib CSV or --calib nominal")
    tagger = rc.tagger_bar(geom) if rc.name != runconfig.DEFAULT else None
    bar_cut = rc.bar_cut(geom)
    timers = {"geometry_conditions_config_s": time.perf_counter() - geometry_start}
    a = time.perf_counter()
    world, ab = tr.load_world(gdml_path, banded=not unbanded_world, y_plane=tr.msc_plane(geom))
    ab = None if no_absorbers else ab
    below = None
    if rc.below_world and not no_below_world:
        from .transport import _below_kw
        below, ab = tr.load_below(explicit_gdml or rc.gdml_path(), geom,
                                  **_below_kw(rc, {"below_margin": below_margin})), None
    lib = tier1b.read_library(library)
    tabs = PH.load_tables(cfg.response)
    if jax:
        from ..response import photons_jax
        tabs = {k: photons_jax.DeviceTable(v) for k, v in tabs.items()}
    pmap = SI.photon_map()
    timers["setup_s"] = time.perf_counter() - a
    stype_name = runconfig.sector_type_names(geom, ccfg) if rc.name != runconfig.DEFAULT else SECTOR_TYPES
    return dict(rc=rc, cfg=cfg, geom=geom, ccfg=ccfg, cond=cond, tagger=tagger, bar_cut=bar_cut,
                world=world, ab=ab, below=below, lib=lib, tabs=tabs, pmap=pmap, stype_name=stype_name), timers


def run(inputs, outdir, library=None, calib=None, deviation=None, gdml=None, jax=False, max_events=None, seed=1,
        compare=None, counters=None, reco_cpu=False, emit=None, msc_tail=False, tail_k=None, sliced_msc=False,
        first_muon_only=False, no_absorbers=False, run_config=None, window=None, benchmark=None, prepared=None, **kw):
    bench_start = time.perf_counter() if benchmark is not None else None
    rc, cfg, gdml_path = _resolve(library, calib, deviation, gdml, jax, counters, run_config, window,
                                  kw.get("fused_response", False))
    if benchmark is not None:
        benchmark.setup(rc, cfg, gdml_path, library, calib, deviation)
    t0 = time.perf_counter()
    if prepared is None:
        payload, T = _load_setup(rc, cfg, gdml_path, gdml, library, calib, deviation, jax, no_absorbers,
                                kw.get("unbanded_world", False), kw.get("no_below_world", False),
                                kw.get("below_margin"))
        if benchmark is None:
            T.pop("geometry_conditions_config_s")
    else:
        signature = _signature(rc, cfg, gdml_path, library, calib, deviation, no_absorbers,
                               kw.get("unbanded_world", False), kw.get("no_below_world", False),
                               kw.get("below_margin"))
        if signature != prepared.signature:
            raise ValueError("prepared context does not match requested configuration/assets")
        a = time.perf_counter()
        prepared.audit()
        payload = prepared.payload
        T = {"setup_s": 0.0, "geometry_conditions_config_s": 0.0, "context_reused": True,
             "context_audit_s": time.perf_counter() - a}
    rc, cfg, geom, ccfg, cond = (payload[k] for k in ("rc", "cfg", "geom", "ccfg", "cond"))
    tagger, bar_cut, world, ab, below = (payload[k] for k in ("tagger", "bar_cut", "world", "ab", "below"))
    lib, tabs, pmap, stype_name = (payload[k] for k in ("lib", "tabs", "pmap", "stype_name"))
    out_cl, n_gen = [], 0

    def host_stage(i, f):
        """CPU part of file i (read, transport, tier 1b) with its own rng and timers: the same draws whether it runs
        inline or in the prefetch thread."""
        rng = np.random.default_rng([int(seed), i])
        t = {}
        a = time.perf_counter() if benchmark is not None else None
        gen = tables.read(f, "gen")
        if benchmark is not None:
            t["input_io_s"] = time.perf_counter() - a
            benchmark.check_input(i, f, gen)
        if max_events:
            keep = gen.event.isin(np.sort(gen.event.unique())[:int(max_events)])
            gen = gen[keep].reset_index(drop=True)
        keys = gen[tables.KEY].drop_duplicates().to_numpy() if benchmark is not None else None
        ngen = int(gen[tables.KEY].drop_duplicates().shape[0])
        a = time.perf_counter()
        st, pa = tr.transport(gen, geom, rng, stage="C", world=world, msc_tail=msc_tail, tail_k=tail_k,
                              sliced_msc=sliced_msc, all_muons=not first_muon_only, absorbers=ab, tagger_bar=tagger,
                              bar_cut=bar_cut, below=below,
                              below_msc=below is not None and (rc.below_msc or bool(kw.get("below_msc"))))
        t["transport_s"] = time.perf_counter() - a
        a = time.perf_counter()
        st, pa, _ = tier1b.sample(st, pa, gen, lib, geom, rng, showers=rc.showers)
        t["tier1b_s"] = time.perf_counter() - a
        return ngen, st, pa, rng, t, keys

    files = list(tables.expand(inputs))
    pool = None
    if kw.get("prefetch") and len(files) > 1:   # file i+1's host stage runs while the device works on file i
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=1)
        nxt = pool.submit(host_stage, 0, files[0])
    try:
        for i, f in enumerate(files):
            if pool is not None:
                a = time.perf_counter()
                ngen, st, pa, rng, t, keys = nxt.result()
                T["host_wait_s"] = T.get("host_wait_s", 0.0) + time.perf_counter() - a
                if i + 1 < len(files):
                    nxt = pool.submit(host_stage, i + 1, files[i + 1])
            else:
                ngen, st, pa, rng, t, keys = host_stage(i, f)
            n_gen += ngen
            for k_, v_ in t.items():
                T[k_] = T.get(k_, 0.0) + v_
            T["steps"] = T.get("steps", 0) + len(st)
            T["events_with_steps"] = T.get("events_with_steps", 0) + int(st[tables.KEY].drop_duplicates().shape[0])
            file_timers = {} if benchmark is not None else T
            Q = chain.charges(st, pa, geom, cond, cfg, rng, tabs=tabs, pmap=pmap, timers=file_timers)
            D = chain.digis(Q, cond, cfg, rng, timers=file_timers)
            a = time.perf_counter() if benchmark is not None else None
            W = chain.reco_waveforms(D, cond)
            if benchmark is not None:
                file_timers["waveform_pack_s"] = time.perf_counter() - a
            a = time.perf_counter()
            detail_kw = {"detail": True} if benchmark is not None else {}
            if reco_cpu and jax:   # response on the accelerator, reco with the JAX CPU backend (oneAPI: reco kernels slow)
                import jax as _jax
                with _jax.default_device(_jax.devices("cpu")[0]):
                    outs, _ = chain.clusters(W, cond, geom, ccfg, cfg, timers=file_timers, **detail_kw)
            else:
                outs, _ = chain.clusters(W, cond, geom, ccfg, cfg, timers=file_timers, **detail_kw)
            file_timers["reco_s"] = file_timers.get("reco_s", 0.0) + time.perf_counter() - a
            if benchmark is not None:
                for k_, v_ in file_timers.items():
                    T[k_] = T.get(k_, 0) + v_
                a = time.perf_counter()
                benchmark.output(i, f, keys, outs, st=st, pa=pa, charges=Q, digis=D, waveforms=W,
                                 timers={**t, **file_timers})
                T["detail_output_s"] = T.get("detail_output_s", 0.0) + time.perf_counter() - a
                out_cl += [{"cl_type": o["cl_type"]} for o in outs]
            else:
                out_cl += outs
    finally:
        if pool is not None:
            pool.shutdown()
    if prepared is not None:
        a = time.perf_counter()
        prepared.audit()
        T["context_audit_s"] += time.perf_counter() - a
    cl_type = np.concatenate([o["cl_type"] for o in out_cl]) if out_cl else np.zeros(0, int)
    res = {"generated_events": n_gen, "timers": T,
           "run_config": rc.name, "window": list(cfg.window),
           "clusters_per_generated": {stype_name.get(int(k), str(k)): float((cl_type == k).sum() / n_gen)
                                      for k in np.unique(cl_type)}}
    nev = max(T.get("events_with_steps", 1), 1)
    res["ms_per_event_with_steps"] = {k[:-2]: 1e3 * v / nev for k, v in T.items() if k.endswith("_s") and k != "setup_s"}
    if compare:
        from ..reco import reftables
        c = reftables.read(tables.expand(compare), "clusters", ["sectortype"])
        import awkward as ak
        st_ref = ak.to_numpy(c.sectortype)
        res["compare_generated_events"] = n_gen
        res["clusters_per_generated_ref"] = {stype_name.get(int(k), str(k)): float((st_ref == k).sum() / n_gen)
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
                       "run_config": rc.name, "type_names": {int(k): v for k, v in stype_name.items()},
                       "device": res.get("device"), "reco_device": res.get("reco_device"),
                       "inputs": list(tables.expand(inputs))}, fh)
    if benchmark is not None:
        res["sector_type_names"] = {str(k): v for k, v in stype_name.items()}
        res["instrumented_wall_before_stats_s"] = time.perf_counter() - bench_start
    a = time.perf_counter() if benchmark is not None else None
    plots.write_json(outdir, "runchain_stats.json", res)
    if benchmark is not None:
        T["stats_output_s"] = time.perf_counter() - a
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
    names = {**SECTOR_TYPES, **{int(k): v for j in J for k, v in j.get("type_names", {}).items()}}
    slowest = max(j["wall_s"] for j in J)
    wall_s = float(wall) if wall else slowest
    res = {"shards": len(J), "generated_events": n_gen, "events_with_steps": acc,
           "clusters_per_generated": {names.get(k, str(k)): v / max(n_gen, 1) for k, v in sorted(cl.items())},
           "wall_s": wall_s, "slowest_shard_wall_s": slowest,
           "accepted_events_per_node_hour": 3600.0 * acc / wall_s, "generated_events_per_node_hour": 3600.0 * n_gen / wall_s,
           "stage_s_summed_over_shards": {k: v for k, v in T.items() if k.endswith("_s")},
           "ms_per_event_with_steps_per_process": {k[:-2]: 1e3 * v / max(acc, 1) for k, v in T.items()
                                                   if k.endswith("_s") and k != "setup_s"},
           "devices": sorted({str(j.get("device")) for j in J}), "reco_devices": sorted({str(j.get("reco_device")) for j in J})}
    plots.write_json(outdir, "runchain_merge_stats.json", res)
    print(json.dumps(res, indent=1))
    return res
