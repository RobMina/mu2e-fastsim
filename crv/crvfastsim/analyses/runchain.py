"""runchain -- the whole fast CRV chain on CRY gen tables, no art: transport -> tier 1b -> photons -> SiPM charges ->
digis -> pulses -> clusters (crvfastsim/chain.py), with the cost of every stage. With --compare (CrvRecoTableDumper
files of the stock chain on the same shards) it reports clusters per generated event per sector type, fast vs stock.

    runchain gen.<shard>.root [...] --calib crvcalib.csv [--deviation crvphoton.csv] [--jax]
             [--tier1b strict_v2|legacy [--tier1b-dir DIR] | --library lib.root]
             [--reco-cpu] [--max-events N] [--compare ref.<shard>.root ...] [--emit shard.json] --outdir DIR
    runchain-merge shard*.json --outdir DIR      (node totals: events, clusters per sector type, rates)

Tier 1b (crvfastsim.tier1b_assets): default asset set strict_v2 for extracted_v04 (strict library, crossing context,
containment, point and topology placement, templates); --tier1b legacy is the tier1b_v7 library without placement (the
default before 2026-10-28) and the only set of the named run configurations; --library is a custom library sampled
without placement options.

Named conditions manifests are supported only by ordinary, non-prepared runs. They are mutually exclusive
with calib/deviation. Prepared contexts and benchmark callbacks reject manifests before any setup: their
existing CSV signatures, sealing and benchmark asset inventories do not cover manifest bundles.
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

from .. import chain, geometry, readout, runconfig, tables, tier1b, tier1b_assets
from ..timewindow import resolve_runchain_window, validate_runchain_merge
from .. import transport as tr
from ..response import photons as PH
from ..response import sipm as SI

DEFAULT_COUNTERS = "Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"
SECTOR_TYPES = {1: "EX", 2: "T1", 3: "T2", 4: "M"}


def _resolve(library, calib, deviation, gdml, jax, counters, run_config, window, fused_response=False,
             conditions_manifest=None, tier1b_set=None, tier1b_dir=None):
    """Resolve setup arguments without loading large assets or drawing random numbers."""
    if conditions_manifest is not None and (calib is not None or deviation is not None):
        raise SystemExit("--conditions-manifest cannot be combined with --calib or --deviation")
    if not (calib or conditions_manifest):
        raise SystemExit("runchain needs either --calib or --conditions-manifest")
    from .transport import default_gdml
    rc, time_window = resolve_runchain_window(run_config, window)
    if counters:
        rc = dataclasses.replace(rc, counters=counters)
    cfg = rc.chain_config(jax=bool(jax)) if rc.name != runconfig.DEFAULT or window else chain.ChainConfig(jax=bool(jax))
    if fused_response:
        cfg = dataclasses.replace(cfg, fused_response=True)
    gdml_path = gdml or (rc.gdml_path() if rc.name != runconfig.DEFAULT else default_gdml())
    try:
        spec = tier1b_assets.resolve(tier1b_set, library, tier1b_dir, rc.name)
    except ValueError as e:
        raise SystemExit(f"runchain: {e}")
    return rc, cfg, gdml_path, time_window, spec


def _signature(rc, cfg, gdml, spec, calib, deviation, no_absorbers, unbanded_world, no_below_world,
               below_margin=None):
    canonical = lambda p: str(Path(p).resolve()) if p and p != "nominal" else p
    return {"run_config": dataclasses.asdict(rc), "chain_config": dataclasses.asdict(cfg),
            "paths": [canonical(p) for p in (gdml, *spec.paths(), calib, deviation)],
            "tier1b": {"set": spec.name, "options": dict(spec.options), "library_sha256": spec.library_sha256},
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
            no_below_world=False, below_margin=None, conditions_manifest=None, geometry_readout=False,
            tier1b_set=None, tier1b_dir=None):
    """Load and seal reusable CSV setup; named manifests and physical masks are not supported by reuse/benchmark."""
    if geometry_readout:
        raise ValueError("geometry readout is not supported with prepared contexts or benchmarks")
    if conditions_manifest is not None:
        raise ValueError("conditions manifests are not supported with prepared contexts or benchmarks")
    if fused_response:
        raise ValueError("prepared context does not support memory-dependent fused response batching")
    from .context_audit import freeze_arrays, content_fingerprint
    from ..reco import waveform as WF
    rc, cfg, gdml_path, _, spec = _resolve(library, calib, deviation, gdml, jax, counters, run_config, window,
                                           fused_response, tier1b_set=tier1b_set, tier1b_dir=tier1b_dir)
    rc, cfg = copy.deepcopy((rc, cfg))
    sig = _signature(rc, cfg, gdml_path, spec, calib, deviation, no_absorbers, unbanded_world, no_below_world,
                     below_margin)
    if benchmark is not None:
        benchmark.setup(rc, cfg, gdml_path, spec, calib, deviation)
    payload, timers = _load_setup(rc, cfg, gdml_path, gdml, spec, calib, deviation, jax,
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


def _load_setup(rc, cfg, gdml_path, explicit_gdml, spec, calib, deviation, jax,
                no_absorbers, unbanded_world, no_below_world, below_margin=None,
                conditions_manifest=None, configured_rc=None):
    geometry_start = time.perf_counter()
    geom = rc.geometry()
    ccfg = rc.counters_cfg()
    conditions_metadata = {"status": "unverified_legacy", "calib": calib, "deviation": deviation}
    if conditions_manifest is not None:
        from ..named_conditions import load_conditions
        cond, conditions_metadata = load_conditions(conditions_manifest, configured_rc, geom)
    else:
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
    lib = tier1b_assets.load(spec, geom)   # inside setup_s (no own timer: every *_s key counts as a stage)
    tabs = PH.load_tables(cfg.response)
    if jax:
        from ..response import photons_jax
        tabs = {k: photons_jax.DeviceTable(v) for k, v in tabs.items()}
    pmap = SI.photon_map()
    timers["setup_s"] = time.perf_counter() - a
    stype_name = runconfig.sector_type_names(geom, ccfg) if rc.name != runconfig.DEFAULT else SECTOR_TYPES
    return dict(rc=rc, cfg=cfg, geom=geom, ccfg=ccfg, cond=cond, tagger=tagger, bar_cut=bar_cut,
                world=world, ab=ab, below=below, lib=lib, tabs=tabs, pmap=pmap, stype_name=stype_name,
                conditions_metadata=conditions_metadata), timers


def run(inputs, outdir, library=None, calib=None, deviation=None, gdml=None, jax=False, max_events=None, seed=1,
        compare=None, counters=None, reco_cpu=False, emit=None, msc_tail=False, tail_k=None, sliced_msc=False,
        first_muon_only=False, no_absorbers=False, run_config=None, window=None, benchmark=None, prepared=None,
        conditions_manifest=None, geometry_readout=False, tier1b_set=None, tier1b_dir=None, **kw):
    if geometry_readout:
        if prepared is not None or benchmark is not None:
            raise ValueError("geometry readout is not supported with prepared contexts or benchmarks")
        name = run_config if isinstance(run_config, str) else getattr(run_config, "name", None)
        if name not in ("run1a_v01", "run2_v10") or conditions_manifest is None or counters is not None:
            raise ValueError("geometry readout requires Run1A/Run2 --run-config and --conditions-manifest; no counters override")
    if conditions_manifest is not None and (prepared is not None or benchmark is not None):
        raise ValueError("conditions manifests are not supported with prepared contexts or benchmarks")
    from .transport import _scatter_in_kw
    bench_start = time.perf_counter() if benchmark is not None else None
    rc, cfg, gdml_path, time_window, spec = _resolve(library, calib, deviation, gdml, jax, counters, run_config, window,
                                                     kw.get("fused_response", False), conditions_manifest,
                                                     tier1b_set, tier1b_dir)
    for warning in time_window["warnings"]:
        print(f"# time window: {warning}", flush=True)
    if benchmark is not None:
        benchmark.setup(rc, cfg, gdml_path, spec, calib, deviation)
    t0 = time.perf_counter()
    if prepared is None:
        manifest_kw = {}
        if conditions_manifest is not None:
            # Validate the stock snapshot against its configured window, not the cosmic crop.
            configured_rc = dataclasses.replace(rc, window=tuple(time_window["configured_window_ns"]))
            manifest_kw = dict(conditions_manifest=conditions_manifest, configured_rc=configured_rc)
        payload, T = _load_setup(rc, cfg, gdml_path, gdml, spec, calib, deviation, jax, no_absorbers,
                                kw.get("unbanded_world", False), kw.get("no_below_world", False),
                                kw.get("below_margin"), **manifest_kw)
        if benchmark is None:
            T.pop("geometry_conditions_config_s")
    else:
        signature = _signature(rc, cfg, gdml_path, spec, calib, deviation, no_absorbers,
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
    readout_metadata = {}
    if geometry_readout:
        cond, evidence = readout.apply(cond, geom, ccfg)
        readout_metadata["geometry_readout"] = evidence
    tagger, bar_cut, world, ab, below = (payload[k] for k in ("tagger", "bar_cut", "world", "ab", "below"))
    lib, tabs, pmap, stype_name = (payload[k] for k in ("lib", "tabs", "pmap", "stype_name"))
    conditions_metadata = payload.get("conditions_metadata",
                                      {"status": "unverified_legacy", "calib": calib, "deviation": deviation})
    out_cl, n_gen = [], 0
    transport_diagnostics = {}

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
        diagnostics = {}
        st, pa = tr.transport(gen, geom, rng, stage="C", world=world, msc_tail=msc_tail, tail_k=tail_k,
                              sliced_msc=sliced_msc, all_muons=not first_muon_only, absorbers=ab, tagger_bar=tagger,
                              bar_cut=bar_cut, below=below,
                              below_msc=below is not None and (rc.below_msc or bool(kw.get("below_msc"))),
                              **_scatter_in_kw(kw, diagnostics))
        if diagnostics:
            transport_diagnostics[str(i)] = {"input": str(f), **diagnostics}
        t["transport_s"] = time.perf_counter() - a
        a = time.perf_counter()
        st, pa, _ = tier1b.sample(st, pa, gen, lib, geom, rng, showers=rc.showers, **spec.options)
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
    res = {"generated_events": n_gen, "timers": T, "conditions": conditions_metadata, **readout_metadata,
           "tier1b": spec.describe(),
           "run_config": rc.name, "window": list(cfg.window), "time_window": time_window,
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
    if transport_diagnostics:
        res["scatter_in_diagnostics"] = transport_diagnostics
    res["wall_s"] = time.perf_counter() - t0
    if jax:
        import jax as _jax
        res["device"] = str(_jax.devices()[0])
        res["reco_device"] = "cpu" if reco_cpu else res["device"]
    if emit:   # per-shard accumulators for a merge (node jobs: one process per tile)
        with open(emit, "w") as fh:
            json.dump({"generated_events": n_gen, "timers": T, "wall_s": res["wall_s"],
                       "conditions": conditions_metadata, "scatter_in_diagnostics": transport_diagnostics,
                       "tier1b": spec.describe(),
                       **readout_metadata,
                       "cl_type_counts": {int(k): int((cl_type == k).sum()) for k in np.unique(cl_type)},
                       "run_config": rc.name, "window": list(cfg.window), "time_window": time_window,
                       "type_names": {int(k): v for k, v in stype_name.items()},
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
    time_metadata = validate_runchain_merge(J)
    time_metadata.update(readout.validate_merge(J))
    if any(j.get("conditions") != J[0].get("conditions") for j in J[1:]):
        raise ValueError("runchain merge: incompatible conditions provenance")
    if J[0].get("conditions") is not None:
        time_metadata["conditions"] = J[0]["conditions"]
    t1b = lambda j: {k: (j.get("tier1b") or {}).get(k) for k in ("set", "options", "library_sha256")}
    if any(t1b(j) != t1b(J[0]) for j in J[1:]):
        raise ValueError("runchain merge: shards ran different tier-1b asset sets")
    if J[0].get("tier1b") is not None:
        time_metadata["tier1b"] = t1b(J[0])
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
    res.update(time_metadata)
    plots.write_json(outdir, "runchain_merge_stats.json", res)
    print(json.dumps(res, indent=1))
    return res
