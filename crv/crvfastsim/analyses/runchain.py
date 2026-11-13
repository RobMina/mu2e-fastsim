"""runchain -- the whole fast CRV chain on CRY gen tables, no art: transport -> tier 1b -> photons -> SiPM charges ->
digis -> pulses -> clusters (crvfastsim/chain.py), with the cost of every stage. With --compare (CrvRecoTableDumper
files of the stock chain on the same shards) it reports clusters per generated event per sector type, fast vs stock.

    runchain gen.<shard>.root [...] --calib crvcalib.csv [--deviation crvphoton.csv] [--jax]
             [--tier1b strict_v5 [--tier1b-dir DIR] | --library lib.root]
             [--reco-cpu] [--max-events N] [--compare ref.<shard>.root ...] [--emit shard.json] --outdir DIR
             [--steps-input] [--save-reco reco.npz]   (CrvStep tables in: no transport or tier 1b; clusters, members
                                                       and pulses of one input file out, for validation). With
                                                       --steps-input generated_events counts the events with steps
                                                       (events_basis in the stats and --emit)
    runchain-merge shard*.json --outdir DIR      (node totals: events, clusters per sector type, rates)

Tier 1b (crvfastsim.tier1b_assets): default asset set strict_v5 for extracted_v04 (strict library with step-tree roots,
crossing context, containment, point and topology placement, v3 templates, chord placement, root-coherent draws);
run1a_v01 has its own set (--tier1b run1a_v01, opt-in); the other named configurations need --library, a custom
library sampled without placement options. A set may switch the non-muon showers on (options['showers']).
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

from .. import chain, geometry, host_workers, readout, runconfig, tables, tier1b, tier1b_assets
from ..timewindow import resolve_runchain_window, validate_runchain_merge
from .. import transport as tr
from ..response import photons as PH
from ..response import sipm as SI

DEFAULT_COUNTERS = "Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"
SECTOR_TYPES = {1: "EX", 2: "T1", 3: "T2", 4: "M"}


def resident_default(resident_response, jax, fused_response=False, backend=None):
    """Whether runchain keeps the photons on the device into the SiPM stage: None (the default) means on the JAX path
    on an accelerator unless the fused response is chosen, and off on the CPU backend, where nothing crosses a bus and
    the two-stage twins run faster (W5 on one core: 90.7 against 79.3 ev/s, 11.3 against 13.8 GB). Same draws and
    charges either way (byte-identical outputs). backend: the JAX default backend (looked up when None)."""
    if resident_response is None:
        if not jax or fused_response:
            return False
        if backend is None:
            import jax as _jax
            backend = _jax.default_backend()
        return backend != "cpu"
    return bool(resident_response)


def check_conditions(rc, calib):
    """Refuse --calib nominal for a configuration whose stock job reads its own conditions (RunConfig.conditions):
    nominal conditions count the channels the status table ignores as live (Run 1A: 45 of 5120, about 1 % more pulses
    than stock) and drop the time offsets and photon-yield deviations."""
    if calib == "nominal" and rc.conditions:
        raise SystemExit(f"runchain: {rc.name} uses the {rc.conditions} conditions of its stock job; --calib nominal is "
                         f"refused (it counts the channels the {rc.conditions} status table ignores as live). Pass the "
                         f"converted CSVs: --calib/--deviation from scripts/offline_conditions_csv.py on "
                         f"CRVConditions/data/{{calib,status,photonyield}}_{rc.conditions}.txt")


def _resolve(library, calib, deviation, gdml, jax, counters, run_config, window, fused_response=False,
             tier1b_set=None, tier1b_dir=None, resident_response=None, device_refit=False):
    """Resolve setup arguments without loading large assets or drawing random numbers.

    resident_response: resident_default (None: on with jax on an accelerator unless fused_response, off on the CPU
    backend; False: the two-stage path)."""
    if not calib:
        raise SystemExit("runchain needs --calib (a CRVCalib CSV or nominal)")
    from .transport import default_gdml
    rc, time_window = resolve_runchain_window(run_config, window)
    check_conditions(rc, calib)
    if counters:
        rc = dataclasses.replace(rc, counters=counters)
    cfg = rc.chain_config(jax=bool(jax)) if rc.name != runconfig.DEFAULT or window else chain.ChainConfig(jax=bool(jax))
    if fused_response:
        cfg = dataclasses.replace(cfg, fused_response=True)
    if resident_default(resident_response, jax, fused_response):
        cfg = dataclasses.replace(cfg, resident_response=True)
    if device_refit:
        cfg = dataclasses.replace(cfg, device_refit=True)
    gdml_path = gdml or (rc.gdml_path() if rc.name != runconfig.DEFAULT else default_gdml())
    try:
        spec = tier1b_assets.resolve(tier1b_set, library, tier1b_dir, rc.name)
    except ValueError as e:
        raise SystemExit(f"runchain: {e}")
    return rc, cfg, gdml_path, time_window, spec


def _signature(rc, cfg, gdml, spec, calib, deviation, below_margin=None):
    canonical = lambda p: str(Path(p).resolve()) if p and p != "nominal" else p
    return {"run_config": dataclasses.asdict(rc), "chain_config": dataclasses.asdict(cfg),
            "paths": [canonical(p) for p in (gdml, *spec.paths(), calib, deviation)],
            "tier1b": {"set": spec.name, "options": dict(spec.options), "library_sha256": spec.library_sha256},
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


def prepare(library=None, calib=None, deviation=None, gdml=None, jax=False, counters=None,
            run_config=None, window=None, benchmark=None, fused_response=False,
            below_margin=None, tier1b_set=None, tier1b_dir=None, resident_response=None, device_refit=False):
    """Load and seal reusable CSV setup."""
    if fused_response:
        raise ValueError("prepared context does not support memory-dependent fused response batching")
    from .context_audit import freeze_arrays, content_fingerprint
    from ..reco import waveform as WF
    rc, cfg, gdml_path, _, spec = _resolve(library, calib, deviation, gdml, jax, counters, run_config, window,
                                           fused_response, tier1b_set=tier1b_set, tier1b_dir=tier1b_dir,
                                           resident_response=resident_response, device_refit=device_refit)
    rc, cfg = copy.deepcopy((rc, cfg))
    sig = _signature(rc, cfg, gdml_path, spec, calib, deviation, below_margin)
    if benchmark is not None:
        benchmark.setup(rc, cfg, gdml_path, spec, calib, deviation)
    payload, timers = _load_setup(rc, cfg, gdml_path, gdml, spec, calib, deviation, jax, below_margin)
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


def _load_setup(rc, cfg, gdml_path, explicit_gdml, spec, calib, deviation, jax, below_margin=None):
    geometry_start = time.perf_counter()
    geom = rc.geometry()
    ccfg = rc.counters_cfg()
    conditions_metadata = {"status": "unverified_legacy", "calib": calib, "deviation": deviation}
    cond = runconfig.nominal_conditions(geom) if calib == "nominal" else chain.Conditions.from_csv(calib, deviation)
    if len(cond.status) < 4 * geom.nbars:
        raise SystemExit(f"runchain: conditions cover {len(cond.status)} channels, {rc.name} has {4 * geom.nbars}; "
                         "pass the configuration's conditions CSVs (--calib/--deviation: the converted conditions of the "
                         "input release, or scripts/offline_conditions_csv.py on the Offline CRVConditions text files); "
                         "--calib nominal is only for configurations without stock conditions (it treats every "
                         "channel as live)")
    cond = readout.apply(cond, geom, ccfg)   # physically absent SiPMs make no charges (Offline hasCMB)
    tagger = rc.tagger_bar(geom) if rc.name != runconfig.DEFAULT else None
    bar_cut = rc.bar_cut(geom)
    timers = {"geometry_conditions_config_s": time.perf_counter() - geometry_start}
    a = time.perf_counter()
    world, ab = tr.load_world(gdml_path, banded=True, y_plane=tr.msc_plane(geom))
    below = None
    if rc.below_world:
        from .transport import _below_kw
        below, ab = tr.load_below(explicit_gdml or rc.gdml_path(), geom,
                                  **_below_kw(rc, {"below_margin": below_margin})), None
    lib = tier1b_assets.load(spec, geom)   # inside setup_s (no own timer: every *_s key counts as a stage)
    tabs = rc.lookup_tables(cfg.response)        # pinned by the run configuration's asset set
    if jax:
        from ..response import photons_jax
        tabs = {k: photons_jax.DeviceTable(v) for k, v in tabs.items()}
    pmap = rc.photon_map()
    timers["setup_s"] = time.perf_counter() - a
    stype_name = runconfig.sector_type_names(geom, ccfg) if rc.name != runconfig.DEFAULT else SECTOR_TYPES
    return dict(rc=rc, cfg=cfg, geom=geom, ccfg=ccfg, cond=cond, tagger=tagger, bar_cut=bar_cut,
                world=world, ab=ab, below=below, lib=lib, tabs=tabs, pmap=pmap, stype_name=stype_name,
                conditions_metadata=conditions_metadata), timers


def approximations_record(selection, jax):
    """The approximations record of a run (mu2efastsim.approximations): ``selection`` is a Selection (the CLI), a
    config section ({preset: ..., NAME: LEVEL}) or None (the default levels). Python callers pass the options the
    levels set themselves (Selection.apply_kwargs); this only records them."""
    from mu2efastsim import approximations as A
    from .. import approximations as W5
    if selection is None:
        selection = W5.WORKFLOW.select()
    elif not isinstance(selection, A.Selection):
        preset, overrides = A.read_section(selection, W5.WORKFLOW.name)
        selection = W5.WORKFLOW.select(preset, overrides)
    if not jax:
        selection.set_effective("pulse_fit", "full_length", "no --jax: the NumPy response fits every waveform at "
                                                            "full length")
    return selection.record()


def _save_reco(path, outs, n_events, run_config, approximations=None):
    """Clusters (cl_*), their member pulses (mem_cl, mem_idx: pulse index in its event) and the pulses (p_ev, p_idx,
    p_bar, p_sipm, p_pe, p_time) of one runchain input, as an .npz (the validation input of the reco comparison), with
    the run's approximations record (JSON) when given."""
    cl = {k: [] for k in ("cl_event", "cl_type", "cl_pe", "cl_start", "cl_end", "cl_avgtime")}
    mem = {k: [] for k in ("mem_cl", "mem_idx")}
    pul = {k: [] for k in ("ev", "idx", "bar", "sipm", "pe", "time")}
    ncl = 0
    for o in outs:
        for k in cl:
            cl[k].append(np.asarray(o.get(k, [])))
        # Cluster indices are chunk-local; pulse indices remain local to their event.
        mem["mem_cl"].append(np.asarray(o.get("mem_cl", []), np.int64) + ncl)
        mem["mem_idx"].append(np.asarray(o.get("mem_idx", []), np.int64))
        p = o.get("pulses", {})
        for k in pul:
            pul[k].append(np.asarray(p.get(k, [])))
        ncl += len(cl["cl_event"][-1])

    def cat(parts, dtype=None):
        # Untyped empty chunks must not promote populated integer or float32 columns.
        populated = [p for p in parts if len(p)]
        return np.concatenate(populated or parts) if parts else np.empty(0, dtype=dtype)

    np.savez_compressed(path, **{k: cat(v) for k, v in cl.items()},
                        **{k: cat(v, np.int64) for k, v in mem.items()},
                        **{f"p_{k}": cat(v) for k, v in pul.items()},
                        n_events=np.int64(n_events), run_config=np.str_(run_config),
                        **({} if approximations is None else
                           {"approximations": np.str_(json.dumps(approximations, sort_keys=True))}))


def resolve_generator(gen, inputs, jax):
    """Generator backend and inputs. "auto" (the runchain CLI default): the gen tables given as inputs (the CPU path and
    the reference); with no inputs, or the placeholder 'native', device CRY when the chain runs on JAX (--jax)."""
    inputs = list(inputs)
    if gen not in ("auto", "table", "native", "device"):
        raise ValueError("gen must be auto, table, native or device")
    if gen in ("native", "device") and not inputs:
        inputs = ["native"]
    if gen == "auto":
        if inputs and inputs != ["native"]:
            return "table", inputs
        if not jax:
            raise SystemExit("runchain: give gen tables (the CPU path), or --jax for device CRY, or --gen native")
        return "device", ["native"]
    if not inputs:
        raise SystemExit("runchain: gen tables needed with --gen table")
    return gen, inputs


DEVICE_TIER1B_SETS = ("strict_v5",)


def resolve_backends(jax, transport_backend, tier1b_backend, rc, spec, below, kw):
    """Backends of transport and tier 1b. "auto" (the runchain CLI default) is the device backend when the chain runs
    on JAX (--jax) and the device twin covers the configuration: the extracted run configuration with the straight-line
    world (no below-plane material or MSC), a strict asset set for tier 1b and no prefetch; NumPy otherwise."""
    device = bool(jax) and rc.name == runconfig.DEFAULT
    if transport_backend == "auto":
        transport_backend = "jax" if device and below is None and not kw.get("below_msc") else "numpy"
    if tier1b_backend == "auto":
        tier1b_backend = "jax" if device and spec.name in DEVICE_TIER1B_SETS and not kw.get("prefetch") else "numpy"
    return transport_backend, tier1b_backend


def _add_receipt(total, part):
    """One receipt for consecutive chunks of one source: counts and live time summed, the event range spanned."""
    if total is None:
        return dict(part, chunks=1)
    out = dict(part)
    out["accepted_events"] = total["accepted_events"] + part["accepted_events"]
    out["live_seconds"] = total["live_seconds"] + part["live_seconds"]
    out["event_range"] = dict(part["event_range"], first_event=total["event_range"]["first_event"])
    out["chunks"] = total["chunks"] + 1
    return out


def run(inputs, outdir, library=None, calib=None, deviation=None, gdml=None, jax=False, max_events=None, seed=1,
        compare=None, counters=None, reco_cpu=False, emit=None, run_config=None, window=None, benchmark=None,
        prepared=None,
        tier1b_set=None, tier1b_dir=None, steps_input=False, save_reco=None,
        transport_context=None, tier1b_context=None, tier1b_backend="numpy",
        gen="table", gen_events=None, gen_seed=None, gen_subrun=1, gen_chunk=50000, gen_tables=None,
        approximations=None, **kw):
    removed = sorted(set(kw) & {"msc_tail", "tail_k", "sliced_msc", "first_muon_only", "no_absorbers",
                                "unbanded_world", "no_below_world"})
    if removed:
        raise ValueError(f"runchain: options {removed} were removed (the transport subcommand keeps them)")
    approx = approximations_record(approximations, jax)
    caps = kw.get("capacities")   # crvfastsim.capacity.Capacities: fixed device shapes (resident JAX path)
    if caps is not None and not jax:
        raise ValueError("capacities need the JAX path")
    caps_kw = {} if caps is None else {"caps": caps}
    gen, inputs = resolve_generator(gen, inputs, jax)
    native = gen in ("native", "device")
    generator_backend = gen
    if native:
        if list(inputs) != ["native"]:
            raise ValueError("--gen native requires exactly one placeholder input: native")
        if not isinstance(gen_chunk, int) or isinstance(gen_chunk, bool) or gen_chunk < 1:
            raise ValueError("--gen-chunk must be a positive integer")
        if not isinstance(gen_events, int) or isinstance(gen_events, bool) or not 0 < gen_events < 2**20:
            raise ValueError("--gen native needs --gen-events in [1,2**20-1] for packed event keys")
        if not isinstance(gen_subrun, int) or isinstance(gen_subrun, bool) or not 0 <= gen_subrun < 2**28:
            raise ValueError("native gen-subrun must be in [0,2**28-1] for packed int64 channel labels")
        effective_gen_seed = seed if gen_seed is None else gen_seed
        if not isinstance(effective_gen_seed, int) or isinstance(effective_gen_seed, bool) or not 0 <= effective_gen_seed < 2**64:
            raise ValueError("native gen-seed must be uint64")
        if max_events or prepared is not None or benchmark is not None or kw.get("prefetch") or compare:
            raise ValueError("native generation does not support max-events/prepared/benchmark/prefetch/paired compare")
        if run_config not in (None, "extracted_v04"):
            raise ValueError("native CRY is currently configured only for extracted_v04")
    elif gen_events is not None or gen_seed is not None:
        raise ValueError("gen-events/gen-seed require --gen native")
    if (steps_input or save_reco) and (native or benchmark is not None or prepared is not None or kw.get("prefetch")):
        raise ValueError("--steps-input / --save-reco run on table inputs without benchmark, prepared context or prefetch")
    if save_reco and len(list(tables.expand(inputs))) != 1:
        raise ValueError("--save-reco takes one input file")
    native_receipt = None
    bench_start = time.perf_counter() if benchmark is not None else None
    rc, cfg, gdml_path, time_window, spec = _resolve(library, calib, deviation, gdml, jax, counters, run_config, window,
                                                     kw.get("fused_response", False), tier1b_set, tier1b_dir,
                                                     kw.get("resident_response"), kw.get("device_refit", False))
    for warning in time_window["warnings"]:
        print(f"# time window: {warning}", flush=True)
    if benchmark is not None:
        benchmark.setup(rc, cfg, gdml_path, spec, calib, deviation)
    t0 = time.perf_counter()
    if prepared is None:
        payload, T = _load_setup(rc, cfg, gdml_path, gdml, spec, calib, deviation, jax, kw.get("below_margin"))
        if benchmark is None:
            T.pop("geometry_conditions_config_s")
    else:
        signature = _signature(rc, cfg, gdml_path, spec, calib, deviation, kw.get("below_margin"))
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
    conditions_metadata = payload.get("conditions_metadata",
                                      {"status": "unverified_legacy", "calib": calib, "deviation": deviation})
    out_cl, n_gen = [], 0
    kw["transport_backend"], tier1b_backend = resolve_backends(jax, kw.get("transport_backend", "numpy"), tier1b_backend,
                                                               rc, spec, below, kw)
    device_transport = None
    range_out = rc.range_out and not kw.get("legacy_transport")
    if kw.get("transport_backend", "numpy") == "jax":
        from ..transport_jax import DeviceTransport, transport as run_device_transport
        if below is not None or kw.get("below_msc"):
            raise ValueError("JAX transport supports the straight-line default only")
        if transport_context is not None and prepared is None:
            raise ValueError("device transport reuse requires an audited prepared chain context")
        cached = transport_context.get("device") if transport_context is not None else None
        if cached is not None and (cached.geom is not geom or cached.stage != "C" or cached.range_out != bool(range_out)):
            raise ValueError("cached device transport does not match prepared geometry")
        device_transport = cached or DeviceTransport(geom, world, ab, tagger, bar_cut, range_out=range_out)
        if transport_context is not None:
            transport_context["device"] = device_transport

    device_tier1b = None
    if tier1b_backend not in ("numpy", "jax"):
        raise ValueError(f"unknown tier1b backend {tier1b_backend!r}")
    if tier1b_backend == "jax":
        from ..tier1b_jax import DeviceRandom, DeviceTier1b
        if spec.name not in DEVICE_TIER1B_SETS or rc.name != runconfig.DEFAULT:
            raise ValueError("device tier1b supports the extracted strict_v5 set only")
        if kw.get("prefetch"):
            raise ValueError("device tier1b prefetch is not certified")
        if tier1b_context is not None and prepared is None:
            raise ValueError("device tier1b reuse requires an audited prepared context")
        cached = tier1b_context.get("device") if tier1b_context is not None else None
        if cached is not None and (cached.geom is not geom or cached.lib is not lib):
            raise ValueError("cached device tier1b does not match prepared assets")
        a = time.perf_counter()
        device_tier1b = cached or DeviceTier1b(lib, geom)
        T["tier1b_preparation_s"] = time.perf_counter() - a
        if tier1b_context is not None:
            tier1b_context["device"] = device_tier1b

    source = None
    tier1b_options = {"showers": rc.showers, **spec.options}
    host_ctx = dict(geom=geom, world=world, ab=ab, tagger=tagger, bar_cut=bar_cut, below=below, range_out=range_out,
                    below_msc=below is not None and (rc.below_msc or bool(kw.get("below_msc"))), lib=lib,
                    tier1b_options=tier1b_options)

    def host_stage(i, f):
        """CPU part of file i (read, transport, tier 1b) with its own rng and timers: the same draws whether it runs
        inline or in the prefetch thread."""
        rng = np.random.default_rng([int(seed), i])
        t = {}
        a = time.perf_counter() if benchmark is not None else None
        nonlocal native_receipt, source
        if steps_input:   # G4 or tier-1b step tables: the response and reco only
            st, pa = tables.read(f, "steps"), tables.read(f, "particles")
            if max_events:
                ev = np.sort(st.event.unique())[:int(max_events)]
                st, pa = st[st.event.isin(ev)].reset_index(drop=True), pa[pa.event.isin(ev)].reset_index(drop=True)
            return int(st[tables.KEY].drop_duplicates().shape[0]), st, pa, rng, t, None
        if native:   # chunk i of gen_events, from one source opened at the first chunk
            from mu2efastsim.cosmic_cry import CosmicConfig, CosmicSource
            started = time.perf_counter()
            if source is None:
                if generator_backend == "device":
                    from mu2efastsim.cry_device import DeviceCosmicSource as Source
                else:
                    Source = CosmicSource
                source = Source(CosmicConfig.extracted(gdml_path), seed=seed if gen_seed is None else gen_seed).__enter__()
            n = min(gen_chunk, gen_events - i * gen_chunk)
            gen, _, receipt = (source.generate(n, subrun=gen_subrun) if i == 0 else
                               source.generate(n, subrun=gen_subrun, first_event=1 + i * gen_chunk))
            native_receipt = _add_receipt(native_receipt, receipt)
            t["generation_s"] = time.perf_counter() - started
        elif gen_tables is not None and f in gen_tables:   # read into memory by the caller (analyses.runchain_bench)
            gen = gen_tables[f].copy()
        else:
            gen = tables.read(f, "gen")
        if benchmark is not None:
            t["input_io_s"] = time.perf_counter() - a
            benchmark.check_input(i, f, gen)
        if max_events:
            keep = gen.event.isin(np.sort(gen.event.unique())[:int(max_events)])
            gen = gen[keep].reset_index(drop=True)
        keys = gen[tables.KEY].drop_duplicates().to_numpy() if benchmark is not None else None
        ngen = int(gen[tables.KEY].drop_duplicates().shape[0])
        if device_transport is None and device_tier1b is None:   # the NumPy host stage (as in --host-workers)
            st, pa = host_workers.numpy_host_stage(gen, rng, host_ctx, t)
            return ngen, st, pa, rng, t, keys
        a = time.perf_counter()
        if device_transport is not None:
            st, pa = run_device_transport(gen, geom, rng, prepared=device_transport)
        else:
            st, pa = tr.transport(gen, geom, rng, stage="C", world=world, absorbers=ab, tagger_bar=tagger,
                                  bar_cut=bar_cut, below=below, range_out=range_out,
                                  below_msc=host_ctx["below_msc"])
        t["transport_s"] = time.perf_counter() - a
        a = time.perf_counter()
        if device_tier1b is None:
            st, pa, _ = tier1b.sample(st, pa, gen, lib, geom, rng, **tier1b_options)
        else:
            device_rng = DeviceRandom(int(rng.integers(0, 2**32, dtype=np.uint32)))
            st, pa, _ = device_tier1b.sample(st, pa, gen, device_rng, **tier1b_options)
        t["tier1b_s"] = time.perf_counter() - a
        return ngen, st, pa, rng, t, keys

    files = list(tables.expand(inputs)) if not native else [f"native:{k}" for k in range(-(-gen_events // gen_chunk))]
    pool, workers = None, int(kw.get("host_workers") or 0)
    if workers and (native or steps_input or benchmark is not None or device_transport is not None
                    or device_tier1b is not None):
        raise ValueError("--host-workers runs the NumPy transport and tier 1b of table inputs (no native generator, "
                         "steps input, detail benchmark or device transport / tier 1b)")
    if workers:   # host stages of the next `workers` files in worker processes (same draws as in process)
        wpool = host_workers.pool(host_ctx, workers, (id(geom), id(world), id(lib), workers))
        gen_of = lambda f: gen_tables.get(f) if gen_tables is not None else None
        pending = {j: host_workers.submit(wpool, j, files[j], gen_of(files[j]), seed, max_events)
                   for j in range(len(files))}                  # all queued now: the workers take them in order
        if kw.get("host_ahead"):   # (files, gen tables, seed) of the next run: queued behind this run's files
            host_workers.submit_ahead(wpool, *kw["host_ahead"], max_events)
    elif kw.get("prefetch") and len(files) > 1:   # file i+1's host stage runs while the device works on file i
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=1)
        nxt = pool.submit(host_stage, 0, files[0])
    try:
        for i, f in enumerate(files):
            if workers:
                a = time.perf_counter()
                ngen, st, pa, rng, t = pending.pop(i).result()
                keys = None
                T["host_wait_s"] = T.get("host_wait_s", 0.0) + time.perf_counter() - a
            elif pool is not None:
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
            Q = chain.charges(st, pa, geom, cond, cfg, rng, tabs=tabs, pmap=pmap, timers=file_timers, **caps_kw)
            D = chain.digis(Q, cond, cfg, rng, timers=file_timers, **caps_kw)
            a = time.perf_counter() if benchmark is not None else None
            W = chain.reco_waveforms(D, cond)
            if benchmark is not None:
                file_timers["waveform_pack_s"] = time.perf_counter() - a
            a = time.perf_counter()
            detail_kw = {"detail": True} if benchmark is not None or save_reco else {}
            if reco_cpu and jax:   # response on the accelerator, reco with the JAX CPU backend (oneAPI: reco kernels slow)
                import jax as _jax
                with _jax.default_device(_jax.devices("cpu")[0]):
                    outs, _ = chain.clusters(W, cond, geom, ccfg, cfg, timers=file_timers, **caps_kw, **detail_kw)
            else:
                outs, _ = chain.clusters(W, cond, geom, ccfg, cfg, timers=file_timers, **caps_kw, **detail_kw)
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
        if source is not None:
            source.__exit__(None, None, None)
    if prepared is not None:
        a = time.perf_counter()
        prepared.audit()
        T["context_audit_s"] += time.perf_counter() - a
    if save_reco:
        _save_reco(save_reco, out_cl, n_gen, rc.name, approx)
    cl_type = np.concatenate([o["cl_type"] for o in out_cl]) if out_cl else np.zeros(0, int)
    res = {"generated_events": n_gen, "timers": T, "conditions": conditions_metadata,
           "tier1b": spec.describe(),
           "transport_backend": "jax" if device_transport is not None else "numpy", "range_out": bool(range_out),
           "tier1b_backend": "jax" if device_tier1b is not None else "numpy",
           "run_config": rc.name, "window": list(cfg.window), "time_window": time_window,
           "approximations": approx,
           "events_basis": "events_with_steps" if steps_input else "generated",
           "clusters_per_generated": {stype_name.get(int(k), str(k)): float((cl_type == k).sum() / n_gen)
                                      for k in np.unique(cl_type)}}
    if caps is not None:
        res["capacity"] = caps.report()
    if native_receipt is not None:
        live = native_receipt["live_seconds"]
        res["native_generator"] = native_receipt
        res["live_seconds"] = live
        res["clusters_per_live_second"] = {stype_name.get(int(k), str(k)): float((cl_type == k).sum() / live)
                                            for k in np.unique(cl_type)}
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
        from mu2efastsim import precision as _dprec
        res["device"] = str(_jax.devices()[0])
        res["reco_device"] = "cpu" if reco_cpu else res["device"]
        res["precision"] = _dprec.provenance()   # float32 discrete results depend on the device model: record both
    if emit:   # per-shard accumulators for a merge (node jobs: one process per tile)
        with open(emit, "w") as fh:
            json.dump({"generated_events": n_gen, "timers": T, "wall_s": res["wall_s"],
                       "conditions": conditions_metadata,
                       **({"native_generator": native_receipt, "live_seconds": native_receipt["live_seconds"]}
                          if native_receipt is not None else {}),
                       "tier1b": spec.describe(),
                       "cl_type_counts": {int(k): int((cl_type == k).sum()) for k in np.unique(cl_type)},
                       "run_config": rc.name, "window": list(cfg.window), "time_window": time_window,
                       "approximations": approx,
                       "events_basis": "events_with_steps" if steps_input else "generated",
                       "type_names": {int(k): v for k, v in stype_name.items()},
                       "device": res.get("device"), "reco_device": res.get("reco_device"),
                       "precision": res.get("precision"), "inputs": list(tables.expand(inputs))}, fh)
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
    from mu2efastsim.cry_receipts import merge_receipts
    generator_metadata = merge_receipts(J)
    if any(j.get("conditions") != J[0].get("conditions") for j in J[1:]):
        raise ValueError("runchain merge: incompatible conditions provenance")
    if J[0].get("conditions") is not None:
        time_metadata["conditions"] = J[0]["conditions"]
    t1b = lambda j: {k: (j.get("tier1b") or {}).get(k) for k in ("set", "options", "library_sha256")}
    if any(t1b(j) != t1b(J[0]) for j in J[1:]):
        raise ValueError("runchain merge: shards ran different tier-1b asset sets")
    if J[0].get("tier1b") is not None:
        time_metadata["tier1b"] = t1b(J[0])
    if any(j.get("approximations") != J[0].get("approximations") for j in J[1:]):
        raise ValueError("runchain merge: shards ran different approximation levels")
    if J[0].get("approximations") is not None:
        time_metadata["approximations"] = J[0]["approximations"]
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
           "devices": sorted({str(j.get("device")) for j in J}), "reco_devices": sorted({str(j.get("reco_device")) for j in J}),
           "precisions": sorted({json.dumps(j.get("precision"), sort_keys=True) for j in J})}
    res.update(time_metadata)
    res.update(generator_metadata)
    if generator_metadata:
        live = generator_metadata["live_seconds"]
        res["clusters_per_live_second"] = {names.get(k, str(k)): v / live for k, v in sorted(cl.items())}
    plots.write_json(outdir, "runchain_merge_stats.json", res)
    print(json.dumps(res, indent=1))
    return res
