"""C4 response on the fixed stock selection with represented muons or retained generators."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import awkward as ak
import numpy as np
import pandas as pd

from .paired import KEY


def rekey(selection, primaries, input_kind="represented"):
    """Use dense internal event IDs so labels cannot alias across runs or subruns."""
    events = pd.DataFrame({k: ak.to_numpy(selection[k]) for k in KEY})
    if input_kind == "retained":
        from .generators import integer_columns, validate_generators
        integer_columns(events, KEY, "selection")
        validate_generators(primaries)
    events["dense"] = np.arange(1, len(events) + 1)
    gen = primaries.merge(events, on=list(KEY), how="left", validate="many_to_one")
    if gen.dense.isna().any():
        raise ValueError("primary event absent from cached selection")
    if input_kind == "retained":
        for field in (*KEY, "idx"):
            gen["source_" + field] = gen[field]
        gen = gen.sort_values(["dense", "gen_product_id", "idx"], kind="stable")
        if "e" not in gen or not np.isfinite(gen.e).all():
            raise ValueError("retained generators require finite stored total energy")
    elif input_kind == "represented":
        if (gen.pdg.abs() != 13).any():
            raise ValueError("represented-input route accepts muons only; use retained inputs for other species")
        from mu2efastsim.eloss import MMU
        gen["e"] = np.sqrt(gen.px**2 + gen.py**2 + gen.pz**2 + MMU**2)
    else:
        raise ValueError("unknown input kind: " + input_kind)
    gen["run"], gen["subrun"], gen["event"] = 1, 0, gen.pop("dense").to_numpy(np.int64)
    gen["idx"] = gen.groupby(list(KEY)).cumcount()
    return gen


def bunch_frame(steps, selection):
    """Convert physical step times to the true bunch frame before fixed-window digitisation."""
    out = steps.copy()
    out["t0"] += ak.to_numpy(selection.pbtime_mc)[out.event.to_numpy(np.int64) - 1]
    out["t1"] += ak.to_numpy(selection.pbtime_mc)[out.event.to_numpy(np.int64) - 1]
    return out


def coincidence_rows(outs, geom, selection):
    """Return all cached events, including zero-response events, with reconstructed coincidences."""
    from ..reco import ntview
    events = np.arange(1, len(selection) + 1, dtype=np.int64)
    if not outs or not any(len(o["cl_event"]) for o in outs):
        counts = np.zeros(len(events), dtype=np.int64)
        fields = {k: ak.unflatten(np.zeros(0, dtype=np.int32 if k in ("nLayers", "sectorType") else float), counts)
                  for k in ("time", "timeStart", "timeEnd", "PEs", "nLayers", "sectorType")}
        return ak.zip({**{k: selection[k] for k in KEY}, "coincs": ak.zip(fields)}, depth_limit=1)
    co, _, _ = ntview.build(outs, geom, events=events)
    clusters = {k: np.concatenate([o[k] for o in outs]) if outs else np.zeros(0)
                for k in ("cl_event", "cl_start", "cl_end")}
    order = np.argsort(clusters["cl_event"], kind="stable")
    ev = clusters["cl_event"][order].astype(np.int64)
    counts = np.bincount(ev, minlength=len(events) + 1)[1:]
    reco_offset = ak.to_numpy(selection.pbtime)
    fields = {k: co["crvcoincs." + k] for k in ("time", "PEs", "nLayers", "sectorType")}
    fields["time"] = fields["time"] - reco_offset
    for key, source in (("timeStart", "cl_start"), ("timeEnd", "cl_end")):
        fields[key] = ak.unflatten(clusters[source][order] - reco_offset[ev - 1], counts)
    return ak.zip({**{k: selection[k] for k in KEY}, "coincs": ak.zip(fields)}, depth_limit=1)


def onspill_charges(steps, parts, geom, cond, cfg, rng, tabs, pmap, timers, period):
    """Wrap photon arrival times by the stock microbunch period before SiPM simulation."""
    import jax
    from .. import chain, readout
    from ..response import photons_jax, sipm_jax
    if not np.isfinite(period) or period <= cfg.window[1]:
        raise ValueError("microbunch period must exceed the digitisation window")
    start = time.perf_counter()
    pdg = steps.merge(parts[list(KEY) + ["key", "pdg"]].rename(columns={"key": "sp"}),
                      on=list(KEY) + ["sp"], how="left", validate="many_to_one").pdg.to_numpy()
    key = jax.random.PRNGKey(int(rng.integers(2**31)))
    k1, k2 = jax.random.split(key)
    s, channel, arrival = photons_jax.make_photons(steps, pdg, geom, cfg.response, tabs, cond.deviation, rng, k1)
    status = readout.effective_status(cond)
    keep = (status[channel] & 0b101) == 0
    s, channel = s[keep], channel[keep]
    arrival = np.mod(arrival[keep], period)
    timers["photons_s"] = time.perf_counter() - start
    timers["photons"] = len(arrival)
    labels = chain.event_key(steps)[s] * chain.CHAN_MULT + channel
    start = time.perf_counter()
    charges = sipm_jax.simulate(labels, arrival, s, pmap, cfg.sipm, rng, k2, t_end=cfg.window[1])
    timers["sipm_s"] = time.perf_counter() - start
    timers["charges"] = len(charges["time"])
    return charges


def reconstruct(charges, cond, cfg, rng, geom, ccfg, selection, timers):
    """Keep the full selection when charge production or zero suppression leaves no digis."""
    from .. import chain
    outs = []
    if len(charges["time"]):
        digis = chain.digis(charges, cond, cfg, rng, timers=timers)
        if len(digis["chan"]):
            waveforms = chain.reco_waveforms(digis, cond)
            outs, _ = chain.clusters(waveforms, cond, geom, ccfg, cfg, timers=timers, detail=True)
    else:
        timers.update(digis=0, pulses=0)
    return coincidence_rows(outs, geom, selection)


def load_setup(gdml, calib, deviation, library_dir=None, device_refit=False):
    """Load the event-independent setup of the C4 chain once: run configuration, conditions, transport world,
    optical tables, SiPM photon map and the optional library. Draws no random numbers; pass the result to run()
    (``setup=``) to process several selections in one process without reloading (crvfastsim.veto.stream)."""
    from .. import chain, readout, runconfig, transport
    from ..response import photons, sipm
    start = time.perf_counter()
    rc = runconfig.get("run1a_v01")
    geom, ccfg = rc.geometry(), rc.counters_cfg()
    cfg = rc.chain_config(jax=True)
    if device_refit:
        import dataclasses
        cfg = dataclasses.replace(cfg, device_refit=True)
    cond = chain.Conditions.from_csv(calib, deviation)
    cond = readout.apply(cond, geom, ccfg)
    world, _ = transport.load_world(gdml, banded=True, y_plane=transport.msc_plane(geom))
    below = transport.load_below(gdml, geom, margin=rc.below_margin)
    from ..response.photons_jax import DeviceTable
    from ..response import lookup                 # pinned by the run configuration's asset set
    with lookup.pinned(rc.asset_pin_map() if hasattr(rc, "asset_pin_map") else None):
        tabs = {k: DeviceTable(v) for k, v in photons.load_tables(cfg.response).items()}
        pmap = sipm.photon_map()
    lib = spec = None
    if library_dir:
        from .. import tier1b_assets
        directory = Path(library_dir)
        options = dict(context=True, contain=True, point=True, topology=True, unvisited=True,
                       nearmiss=True, chord=True, anchor=True, per_primary=True)
        spec = tier1b_assets.Tier1bSpec(None, str(directory / "library.root"),
                                      str(directory / "topology.npz"), str(directory / "templates_v3.npz"),
                                      options=options)
        lib = tier1b_assets.load(spec, geom)
    return dict(inputs=(gdml, calib, deviation, library_dir), rc=rc, geom=geom, ccfg=ccfg, cfg=cfg, cond=cond,
                world=world, below=below, tabs=tabs, pmap=pmap, lib=lib, spec=spec,
                setup_s=time.perf_counter() - start)


def run(selection, primaries, gdml, calib, deviation, seed=20261105, period=1695., library_dir=None,
        input_kind="represented", census=None, setup=None):
    """Compose the fixed response chain with represented muons or independently retained generators.
    setup: the result of load_setup() for the same gdml/calib/deviation/library_dir (default: load it here)."""
    if len(selection) == 0:
        raise ValueError("response chain requires a nonempty event selection")
    input_report = {}
    if input_kind == "retained":
        from .generators import prepare
        if census is None:
            raise ValueError("retained inputs require an independent event census")
        primaries, input_report = prepare(selection, primaries, census)
        if (primaries.pdg.abs() != 13).any() and not library_dir:
            raise ValueError("non-muon retained inputs require the shower library; cannot silently omit their response")
    gen = rekey(selection, primaries, input_kind)
    from .. import tier1b, transport
    start = time.perf_counter()
    if setup is None:
        setup = load_setup(gdml, calib, deviation, library_dir)
        reused = False
    elif setup["inputs"] != (gdml, calib, deviation, library_dir):
        raise ValueError("prepared C4 setup was loaded for other geometry, conditions or library")
    else:
        reused = True
    rc, geom, ccfg, cfg, cond = (setup[k] for k in ("rc", "geom", "ccfg", "cfg", "cond"))
    world, below, tabs, pmap, lib, spec = (setup[k] for k in ("world", "below", "tabs", "pmap", "lib", "spec"))
    if lib is not None and input_kind == "retained":
        from .generators import shower_support
        input_report["shower_support"] = shower_support(gen, lib)
    timers = {"setup_s": time.perf_counter() - start}
    if reused:
        timers["setup_reused"] = True
    rng = np.random.default_rng(seed)
    begin = time.perf_counter()
    steps, parts = transport.transport(gen, geom, rng, stage="C", world=world, below=below,
                                       tagger_bar=rc.tagger_bar(geom), bar_cut=rc.bar_cut(geom),
                                       range_out=True, below_msc=rc.below_msc)
    timers["transport_s"] = time.perf_counter() - begin
    library_report = None
    if lib is not None:
        begin = time.perf_counter()
        steps, parts, library_report = tier1b.sample(steps, parts, gen, lib, geom, rng, showers=True, **spec.options)
        timers["tier1b_s"] = time.perf_counter() - begin
    steps = bunch_frame(steps, selection)
    charges = onspill_charges(steps, parts, geom, cond, cfg, rng, tabs, pmap, timers, period)
    fast = reconstruct(charges, cond, cfg, rng, geom, ccfg, selection, timers)
    elapsed = time.perf_counter() - start
    report = dict(label="run1a_v01 library on represented primaries" if library_dir else "primary-only placeholder",
                  library_dir=library_dir, selected_events=len(selection), primary_muons=int((gen.pdg.abs() == 13).sum()),
                  primary_particles=len(gen), nonmuon_primaries=int((gen.pdg.abs() != 13).sum()), input_kind=input_kind,
                  steps=len(steps), clusters=int(ak.sum(ak.num(fast.coincs))), elapsed_s=elapsed, timers=timers,
                  selected_events_per_s=len(selection) / elapsed, seed=seed, microbunch_period_ns=period,
                  limitations=["Represented genealogy muons, not a complete generated-primary census.",
                               "Secondary response uses a field-off library." if library_dir else "No tier-1b secondaries or shower response.",
                               "Straight-line material transport; DS field deflection is not modelled.",
                               "Run 1A named response and saved reference calibration, not the production-era response.",
                               "Onspill photon folding and bunch-frame transform; not a digitisation validation.",
                               "SiPM pre-window history uses the existing chain approximation, not the stock start margin."])
    if input_kind == "retained":
        report["label"] = "run1a_v01 library on retained upstream generators" if library_dir else "retained-muon transport placeholder"
        report["inputs"] = input_report
        report["library_sampling"] = library_report
        report["limitations"][0] = "Retained compressed generators, not a complete original generated stream."
        report["limitations"].append("Non-muon showers use species/energy donor sampling, not input position/direction conditioning.")
    return fast, report


def check_prepared_split(path, split, split_at):
    """Check adjacent preparation metadata when present; legacy inputs still face the empty-event guard."""
    if not path.exists():
        return
    summary = json.loads(path.read_text())
    prepared = summary.get("prepared_split")
    if prepared is None or prepared == "all":
        return
    if prepared not in ("dev", "heldout"):
        raise ValueError("unknown prepared split in inputs.json")
    if split != "all" and (split != prepared or split_at != summary.get("split_at")):
        raise ValueError(f"requested split={split!r} at shard {split_at} conflicts with prepared "
                         f"split={prepared!r} at shard {summary.get('split_at')}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("selection", "primaries", "gdml", "calib", "deviation", "outdir"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--input-kind", choices=("represented", "retained"), default="represented")
    parser.add_argument("--generator-events", help="independent event census required for retained generators")
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--period", type=float, default=1695.)
    parser.add_argument("--library-dir", help="optional run1a_v01 library and sidecars; requires per-primary sampler")
    parser.add_argument("--split", choices=("all", "dev", "heldout"), default="all")
    parser.add_argument("--split-at", type=int, default=22, help="first held-out source-file shard")
    parser.add_argument("--compile-cache", metavar="DIR",
                        help="JAX persistent compilation cache directory (crvfastsim.compile_cache); default off")
    args = parser.parse_args(argv)
    if args.compile_cache:
        from ..compile_cache import enable
        print(f"# compile cache: {enable(args.compile_cache)}", flush=True)
    selection, primaries = ak.from_parquet(args.selection), pd.read_parquet(args.primaries)
    if args.input_kind == "retained" and not args.generator_events:
        parser.error("--input-kind retained requires --generator-events")
    census = pd.read_parquet(args.generator_events) if args.generator_events else None
    from .generators import select_split
    if args.input_kind == "retained":
        check_prepared_split(Path(args.selection).with_name("inputs.json"), args.split, args.split_at)
    selection = select_split(selection, args.split, args.split_at)
    if args.split != "all" and args.input_kind == "represented":
        keep = primaries.shard < args.split_at
        primaries = primaries[keep] if args.split == "dev" else primaries[~keep]
    if args.input_kind == "retained":
        from .generators import prepare
        primaries, _ = prepare(selection, primaries, census)
    fast, report = run(selection, primaries, args.gdml, args.calib, args.deviation,
                       args.seed, args.period, args.library_dir, args.input_kind, census)
    report["split"] = args.split
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(selection, out / "selection.parquet")
    ak.to_parquet(fast, out / "fast.parquet")
    if args.input_kind == "retained":
        primaries.to_parquet(out / "generators.parquet", index=False)
        rekey(selection, primaries, "retained").to_parquet(out / "generator_map.parquet", index=False)
    (out / "chain.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
