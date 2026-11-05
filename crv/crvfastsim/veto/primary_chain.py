"""Primary-only C4 placeholder on represented muons and the cached stock selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import awkward as ak
import numpy as np
import pandas as pd

from .paired import KEY


def rekey(selection, primaries):
    """Use dense internal event IDs so labels cannot alias across runs or subruns."""
    events = pd.DataFrame({k: ak.to_numpy(selection[k]) for k in KEY})
    events["dense"] = np.arange(1, len(events) + 1)
    gen = primaries.merge(events, on=list(KEY), how="left", validate="many_to_one")
    if gen.dense.isna().any():
        raise ValueError("primary event absent from cached selection")
    gen["run"], gen["subrun"], gen["event"] = 1, 0, gen.pop("dense").to_numpy(np.int64)
    gen["idx"] = gen.groupby(list(KEY)).cumcount()
    from mu2efastsim.eloss import MMU
    gen["e"] = np.sqrt(gen.px**2 + gen.py**2 + gen.pz**2 + MMU**2)
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


def run(selection, primaries, gdml, calib, deviation, seed=20261105, period=1695., library_dir=None):
    """Compose existing transport/response/reco without tier-1b or regenerated primaries."""
    from .. import chain, readout, runconfig, transport
    from ..response import photons, sipm
    start = time.perf_counter()
    rc = runconfig.get("run1a_v01")
    geom, ccfg = rc.geometry(), rc.counters_cfg()
    cfg = rc.chain_config(jax=True)
    cond = chain.Conditions.from_csv(calib, deviation)
    cond = readout.apply(cond, geom, ccfg)
    world, _ = transport.load_world(gdml, banded=True, y_plane=transport.msc_plane(geom))
    below = transport.load_below(gdml, geom, margin=rc.below_margin)
    from ..response.photons_jax import DeviceTable
    tabs = {k: DeviceTable(v) for k, v in photons.load_tables(cfg.response).items()}
    pmap = sipm.photon_map()
    lib = spec = None
    if library_dir:
        from .. import tier1b, tier1b_assets
        directory = Path(library_dir)
        options = dict(context=True, contain=True, point=True, topology=True, unvisited=True,
                       nearmiss=True, chord=True, anchor=True, per_primary=True)
        spec = tier1b_assets.Tier1bSpec(None, str(directory / "library.root"),
                                      str(directory / "topology.npz"), str(directory / "templates_v3.npz"),
                                      options=options)
        lib = tier1b_assets.load(spec, geom)
    timers = {"setup_s": time.perf_counter() - start}
    gen = rekey(selection, primaries)
    rng = np.random.default_rng(seed)
    begin = time.perf_counter()
    steps, parts = transport.transport(gen, geom, rng, stage="C", world=world, below=below,
                                       tagger_bar=rc.tagger_bar(geom), bar_cut=rc.bar_cut(geom),
                                       range_out=True, below_msc=rc.below_msc)
    timers["transport_s"] = time.perf_counter() - begin
    if lib is not None:
        begin = time.perf_counter()
        steps, parts, _ = tier1b.sample(steps, parts, gen, lib, geom, rng, showers=True, **spec.options)
        timers["tier1b_s"] = time.perf_counter() - begin
    steps = bunch_frame(steps, selection)
    charges = onspill_charges(steps, parts, geom, cond, cfg, rng, tabs, pmap, timers, period)
    fast = reconstruct(charges, cond, cfg, rng, geom, ccfg, selection, timers)
    elapsed = time.perf_counter() - start
    report = dict(label="run1a_v01 library on represented primaries" if library_dir else "primary-only placeholder",
                  library_dir=library_dir, selected_events=len(selection), primary_muons=len(gen),
                  steps=len(steps), clusters=int(ak.sum(ak.num(fast.coincs))), elapsed_s=elapsed, timers=timers,
                  selected_events_per_s=len(selection) / elapsed, seed=seed, microbunch_period_ns=period,
                  limitations=["Represented genealogy muons, not a complete generated-primary census.",
                               "Secondary response uses a field-off library." if library_dir else "No tier-1b secondaries or shower response.",
                               "Straight-line material transport; DS field deflection is not modelled.",
                               "Run 1A named response and saved reference calibration, not the production-era response.",
                               "Onspill photon folding and bunch-frame transform; not a digitisation validation.",
                               "SiPM pre-window history uses the existing chain approximation, not the stock start margin."])
    return fast, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("selection", "primaries", "gdml", "calib", "deviation", "outdir"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--period", type=float, default=1695.)
    parser.add_argument("--library-dir", help="optional run1a_v01 library and sidecars; requires per-primary sampler")
    parser.add_argument("--split", choices=("all", "dev", "heldout"), default="all")
    parser.add_argument("--split-at", type=int, default=22, help="first held-out source-file shard")
    args = parser.parse_args(argv)
    selection, primaries = ak.from_parquet(args.selection), pd.read_parquet(args.primaries)
    if args.split != "all":
        use = selection.shard < args.split_at
        keep = primaries.shard < args.split_at
        selection, primaries = (selection[use], primaries[keep]) if args.split == "dev" else (selection[~use], primaries[~keep])
    fast, report = run(selection, primaries, args.gdml, args.calib, args.deviation,
                       args.seed, args.period, args.library_dir)
    report["split"] = args.split
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(selection, out / "selection.parquet")
    ak.to_parquet(fast, out / "fast.parquet")
    (out / "chain.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
