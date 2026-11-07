"""Opt-in four-arm saved-production comparison; no transport, tuning or library building.

Requires explicit confirmation of stock on-spill phase-zero/default timing. All source
rows enter reconstruction; only direct-front queries enter sampled-arm comparisons.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import uproot

from ..artsteps import onspill_timing
from .export import EVENT, columns, join, keys, read
from .library import load, sample
from .response import place
from .crystal_placement import place_crystals

ARMS = ("control", "geometric", "crystal")
TIMING = dict(initial_phase_shift=0., time_from_protons_to_dr_marker=200., clock_tick=25.,
              daq_ticks=68, onspill_bins=5, dr_period=1695., crystal_length=200.)


def check_geometry(library, geometry):
    """Compare every saved physical field, sorting by ID and ignoring run labels only."""
    checked = []
    for prefix, identity in (("crystal", "crystal"), ("disk", "disk"), ("vd", "id")):
        names = {k for k in geometry if k.startswith(prefix + "_") and k != prefix + "_run"}
        names |= {k.removeprefix("geometry_") for k in library
                  if k.startswith("geometry_" + prefix + "_") and k != "geometry_" + prefix + "_run"}
        a, b = geometry[prefix + "_" + identity], library["geometry_" + prefix + "_" + identity]
        if len(np.unique(a)) != len(a) or len(np.unique(b)) != len(b) or not np.array_equal(np.sort(a), np.sort(b)):
            raise ValueError(prefix + " geometry identities differ or span duplicate runs")
        for name in sorted(names):
            if name not in geometry or "geometry_" + name not in library:
                raise ValueError("missing geometry field: " + name)
            x, y = geometry[name][np.argsort(a)], library["geometry_" + name][np.argsort(b)]
            equal = np.array_equal(x, y) if x.dtype.kind in "biu" else np.allclose(x, y, atol=1e-6, rtol=0)
            if x.shape != y.shape or not equal:
                raise ValueError("physical geometry mismatch: " + name)
            checked.append(name)
    return checked


def check_independence(tables, library, metadata, ids):
    """Full IDs when saved; legacy pilot row indices are NOT persistent event IDs."""
    source = metadata.get("input")
    if source and Path(source).resolve() == Path(tables).resolve():
        raise ValueError("library source is the comparison file")
    candidates = [container[name] for container in (library, metadata)
                  for name in ("source_full_ids", "source_event_ids", "full_event_ids") if name in container]
    for container in (library, metadata):
        for prefix in ("source_", "entry_"):
            if all(prefix + k in container for k in EVENT):
                candidates.append(np.column_stack([container[prefix + k] for k in EVENT]))
    for value in candidates:
        value = np.asarray(value)
        if value.dtype.names and all(k in value.dtype.names for k in EVENT):
            other = keys(value, EVENT)
        elif value.ndim == 2 and value.shape[1] == 3:
            other = keys(dict(zip(EVENT, value.T)), EVENT)
        else:
            raise ValueError("malformed library full event identities")
        if np.intersect1d(ids, other).size:
            raise ValueError("library/comparison full event identity overlap")
    if not candidates and not source:
        raise ValueError("library needs source-file provenance or full source IDs")
    return ("Available full source IDs checked: no overlap." if candidates else
            "Distinct source-file provenance only; legacy pilot lacks full IDs. No integer-event disjointness claim.")


def prepare(tables, library, sim, *, metadata=None, seed=20261105, stock_matched_phase0=False):
    """Return entries, timing, full_ids, steps_by_arm, geometry and audit in a dict.

    library is a loaded numeric mapping; metadata is its saved JSON mapping. No writes
    or reconstruction. One disk-oriented draw feeds BOTH placement variants.
    """
    if not stock_matched_phase0:
        raise ValueError("explicit stock-matched phase-zero/default timing confirmation required")
    entries, _, geometry, census, _ = read(tables, input_mode="primary")
    with uproot.open(tables) as source:
        events = columns(source, "events", EVENT + ("n_showers",))
        css = columns(source, "showers", EVENT + ("crystal", "energy_g4", "energy_birks", "time", "position_z", "index"))
        for prefix, table in (("crystal", "crystals"), ("disk", "disks"), ("vd", "virtualdetectors")):
            extra = [k.removeprefix("geometry_" + prefix + "_") for k in library
                     if k.startswith("geometry_" + prefix + "_") and k.removeprefix("geometry_") not in geometry]
            geometry.update({prefix + "_" + k: v for k, v in columns(source, table, extra).items()} if extra else {})
    ids = keys(events, EVENT)
    audit = dict(census=census, geometry_fields=check_geometry(library, geometry),
                 independence=check_independence(tables, library, metadata or {}, ids), timing=TIMING)
    row = join(ids, keys(css, EVENT))
    counts = np.bincount(row, minlength=len(ids))
    if np.any(np.diff(row) < 0) or not np.array_equal(counts, events["n_showers"]):
        raise ValueError("ordered whole-source CSS census mismatch")
    if not np.array_equal(css["index"], np.arange(len(row)) - np.repeat(np.cumsum(counts) - counts, counts)):
        raise ValueError("CSS collection order mismatch")
    crystal = css["crystal"].astype(np.int64)
    if np.any(crystal < 0) or np.any(crystal >= len(sim["lru"])):
        raise ValueError("CSS crystal outside calibration")
    steps = dict(event=row, crystal=crystal, energy_g4=css["energy_g4"], energy_birks=css["energy_birks"],
                 time=css["time"], z_local=css["position_z"], lru=sim["lru"][crystal], yields=sim["pe_per_mev"][crystal])
    draw = sample(library, entries, np.random.default_rng(seed), orientation="disk")
    from .reconstruct import steps_for_reco
    placed = {"geometric": place(library, draw), "crystal": place_crystals(library, entries, draw)}
    by_arm = {"control": steps, **{k: steps_for_reco(dict(entries=entries, steps=v), sim) for k, v in placed.items()}}
    total = float(draw["energy_birks"].sum())
    audit["sampling"] = dict(fallback=int(draw["fallback"].sum()), unsupported=int((~draw["supported"]).sum()),
        empty_donors=int(np.sum(draw["supported"] & (np.diff(draw["offsets"]) == 0))),
        unmapped_energy_percent={k: 100 * float(v["energy_birks"][~v["mapped"]].sum()) / total if total else None
                                 for k, v in placed.items()})
    return dict(entries=entries, timing=onspill_timing(events["event"], **TIMING), full_ids=ids,
                steps_by_arm=by_arm, geometry=geometry, audit=audit,
                sampling={k: draw[k] for k in ("donor", "supported", "fallback", "offsets")})


def _json(path, value):
    """Unavailable nonfinite statistics remain null, never filtered event rows."""
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)) and not np.isfinite(x):
            return None
        return x.item() if isinstance(x, np.generic) else x
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def diagnostics(prepared, clusters):
    from .reconstruct import report
    from .compare import paired_mean
    from .validity_report import observables, compare_observables
    from ..readout import ReadoutConfig
    from ..response import fold_step_time
    entry, timing, geom = (prepared[k] for k in ("entries", "timing", "geometry"))
    rows = entry["event_row"]
    origin = np.column_stack([geom["disk_origin_" + a] for a in "xy"])
    axes = np.stack([np.column_stack([geom["disk_axis_" + a + "_" + b] for b in "xyz"]) for a in "xyz"], axis=2)
    if not np.allclose(origin, origin[:1], atol=1e-6, rtol=0) or not np.allclose(axes, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError("incident-relative diagnostics require common transverse origin and aligned axes")
    cfg = ReadoutConfig()
    folded, _ = fold_step_time(entry["time"], **{k: timing[k][rows] for k in ("pbtime", "mbtime", "event_length", "onspill")},
        time_from_protons_to_dr_marker=timing["marker"][rows], digitization_start=cfg.digitization_start,
        digitization_end=cfg.digitization_end, digitization_buffer=cfg.digitization_buffer)
    selected = {name: {k: v[rows] for k, v in value.items()} for name, value in clusters.items()}
    obs = {k: observables(v, entry, folded, origin[0]) for k, v in selected.items()}
    common = np.logical_and.reduce([v["energy"] > 10 for v in selected.values()])
    masks = {"disk0": entry["disk"] == 0, "disk1": entry["disk"] == 1}
    masks.update({f"radius_{lo}_{hi}": (entry["radius"] >= lo) & (entry["radius"] < hi)
                  for lo, hi in ((0, 450), (450, 550), (550, 1000), (1000, np.inf))})
    result = {}
    for name in ARMS:
        fast, stock = selected[name], selected["stockref"]
        distance = np.hypot(fast["x"][common] - stock["x"][common], fast["y"][common] - stock["y"][common])
        result[name] = dict(paired=report(fast, stock, rows), observables=compare_observables(obs[name], obs["stockref"], rows),
            strata={k: report({f: v[m] for f, v in fast.items()}, {f: v[m] for f, v in stock.items()}, rows[m])
                    for k, m in masks.items()}, common_selected=int(common.sum()),
            common_distance_mm=paired_mean(distance, np.zeros(len(distance)), rows[common] // 100))
    return result


def main():
    from .. import chain, stockio
    from ..cluster import Geometry
    from .prepare_chain import stock_clusters
    from .reconstruct import leading, report
    from .compare import plot
    import jax
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("tables", "library", "stock-ntuple", "reco-inputs", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--reco-seed", type=int, default=20261108)
    p.add_argument("--prepare-only", action="store_true", help="check geometry, timing and frozen sampling; do not reconstruct")
    p.add_argument("--stock-matched-phase0", action="store_true", required=True,
                   help="confirm stock on-spill timing matches phase=0, marker=200, tick=25, ticks=68, bins=5, period=1695, length=200")
    args = p.parse_args()
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    if jax.default_backend() != "cpu":
        raise RuntimeError("CPU reconstruction required")
    library, metadata = load(args.library)
    sim = stockio.load_sim_crystal(args.reco_inputs / "Sim_crystal.txt")
    prepared = prepare(args.tables, library, sim, metadata=metadata, seed=args.seed, stock_matched_phase0=args.stock_matched_phase0)
    ids, rows = prepared["full_ids"], prepared["entries"]["event_row"]
    if args.prepare_only:
        args.out.mkdir(parents=True, exist_ok=True)
        _json(args.out / "preparation.json", prepared["audit"])
        print(json.dumps(prepared["audit"], indent=2, allow_nan=False))
        return
    clusters = {"stockref": stock_clusters(args.stock_ntuple, ids)}
    geometry = Geometry.load(args.tables)
    pulse = stockio.load_pulse(args.reco_inputs / "CsI-waveform-2020-02-12.root")
    noise = stockio.load_noise(args.reco_inputs / "ReadoutNoise.root")
    propagation = stockio.load_propagation(args.reco_inputs / "CsI-propag-2020-07-29.root")
    args.out.mkdir(parents=True, exist_ok=True)
    saved = dict(event_row=np.arange(len(ids)), query_event_rows=rows, full_ids=ids)
    saved.update({"entry_" + k: v for k, v in prepared["entries"].items()})
    saved.update({"timing_" + k: v for k, v in prepared["timing"].items()})
    saved.update({"sampling_" + k: v for k, v in prepared["sampling"].items()})
    saved.update({"geometry_" + k: v for k, v in prepared["geometry"].items()})
    runs = []
    for name in ARMS:
        jax.effects_barrier()
        start = time.perf_counter()
        reco = chain.run_chain(prepared["steps_by_arm"][name], prepared["timing"], lambda sipm: stockio.adc_per_pe(sim, sipm),
            pulse, noise, geometry, np.random.default_rng(args.reco_seed), table=propagation, adc2mev=stockio.adc_to_mev)
        jax.effects_barrier()
        elapsed = time.perf_counter() - start
        clusters[name] = leading(reco["clusters"], len(ids))
        saved.update({arm + "_" + k: v for arm, values in clusters.items() for k, v in values.items()})
        np.savez_compressed(args.out / "clusters.npz", **saved)
        runs.append(dict(arm=name, seconds=elapsed, stages=reco["seconds"], n_refit_skipped=reco["n_refit_skipped"],
            n_nonfinite_reco=reco["n_nonfinite_reco"], nonfinite_examples=reco.get("nonfinite_examples"),
            leading_nonfinite={k: int((~np.isfinite(v)).sum()) for k, v in clusters[name].items()}))
        _json(args.out / "timing.json", runs)
        print(json.dumps(runs[-1]), flush=True)
    result = dict(audit=prepared["audit"], arms=diagnostics(prepared, clusters), runs=runs,
        original_control_all_rows=report(clusters["control"], clusters["stockref"], np.arange(len(ids))),
        provenance={k: str(v) for k, v in vars(args).items()},
        scope="Stored population; sampled arms report ONLY common direct-front queries, not non-query acceptance. "
              "One call per arm; current native cache/default 15 iterations; no performance claim. "
              "Incident state/truth E/p are simulation diagnostics. Paired 100-row block errors exclude library uncertainty.")
    _json(args.out / "numbers.json", result)
    summary = {name: dict(selected=value["paired"]["fast_selected"],
        stock_selected=value["paired"]["stock_selected"], ratio=value["paired"]["acceptance_ratio"],
        energy=value["paired"]["energy_mean"], time=value["paired"].get("paired_time_ns"),
        common_distance_mm=value["common_distance_mm"], common_selected=value["common_selected"],
        shapes={k: {field: v[field] for field in ("chi2", "ndf") if field in v}
                for k, v in value["observables"].items()},
        strata={k: dict(entries=v["events"], ratio=v["acceptance_ratio"], energy=v["energy_mean"])
                for k, v in value["strata"].items()}) for name, value in result["arms"].items()}
    _json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    for name in ARMS:
        hist = result["arms"][name]["observables"]
        _json(args.out / (name + "_histograms.json"), hist)
        plot(hist, args.out / (name + ".png"), name + ": conditional production front entries", labels=("Stock reconstruction", name))


if __name__ == "__main__":
    main()
