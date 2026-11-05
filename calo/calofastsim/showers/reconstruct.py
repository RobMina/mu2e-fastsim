"""calorimeter reconstruction on saved fast CE states, with all generated events retained."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from .compare import paired_hist, paired_mean, paired_ratio
from .handoff import from_cechain
from .library import load


def load_chain(path):
    """Load a caller-selected frozen calorimeter package separately from this package."""
    alias = "shower_reco_backend"
    spec = importlib.util.spec_from_file_location(alias, path / "__init__.py", submodule_search_locations=[str(path)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[alias] = package
    spec.loader.exec_module(package)
    return tuple(importlib.import_module(alias + "." + name) for name in ("chain", "stockio", "cluster"))


def steps_for_reco(handoff, sim):
    """Map sampled CSS query indices back to the full source-event row universe."""
    draw, query = handoff["steps"], handoff["entries"]
    keep = draw["mapped"]
    crystal = draw["crystal"][keep]
    return dict(event=query["event_row"][draw["entry"][keep]], crystal=crystal,
                energy_g4=draw["energy_g4"][keep], energy_birks=draw["energy_birks"][keep],
                time=draw["time"][keep], z_local=draw["local_position"][keep, 2],
                lru=sim["lru"][crystal], yields=sim["pe_per_mev"][crystal])


def leading(clusters, n):
    out = {k: np.full(n, np.nan) for k in ("energy", "time", "x", "y", "z")}
    out["energy"][:] = 0
    ev = clusters["event"]
    if len(ev):
        order = np.lexsort((-np.asarray(clusters["energy"]), ev))
        first = order[np.r_[True, np.diff(ev[order]) != 0]]
        for k in out:
            out[k][ev[first]] = clusters[k][first]
    return out


def report(fast, stock, rows):
    group = rows // 100
    selected_f, selected_s = fast["energy"] > 10, stock["energy"] > 10
    both = selected_f & selected_s
    result = dict(events=len(rows), fast_selected=int(selected_f.sum()), stock_selected=int(selected_s.sum()),
                  acceptance_difference=paired_mean(selected_f, selected_s, group),
                  acceptance_ratio=paired_ratio(selected_f, selected_s, group),
                  energy_mean=paired_mean(fast["energy"], stock["energy"], group),
                  energy_shape=paired_hist(fast["energy"], stock["energy"], group, np.arange(0., 111., 5.)),
                  both_selected=int(both.sum()), fast_only=int(np.sum(selected_f & ~selected_s)),
                  stock_only=int(np.sum(selected_s & ~selected_f)))
    if both.any():
        result["paired_time_ns"] = paired_mean(fast["time"][both], stock["time"][both], group[both])
        result["paired_position_distance_mm"] = float(np.mean(np.sqrt(sum((fast[k][both] - stock[k][both]) ** 2 for k in ("x", "y")))))
    return result


def main():
    from beamfastsim import cechain
    import jax

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("chain", "chain-report", "inputs", "library", "reco-code", "reco-inputs", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--maps", nargs="+", required=True)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--device", choices=("cpu", "gpu"), required=True)
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--reco-commit", required=True)
    p.add_argument("--live-transport", choices=("numpy", "device"), help="time the complete route in one call, not a stage sum")
    p.add_argument("--transport-geometry", type=Path)
    args = p.parse_args()
    if args.repeat < 2:
        p.error("at least one cold and one warm repeat required")
    if args.live_transport and args.transport_geometry is None:
        p.error("live transport requires its geometry")
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != args.device:
        raise RuntimeError(f"requested {args.device}; actual {jax.devices()}")
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.chain, allow_pickle=False) as f:
        saved = {k: f[k] for k in f.files}
    rows = saved["event_row"]
    with np.load(args.inputs, allow_pickle=False) as f:
        conditions = {k.removeprefix("condition_"): f[k][rows] for k in f.files if k.startswith("condition_")}
        stock = {k.removeprefix("stock_cluster_"): f[k][rows] for k in f.files if k.startswith("stock_cluster_")}
        starts = {k: f[k][rows] for k in ("x0", "u0", "p0", "t0")}
    if not stock:
        p.error("prepared inputs must include saved stock reconstruction")
    library, _ = load(args.library)
    field = cechain.field_maps(args.maps)
    chain, io, cluster = load_chain(args.reco_code)
    sim = io.load_sim_crystal(args.reco_inputs / "Sim_crystal.txt")
    pulse = io.load_pulse(args.reco_inputs / "CsI-waveform-2020-02-12.root")
    noise = io.load_noise(args.reco_inputs / "ReadoutNoise.root")
    propagation = io.load_propagation(args.reco_inputs / "CsI-propag-2020-07-29.root")
    geometry = cluster.Geometry.load(args.reco_inputs / "calo_geometry_run1430.npz")
    timing = {k: conditions[k] for k in ("pbtime", "mbtime", "event_length", "marker", "crystal_length")}
    timing["onspill"] = conditions["spill"] == 1
    result = SimpleNamespace(fate=saved["fate"], x=saved["tracker_x"], u=saved["tracker_u"],
                             T=saved["tracker_T"], t=saved["tracker_t"])
    if args.live_transport:
        transport_geometry = cechain.from_offline(str(args.transport_geometry), str(args.transport_geometry.parent))
        if args.live_transport == "device":
            from beamfastsim import cechain_jax
            device_field = cechain_jax.device_maps(args.maps, np.float64)
    runs, first = [], None
    for repeat in range(args.repeat):
        total_start = time.perf_counter()
        transport_seconds = None
        if args.live_transport:
            if args.live_transport == "device":
                result = cechain_jax.transport(jax.random.PRNGKey(args.seed), transport_geometry, device_field,
                                              **starts, dtype=np.float64, lanes=1024, shrink=32, unroll=args.device == "gpu")
            else:
                result = cechain.transport(np.random.default_rng(args.seed), transport_geometry, field, **starts, backend="numpy")
            jax.effects_barrier()
            transport_seconds = time.perf_counter() - total_start
            for name, field_name in (("fate", "fate"), ("x", "tracker_x"), ("u", "tracker_u"), ("T", "tracker_T"), ("t", "tracker_t")):
                if not np.array_equal(getattr(result, name), saved[field_name]):
                    raise RuntimeError("live timing repeat differs from the saved same-seed transport")
        start = time.perf_counter()
        handoff = from_cechain(result, field, library, np.random.default_rng(args.seed + 1))
        steps = steps_for_reco(handoff, sim)
        handoff_seconds = time.perf_counter() - start
        start = time.perf_counter()
        reco = chain.run_chain(steps, timing, lambda sipm: io.adc_per_pe(sim, sipm), pulse, noise, geometry,
                               np.random.default_rng(args.seed + 3), table=propagation, adc2mev=io.adc_to_mev)
        jax.effects_barrier()
        reco_seconds = time.perf_counter() - start
        total_seconds = time.perf_counter() - total_start
        fast = leading(reco["clusters"], len(rows))
        if first is None:
            first = fast
            np.savez_compressed(args.out / "clusters.npz", event_row=rows,
                                **{"fast_" + k: v for k, v in fast.items()}, **{"stock_" + k: v for k, v in stock.items()})
        elif not np.array_equal(first["energy"], fast["energy"]):
            raise RuntimeError("same-seed reconstruction repeat differs")
        runs.append(dict(repeat=repeat, handoff_seconds=handoff_seconds, reco_seconds=reco_seconds,
                         transport_seconds=transport_seconds, total_seconds=total_seconds,
                         downstream_seconds=handoff_seconds + reco_seconds, stage_seconds=reco.get("seconds"),
                         clusters=len(reco["clusters"]["event"]), selected=int((fast["energy"] > 10).sum()),
                         n_refit_skipped=reco["n_refit_skipped"], n_nonfinite_reco=reco["n_nonfinite_reco"]))
        (args.out / "timing.json").write_text(json.dumps(runs, indent=2) + "\n")
        print(json.dumps(runs[-1]), flush=True)
    parent = json.loads(args.chain_report.read_text())
    transport = float(np.median([r["transport_seconds"] for r in parent["timing"]["measurements"][1:]]))
    downstream = float(np.median([r["downstream_seconds"] for r in runs[1:]]))
    total = transport + downstream
    method = "sum of separately measured sequential stages on matching CPU/device allocations; excludes setup and compilation"
    if args.live_transport:
        transport = float(np.median([r["transport_seconds"] for r in runs[1:]]))
        total = float(np.median([r["total_seconds"] for r in runs[1:]]))
        method = "single-call CE birth through calorimeter clusters, host/device synchronization included; excludes setup and compilation"
    summary = report(first, stock, rows)
    summary.update(repeats=runs, timing=dict(warm_transport_seconds=transport, warm_downstream_seconds=downstream,
                   composed_chain_seconds=total, generated_per_second=len(rows) / total,
                   stock_grid_slot_equivalent=len(rows) / total / parent["timing"]["stock_generated_per_second"],
                   method=method),
                   reco_commit=args.reco_commit, device=[str(d) for d in jax.devices()],
                   comparisons="same generated primaries; leading cluster >10 MeV; zeros retained; independent stock/fast stochastic response",
                   error_method="paired 100-source-event block jackknife; library uncertainty excluded",
                   provenance={k: str(v) for k, v in vars(args).items()})
    (args.out / "numbers.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "energy_shape"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
