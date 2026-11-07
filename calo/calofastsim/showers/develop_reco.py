"""Development-only saved cross-fit CSS replay through the unmodified current CAL1 backend."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import time

import numpy as np

from .export import EVENT, keys
from .prepare_chain import stock_clusters
from .reconstruct import leading, load_chain, report


ARMS = ("stock", "baseline", "crystal", "direction", "combined")
SOURCE_ROWS = 1000


def steps_for_backend(source, sim, query_rows=None):
    """Validate canonical saved steps, retaining source row IDs rather than compacting queries."""
    q, event, crystal, keep = (np.asarray(source[k]) for k in ("query_event_rows", "event_row", "crystal", "mapped"))
    for name, value in (("query", q), ("event", event), ("crystal", crystal)):
        if value.ndim != 1 or value.dtype.kind not in "iu":
            raise ValueError(f"{name} IDs must be one-dimensional integers")
    if np.any((q < 0) | (q >= SOURCE_ROWS)) or len(np.unique(q)) != len(q):
        raise ValueError("query IDs must be unique development rows in [0, 1000)")
    if query_rows is not None and not np.array_equal(q, query_rows):
        raise ValueError("arms must have identical ordered query IDs")
    if keep.dtype.kind != "b" or keep.shape != event.shape or crystal.shape != event.shape:
        raise ValueError("mapped, crystal and event arrays must be aligned; mapped must be boolean")
    if not np.all(np.isin(event, q)):
        raise ValueError("step refers to a non-query event")
    fields = {k: np.asarray(source[k]) for k in ("energy_g4", "energy_birks", "time", "local_position")}
    for name, value in fields.items():
        if value.shape != ((len(event), 3) if name == "local_position" else event.shape):
            raise ValueError(f"invalid saved {name} shape")
        if not np.all(np.isfinite(value[keep])):
            raise ValueError(f"nonfinite mapped {name}")
    c = crystal[keep]
    if np.any(c < 0) or np.any(c >= len(sim["lru"])) or np.any(c >= len(sim["pe_per_mev"])):
        raise ValueError("mapped crystal has no calibration")
    steps = {k: v[keep] for k, v in fields.items() if k != "local_position"}
    steps.update(event=event[keep], crystal=c, z_local=fields["local_position"][keep, 2],
                 lru=sim["lru"][c], yields=sim["pe_per_mev"][c])
    if not all(np.all(np.isfinite(steps[k])) for k in ("lru", "yields")):
        raise ValueError("nonfinite crystal calibration")
    return steps, q


def development_inputs(path, stock_path, stock_path2):
    """Join both stock seeds on ALL full event identities before slicing development rows."""
    with np.load(path, allow_pickle=False) as source:
        conditions = {k.removeprefix("condition_"): source[k] for k in source.files if k.startswith("condition_")}
    if any(v.shape != (2000,) for v in conditions.values()):
        raise ValueError("prepared inputs must contain 2000 aligned event conditions")
    ids = keys(conditions, EVENT)
    if len(ids) != 2000 or len(np.unique(ids)) != len(ids):
        raise ValueError("prepared inputs must contain 2000 unique full event identities")
    stocks = [{k: v[:SOURCE_ROWS] for k, v in stock_clusters(p, ids).items()} for p in (stock_path, stock_path2)]
    timing = {k: conditions[k][:SOURCE_ROWS] for k in ("pbtime", "mbtime", "event_length", "marker", "crystal_length")}
    timing["onspill"] = conditions["spill"][:SOURCE_ROWS] == 1
    return timing, stocks


def energy_summary(values):
    """Physical-MeV summaries on selected clusters; no small-sample peak fit."""
    energy = np.asarray(values)
    selected = energy[np.isfinite(energy) & (energy > 10)]
    return dict(selected_above10=len(selected), quantile_probabilities=[.05, .16, .5, .84, .95],
                quantiles_mev=np.quantile(selected, [.05, .16, .5, .84, .95]).tolist() if len(selected) else None,
                fraction_below70_given_above10=float(np.mean(selected < 70)) if len(selected) else None,
                fraction_above100_given_above10=float(np.mean(selected > 100)) if len(selected) else None)


def replay(chain, io, sim, pulse, noise, geometry, propagation, steps, timing, *, repeat, seed, synchronize):
    """Time same-seed calls and require exact leading E/t/xyz reproducibility, including NaNs."""
    if repeat < 2:
        raise ValueError("at least one cold and one warm repeat required")
    first, runs = None, []
    for index in range(repeat):
        synchronize()
        start = time.perf_counter()
        reco = chain.run_chain(steps, timing, lambda sipm: io.adc_per_pe(sim, sipm), pulse, noise, geometry,
                               np.random.default_rng(seed), table=propagation, adc2mev=io.adc_to_mev)
        synchronize()
        elapsed = time.perf_counter() - start
        fast = leading(reco["clusters"], SOURCE_ROWS)
        if first is None:
            first = fast
        elif any(not np.array_equal(first[k], fast[k], equal_nan=True) for k in first):
            raise RuntimeError("same-seed reconstruction repeat differs in leading E/t/xyz")
        runs.append(dict(repeat=index, phase="cold" if index == 0 else "warm", reco_seconds=elapsed,
                         stage_seconds=reco["seconds"], selected=int((fast["energy"] > 10).sum()),
                         n_refit_skipped=int(reco["n_refit_skipped"]), n_nonfinite_reco=int(reco["n_nonfinite_reco"])))
        print(json.dumps(runs[-1]), flush=True)
    return first, runs


def main(argv=None):
    import jax

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("steps", "inputs", "reco-code", "reco-inputs", "out", "stock-ntuple", "stock-ntuple2"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS[:3]))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--save-clusters", action="store_true", help="save first-repeat 1000-row arrays, including both stock seeds")
    args = parser.parse_args(argv)
    if args.repeat < 2 or len(set(args.arms)) != len(args.arms):
        parser.error("require repeat >= 2 and unique arms")
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != args.device:
        raise RuntimeError("requested device differs from JAX backend")
    timing, stocks = development_inputs(args.inputs, args.stock_ntuple, args.stock_ntuple2)
    chain, io, cluster = load_chain(args.reco_code)
    template = importlib.import_module(chain.__package__ + ".templatefit")
    cfg = chain.run_chain.__kwdefaults__["fit_cfg"]
    if cfg.iterations != 15 or chain.fit_digis is not template.fit_digis or not hasattr(template, "_CACHE"):
        raise ValueError("require native current CAL1 template-fit cache and default 15 iterations")
    backend = dict(source=str(args.reco_code.resolve()), fit_config=asdict(cfg), native_cache=True, fit_adapter=False)
    sim = io.load_sim_crystal(args.reco_inputs / "Sim_crystal.txt")
    pulse = io.load_pulse(args.reco_inputs / "CsI-waveform-2020-02-12.root")
    noise = io.load_noise(args.reco_inputs / "ReadoutNoise.root")
    propagation = io.load_propagation(args.reco_inputs / "CsI-propag-2020-07-29.root")
    geometry = cluster.Geometry.load(args.reco_inputs / "calo_geometry_run1430.npz")
    prepared, query = {}, None
    for arm in args.arms:
        with np.load(args.steps / (arm + "_steps.npz"), allow_pickle=False) as source:
            prepared[arm], query = steps_for_backend(source, sim, query)
    if not len(query):
        raise ValueError("no development queries to report")
    args.out.mkdir(parents=True, exist_ok=True)
    arrays, results = {}, {}
    reference, control = ({k: v[query] for k, v in stock.items()} for stock in stocks)
    for arm, steps in prepared.items():
        jax.clear_caches()
        template._CACHE.clear()
        first, runs = replay(chain, io, sim, pulse, noise, geometry, propagation, steps, timing,
                             repeat=args.repeat, seed=args.seed, synchronize=jax.effects_barrier)
        arrays[arm] = first
        fast = {k: v[query] for k, v in first.items()}
        warm = float(np.median([r["reco_seconds"] for r in runs[1:]]))
        results[arm] = dict(**report(fast, reference, query), energy_distribution=energy_summary(fast["energy"]),
                           repeats=runs, timing=dict(warm_reco_seconds=warm, source_rows=SOURCE_ROWS, queried_rows=len(query),
                           source_rows_per_second_conditional_css=SOURCE_ROWS / warm, queried_rows_per_second=len(query) / warm))
        (args.out / "timing.json").write_text(json.dumps({k: v["repeats"] for k, v in results.items()}, indent=2) + "\n")
    if "baseline" in arrays:
        baseline = {k: v[query] for k, v in arrays["baseline"].items()}
        for arm in arrays:
            results[arm]["vs_baseline"] = report({k: v[query] for k, v in arrays[arm].items()}, baseline, query)
    summary = dict(entries=len(query), source_rows=SOURCE_ROWS, seed=args.seed, backend=backend, arms=results,
                   stock2_vs_stock1=report(control, reference, query),
                   stock_energy_distributions={name: energy_summary(stock["energy"]) for name, stock in
                                               (("stock1", reference), ("stock2", control))},
                   caveat="CONDITIONAL development direct-front queries only (saved stock CSS has 505 queries); not a full-source stock CSS control. "
                          "All 1000 rows enter the backend; nonquery rows have no CSS and zero reconstructed energy. Rates are NOT generated-event throughput.",
                   timing_method="Backend-only synchronized wall time; native/JAX caches cleared before each arm; first cold then warm repeats; excludes IO/setup/CSS sampling.",
                   error_method="Paired 100-source-row block jackknife; cross-fit library uncertainty/cross-fold correlations excluded.",
                   device=[str(d) for d in jax.devices()], provenance={k: str(v) for k, v in vars(args).items()})
    (args.out / "numbers.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if args.save_clusters:
        arrays.update(stock1=stocks[0], stock2=stocks[1])
        np.savez_compressed(args.out / "clusters.npz", event_row=np.arange(SOURCE_ROWS), query_event_rows=query,
                            **{arm + "_" + k: v for arm, values in arrays.items() for k, v in values.items()})
    print(json.dumps(dict(entries=len(query), source_rows=SOURCE_ROWS, backend=backend, output=str(args.out))), flush=True)


if __name__ == "__main__":
    main()
