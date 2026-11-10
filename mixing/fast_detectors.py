"""Development mixed-step response on existing JAX tracker and CRV kernels, without art."""
import argparse
import json
import time
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from .counts import event_ids, event_keys, json_values
from .fast_tracker import STEP_DIRS, compare_reference, comparison, digi_columns
from .crv_compare import compare as compare_crv


def flatten_output(result, offset):
    from crvfastsim.chain import CHAN_MULT
    from crvfastsim.reco.ntview import concat
    d = result["D"]
    c, p = concat(result["clusters"])
    label = np.asarray(d["chan"], np.int64)
    D = {"ev": label // CHAN_MULT + offset, "bar": label % CHAN_MULT // 4,
         "sipm": label % 4, "starttdc": np.asarray(d["starttdc"], np.int64), "adc": d["adc"]}
    P = {**p, "ev": np.asarray(p["ev"], np.int64) + offset}
    C = {"ev": np.asarray(c["cl_event"], np.int64) + offset, "sectortype": c["cl_type"],
         "pe": c["cl_pe"], "time": c["cl_avgtime"], "poscalc": c["cl_poscalc"],
         "x": c["cl_pos"][:, 0], "y": c["cl_pos"][:, 1], "z": c["cl_pos"][:, 2]}
    return D, P, C


def join(parts):
    return tuple({key: np.concatenate([p[index][key] for p in parts]) for key in parts[0][index]}
                 for index in range(3))


def crv_pass(steps, pdg, timing, geom, counters, cond, cfg, tabs, pmap, seed, chunk):
    import jax
    from crvfastsim import onspill
    ev = timing.lookup(steps[["run", "subrun", "event"]].to_numpy())
    order = np.argsort(ev, kind="stable")
    ev = ev[order]
    result, metrics = [], {}
    start = time.perf_counter()
    for first in range(0, len(timing.keys), chunk):
        last = min(first + chunk, len(timing.keys))
        selected = order[np.searchsorted(ev, first):np.searchsorted(ev, last)]
        t = onspill.EventTiming(timing.keys[first:last], timing.mc_pbtime[first:last],
                               timing.reco_pbtime[first:last], timing.event_length[first:last], timing.period)
        rng = np.random.default_rng(np.random.SeedSequence([seed, first]))
        out = onspill.run(steps.iloc[selected], pdg[selected], geom, cond, cfg, rng, timing=t,
                         counters_cfg=counters, key=jax.random.fold_in(jax.random.key(seed), first),
                         tabs=tabs, pmap=pmap, timers={}, detail=True)
        result.append(flatten_output(out, first))
        for key, value in out["timers"].items():
            if isinstance(value, (int, float, np.number)):
                metrics[key] = metrics.get(key, 0) + value
        print(json.dumps({"crv_events_done": last, "events_total": len(timing.keys)}), flush=True)
    return join(result), {"seconds": time.perf_counter() - start, "stages": metrics}


def write_output(path, keys, timing, crv, tracker=None, extra=None, compression=None):
    """CRV and tracker tables (and `extra` {tree: columns}) to one ROOT file; compression None keeps uproot's default
    (ZLIB 1), otherwise an uproot compression object (decoded arrays are the same, file bytes differ)."""
    D, P, C = crv
    def keyed(rows):
        ev = np.asarray(rows["ev"], np.int64)
        result = {name: keys[ev, i] for i, name in enumerate(("run", "subrun", "event"))}
        result.update({k: v for k, v in rows.items() if k != "ev"})
        return result
    with uproot.recreate(path, **({} if compression is None else {"compression": compression})) as root:
        events = {name: keys[:, i] for i, name in enumerate(("run", "subrun", "event"))}
        events.update(mc_pbtime=timing.mc_pbtime, reco_pbtime=timing.reco_pbtime, event_length=timing.event_length)
        for name, rows in (("digis", D), ("pulses", P), ("clusters", C)):
            events["n" + name] = np.bincount(np.asarray(rows["ev"], np.int64), minlength=len(keys))
            root.mktree("crvfast/" + name, keyed(rows))
        root.mktree("crvfast/events", events)
        if tracker is not None:
            root.mktree("trkdigi/events", tracker)
        for name, columns in (extra or {}).items():
            root.mktree(name, columns)


def save_sampling(out, provenance):
    """Persist exact NumPy selections separately from reduced detector tables."""
    arrays = {}
    def split(value, prefix):
        if isinstance(value, np.ndarray):
            arrays[prefix] = value
            return {"npz_key": prefix, "shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, dict):
            return {k: split(v, prefix + "/" + k) for k, v in value.items()}
        return value
    metadata = split(provenance, "sampling")
    np.savez(out / "sampling.npz", **arrays)
    (out / "sampling.json").write_text(json.dumps(json_values(metadata), indent=2, allow_nan=False) + "\n")


def prepare_frames(manifest, max_events, repeats, reference):
    """Measure independent assemblies with optional content-verified source reuse."""
    from .direct_frames import load_manifest
    from .direct_regression import compare_sampling, input_digests
    from .frame_cache import FrameSourceCache
    cache = FrameSourceCache() if repeats > 1 else None
    runs, previous, check_s = [], None, 0.0
    for index in range(repeats):
        start = time.perf_counter()
        kwargs = {"source_cache": cache} if cache is not None else {}
        result = load_manifest(manifest, max_events=max_events, **kwargs)
        elapsed = time.perf_counter() - start
        mixed, tracker, crv, pdg, provenance = result
        runs.append({"pass": index, "seconds": elapsed, "stages": provenance["timing_s"]})
        start = time.perf_counter()
        if reference is not None:
            compare_sampling(provenance, reference)
        if repeats > 1:
            current = input_digests(mixed, tracker, crv, pdg)
            if previous is not None and current != previous:
                raise ValueError("same-seed assembly changed full flat truth or detector inputs")
            previous = current
        check_s += time.perf_counter() - start
        print(json.dumps({"preparation": runs[-1]}), flush=True)
        if index < repeats - 1:
            del result, mixed, tracker, crv, pdg, provenance
    return result, {"passes": runs, "exact_check_s": check_s,
                    "repeats_identical": repeats > 1,
                    "sampling_reference_identical": reference is not None,
                    "cache_hits": cache.hits if cache is not None else 0,
                    "cache_misses": cache.misses if cache is not None else 0}


def main(argv=None):
    from crvfastsim import onspill, runconfig
    from crvfastsim.response import photons_jax, sipm
    from trkfastsim.digi import digicond, fastdigi
    from . import crv_inputs
    import jax
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="saved native mixed-step cache")
    source.add_argument("--frames-manifest", type=Path, help="direct joint-frame sampling manifest")
    parser.add_argument("--expanded", type=Path, help="explicit expanded response FHiCL for direct frames")
    for name in ("truth", "stock", "photon-csv", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--tracker-conditions", type=Path)
    parser.add_argument("--tracker-reference", type=Path)
    parser.add_argument("--tracker-seed", type=int, default=20261105)
    parser.add_argument("--preparation-repeats", type=int, default=1, help="direct-only in-process pool reuse measurement")
    parser.add_argument("--sampling-reference", type=Path, help="direct-only saved sampling receipt, checked before response")
    parser.add_argument("--skip-comparisons", action="store_true", help="single-pass direct throughput; compare saved outputs externally")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.preparation_repeats < 1 or args.chunk < 1 or (args.max_events is not None and args.max_events < 2):
        parser.error("positive chunk/repeats and at least two events required")
    if args.frames_manifest:
        if args.expanded is None or args.tracker_reference is not None:
            parser.error("direct frames require --expanded and cannot claim saved tracker identity")
    elif bool(args.tracker_conditions) != bool(args.tracker_reference):
        parser.error("native cache tracker-conditions and tracker-reference must be supplied together")
    if not args.frames_manifest and (args.sampling_reference is not None or args.preparation_repeats != 1):
        parser.error("sampling-reference/preparation-repeats require direct frames")
    if args.skip_comparisons and (not args.frames_manifest or args.repeats != 1 or args.preparation_repeats != 1
                                  or args.sampling_reference is not None):
        parser.error("skip-comparisons requires direct frames, one preparation, one response, and no sampling-reference")
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("this development benchmark requires a GPU")
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    provenance, tracker_steps, sampling_write_s = None, None, 0.0
    if args.frames_manifest:
        import hashlib
        prepared, preparation = prepare_frames(args.frames_manifest, args.max_events, args.preparation_repeats, args.sampling_reference)
        mixed, tracker_steps, steps, pdg, provenance = prepared
        del prepared
        explicit = {"manifest": args.frames_manifest, "expanded": args.expanded, "photon_csv": args.photon_csv}
        if args.tracker_conditions:
            explicit["tracker_conditions"] = args.tracker_conditions
        provenance["configuration"] = {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                       for name, path in explicit.items()}
        provenance["timing_sources"] = {"mc": str(args.truth.resolve()), "reco": str(args.stock.resolve())}
        keys = mixed.event_ids
        begin = time.perf_counter()
        save_sampling(args.out, provenance)
        sampling_write_s = time.perf_counter() - begin
        mc, reco, length = crv_inputs.read_timing(keys, args.truth, args.stock)
        del mixed
        if args.tracker_conditions is None:
            tracker_steps = None
    else:
        keys, steps, pdg, mc, reco, length = crv_inputs.read_inputs(args.input, args.truth, args.stock)
        n = min(args.max_events or len(keys), len(keys))
        keep = np.isin(event_keys(event_ids(steps)), event_keys(keys[:n]))
        keys, steps, pdg = keys[:n], steps.loc[keep].reset_index(drop=True), pdg[keep]
        mc, reco, length = mc[:n], reco[:n], length[:n]
    input_s, n = time.perf_counter() - start - sampling_write_s, len(keys)
    timing = onspill.EventTiming(keys, mc, reco, length)
    rc = runconfig.get("run1a_v01")
    geom, counters = rc.geometry(), rc.counters_cfg()
    cond = crv_inputs.load_conditions(args.truth, args.photon_csv, geom, counters)
    expanded_path = args.expanded or args.input.parent / "expanded.fcl"
    expanded = digicond.parse_fhicl_dump(expanded_path.read_text())
    cfg = crv_inputs.build_config(expanded, jax=True)
    tabs, pmap = photons_jax.load_device_tables(cfg.response), sipm.photon_map()
    tracker_cond = None
    if args.tracker_conditions:
        if tracker_steps is None:
            with uproot.open(args.input) as root:
                tracker_steps = pd.concat([pd.DataFrame(root[f"{d}/sgsteps"].arrays(fastdigi.STEP_COLS, library="np"))
                                           for d in STEP_DIRS], ignore_index=True)
            tracker_steps = tracker_steps[np.isin(event_keys(event_ids(tracker_steps)), event_keys(keys))]
        tracker_cond = digicond.Conditions.load(args.tracker_conditions)
    setup = time.perf_counter() - start - sampling_write_s
    passes, reference_crv, reference_tracker = [], None, None
    for index in range(args.repeats):
        begin = time.perf_counter()
        crv, metrics = crv_pass(steps, pdg, timing, geom, counters, cond, cfg, tabs, pmap, args.seed, args.chunk)
        if reference_crv is None:
            reference_crv = crv
        elif any(not np.array_equal(a[k], b[k], equal_nan=True) for a, b in zip(reference_crv, crv) for k in a):
            raise ValueError("same-seed CRV repeat changed output arrays")
        row = {"pass": index, "crv": metrics}
        if tracker_cond is not None:
            t0 = time.perf_counter()
            stats = {}
            out_keys, digi, nsteps = fastdigi.digitize_steps(tracker_steps, tracker_cond, jax.random.key(args.tracker_seed),
                keys=keys, chunk=args.chunk, stats=stats, backend="jax",
                event_timing={"pbtime": timing.mc_pbtime, "event_length": timing.event_length})
            if not np.array_equal(out_keys, keys):
                raise ValueError("tracker changed event identities")
            row["tracker"] = {"seconds": time.perf_counter() - t0, "steps": nsteps, **stats}
            if reference_tracker is None:
                reference_tracker = digi
            elif any(not np.array_equal(reference_tracker[k], digi[k], equal_nan=True) for k in digi):
                raise ValueError("same-seed tracker repeat changed output arrays")
        row["combined_response_s"] = time.perf_counter() - begin
        passes.append(row)
        print(json.dumps(row), flush=True)
    columns = None
    if reference_tracker is not None:
        columns = digi_columns(keys, reference_tracker, tracker_cond.nADCSamples)
        if provenance is not None:
            columns.update({name: keys[:, i] for i, name in enumerate(("run", "subrun", "event"))})
        if args.tracker_reference:
            compare_reference(keys, ak.Array(columns), args.tracker_reference)
    from mu2efastsim import precision
    report = {"events": n, "crv_steps": len(steps), "seed": args.seed, "chunk": args.chunk,
              "crv_device_precision": precision.NAME, "tracker_device_precision": "float64",
              "devices": [str(d) for d in jax.devices()], "setup_s": setup, "input_s": input_s,
              "input_mode": "direct_frames" if provenance is not None else "native_cache", "passes": passes,
              "paired_comparisons_skipped": args.skip_comparisons,
              "crv_paired": None if args.skip_comparisons else compare_crv(args.stock, keys, *reference_crv, geom),
              "boundary": "cached mixed steps and explicit timing to CRV response/reco and optional tracker digis; excludes native mixing, setup, and file writes",
              "scope": "development only; reduced CRV diagnostic tables, not a complete art-product bridge",
              "same_seed_repeats_identical": args.repeats > 1,
              "tracker_saved_identical": True if args.tracker_reference else None}
    if provenance is not None:
        report["boundary"] = "direct sampled joint frames to detector response; response timers exclude frame read/sampling/assembly, assets and writing, all reported separately"
        report["comparison_scope"] = "same primary IDs with independently sampled pileup and response; not exact stock donor replay; finite-pool uncertainty excluded"
        report["sampling_write_s"] = sampling_write_s
        report["frame_preparation_s"] = provenance["timing_s"]
        report["preparation"] = preparation
    if columns is not None and not args.skip_comparisons:
        stock = compare_reference(keys, None, args.stock)
        report["tracker_paired"] = comparison(stock, columns)
    begin = time.perf_counter()
    write_output(args.out / "fast.root", keys, timing, reference_crv, columns)
    report["write_s"] = time.perf_counter() - begin
    (args.out / "report.json").write_text(json.dumps(json_values(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"events": n, "output": str(args.out), "passes": passes}), flush=True)


if __name__ == "__main__":
    main()
