"""Development comparison of fast tracker digitisation on native four-stream steps."""
import argparse
import cProfile
import json
import pstats
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from .compare import occupancy, paired_counts
from .counts import event_ids, event_keys, json_values

STEP_DIRS = ("primarySteps", "stopSteps", "eleSteps", "muSteps", "neutralSteps")


def read_input(path):
    """Require a complete event census and all five step collections, including empty ones."""
    from trkfastsim.digi.fastdigi import STEP_COLS
    with uproot.open(path) as root:
        events = root["crvdump/events"].arrays(library="ak")
        stock = root["trkdigi/events"].arrays(library="ak") if "trkdigi/events" in root else None
        keys = event_ids(events)
        if ((stock is not None and not np.array_equal(keys, event_ids(stock)))
                or np.unique(event_keys(keys)).size != len(keys)):
            raise ValueError("tracker/CRV census mismatch or duplicate identities")
        steps = pd.concat([pd.DataFrame(root[f"{name}/sgsteps"].arrays(STEP_COLS, library="np"))
                           for name in STEP_DIRS], ignore_index=True)
    if not np.all(ak.to_numpy(events.spill) == 1):
        raise ValueError("this development driver requires on-spill events")
    timing = {"pbtime": ak.to_numpy(events.pbtime), "event_length": ak.to_numpy(events.evlen)}
    return keys, steps, timing, stock


def field_at_origin(path, origin):
    points = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    at = np.all(np.isclose(points[:, :3], origin, rtol=0, atol=1e-6), axis=1)
    if np.count_nonzero(at) != 1 or not np.isfinite(points[at]).all():
        raise ValueError("field CSV must contain exactly one finite sample at detector origin")
    return float(points[at, 5][0])


def compare_reference(keys, stock, saved):
    with uproot.open(saved) as root:
        ref = root["trkdigi/events"].arrays(library="ak")
    rkeys = event_keys(event_ids(ref))
    order = np.argsort(rkeys)
    where = np.searchsorted(rkeys[order], event_keys(keys))
    if np.any(where >= len(order)) or not np.array_equal(rkeys[order[where]], event_keys(keys)):
        raise ValueError("extraction event is absent from saved development baseline")
    aligned = ref[order[where]]
    if stock is not None:
        for name in stock.fields:
            if name.startswith("d_") and not ak.almost_equal(stock[name], aligned[name], rtol=0, atol=0):
                raise ValueError(f"extraction changed saved tracker output: {name}")
    return aligned


def compare_step_prefix(path, reference):
    """Check an extracted prefix before comparing an all-event fast-input run."""
    with uproot.open(path) as root, uproot.open(reference) as ref:
        events = ref["crvdump/events"].num_entries
        for directory in STEP_DIRS:
            tree = ref[f"{directory}/sgsteps"]
            fields = tree.keys()
            left = root[f"{directory}/sgsteps"].arrays(fields, entry_stop=tree.num_entries, library="ak")
            right = tree.arrays(fields, library="ak")
            if not ak.almost_equal(left, right, rtol=0, atol=0):
                raise ValueError(f"extraction changed step prefix: {directory}")
        for field in ("run", "subrun", "event", "pbtime", "evlen"):
            left = root["crvdump/events"][field].array(entry_stop=events, library="np")
            right = ref["crvdump/events"][field].array(library="np")
            if not np.array_equal(left, right):
                raise ValueError(f"extraction changed timing prefix: {field}")
    return events


def compare_crv(path, reference):
    with uproot.open(path) as root, uproot.open(reference) as ref:
        for tree in ("events", "digis", "pulses", "clusters"):
            fields = [name for name in ref[f"crvdump/{tree}"].keys() if name != "pbtime"]
            left = root[f"crvdump/{tree}"].arrays(fields, library="ak")
            right = ref[f"crvdump/{tree}"].arrays(fields, library="ak")
            if not ak.almost_equal(left, right, rtol=0, atol=0):
                raise ValueError(f"fast-input extraction changed saved CRV {tree}")
    return True


def digi_columns(keys, digi, nsamp):
    counts = np.bincount(digi["ev"], minlength=len(keys))
    def jag(values):
        return ak.unflatten(np.asarray(values, np.uint32), counts)
    return {"run": keys[:, 0].astype(np.uint32), "subrun": keys[:, 1].astype(np.uint32),
            "event": keys[:, 2].astype(np.uint32), "d_sid": jag(digi["sid"]),
            "d_tdc0": jag(digi["tdc"][:, 0]), "d_tdc1": jag(digi["tdc"][:, 1]),
            "d_tot0": jag(digi["tot"][:, 0]), "d_tot1": jag(digi["tot"][:, 1]),
            "d_pmp": jag(digi["pmp"]), "d_flags": jag(np.zeros(len(digi["sid"]), np.uint32)),
            "d_nadc": jag(np.full(len(digi["sid"]), nsamp)),
            "d_adc": ak.unflatten(np.asarray(digi["adc"], np.uint32).ravel(), counts * nsamp)}


def histograms(rows, name, edges):
    values = ak.to_numpy(ak.flatten(rows[name]))
    event = np.repeat(np.arange(len(rows[name])), ak.to_numpy(ak.num(rows[name])))
    bins = np.searchsorted(edges, values, side="right")
    return np.bincount(event * (len(edges) + 1) + bins,
                       minlength=len(rows[name]) * (len(edges) + 1)).reshape(len(rows[name]), -1)


def comparison(stock, fast):
    mapping = (np.arange(65536) & 0xFC00) >> 10
    s = occupancy(stock["d_sid"], mapping, 36)
    f = occupancy(fast["d_sid"], mapping, 36)
    result = {"planes": paired_counts(s, f),
              "total": paired_counts(s.sum(axis=1, keepdims=True), f.sum(axis=1, keepdims=True))}
    for name, edges in (("d_tdc0", np.arange(0, 100001, 2000)), ("d_tdc1", np.arange(0, 100001, 2000)),
                        ("d_tot0", np.arange(0, 17)), ("d_tot1", np.arange(0, 17)),
                        ("d_pmp", np.linspace(0, 1500, 31)), ("d_adc", np.linspace(0, 4096, 33))):
        result[name] = {"edges": edges, "includes_flow_bins": True,
                        **paired_counts(histograms(stock, name, edges), histograms(fast, name, edges))}
    return result


@contextmanager
def diagnostic_profile(out, *, host=False, stages=False):
    """Reuse the tracker stage profiler; diagnostics are not throughput measurements."""
    if stages:
        from trkfastsim.validation.digi_stage_profile import stage_profile
        context = stage_profile(True)
    else:
        context = nullcontext({})
    profiler = cProfile.Profile() if host else None
    if profiler is not None:
        profiler.enable()
    try:
        with context as measurements:
            yield measurements
    finally:
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(str(out / "host_warm.pstats"))
            for sort in ("cumulative", "tottime"):
                with (out / f"host_warm_{sort}.txt").open("w") as stream:
                    pstats.Stats(profiler, stream=stream).sort_stats(sort).print_stats(100)
        if stages:
            (out / "stages_warm.json").write_text(json.dumps(measurements, indent=2) + "\n")


def profile_kind(index, enabled):
    return {1: "host_profile", 2: "synchronized_stages"}.get(index, "uninstrumented") if enabled else "uninstrumented"


def main(argv=None):
    from trkfastsim.digi import digicond, fastdigi
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--gdml", type=Path, required=True)
    parser.add_argument("--detector-origin", type=float, nargs=3, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--stock", type=Path, required=True, help="saved development response, never held-out")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--steps-reference", type=Path, help="saved native extraction prefix")
    parser.add_argument("--profile", action="store_true", help="separate warm host/stage profiles; requires JAX and at least five passes")
    parser.add_argument("--fast-reference", type=Path, help="require exact detector columns against a saved fast development run")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.chunk < 1:
        parser.error("chunk and repeats must be positive")
    if args.profile and (args.backend != "jax" or args.repeats < 5):
        parser.error("profiling requires JAX and at least five passes (warmup, host, stages, two uninstrumented)")
    t0 = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    keys, steps, timing, stock = read_input(args.input)
    extraction_has_digis = stock is not None
    stock = compare_reference(keys, stock, args.stock)
    prefix_events = compare_step_prefix(args.input, args.steps_reference) if args.steps_reference else None
    crv_identical = compare_crv(args.input, args.stock) if not extraction_has_digis else None
    base = args.input.parent
    cfg = digicond.parse_fhicl_dump((base / "expanded.fcl").read_text())
    timing_tag = cfg["physics"]["analyzers"]["crvdump"].get("protonBunchTime", "EWMProducer")
    ewm = cfg["physics"]["producers"][timing_tag]
    if not ewm.get("RecoFromMCTruth") or ewm.get("RecoFromMCTruthErr", 0) != 0:
        raise ValueError("event table must expose unsmeared MC bunch time, not reconstructed PBT")
    bz = field_at_origin(base / "defaultDetectorOrigin.csv", args.detector_origin)
    cond = digicond.build(base / "expanded.fcl", base / "db_TrkPreampStraw.csv", args.gdml,
                         base / "db_TrkAlignStrawSim.csv", global_constants=base / "globalConstants.txt",
                         field_bz_T=bz, detector_origin=args.detector_origin)
    cond.save(args.out / "conditions.npz")
    setup_s = time.perf_counter() - t0
    if args.backend == "jax":
        import jax
        if args.require_gpu and not any(device.platform == "gpu" for device in jax.devices()):
            raise RuntimeError("GPU requested but JAX has no GPU")
        devices = [str(device) for device in jax.devices()]
    else:
        if args.require_gpu:
            parser.error("--require-gpu requires --backend jax")
        devices = ["NumPy host"]
    passes, first = [], None
    for index in range(args.repeats):
        rng = np.random.default_rng(args.seed) if args.backend == "numpy" else jax.random.key(args.seed)
        stats = {}
        kind = profile_kind(index, args.profile)
        with diagnostic_profile(args.out, host=kind == "host_profile", stages=kind == "synchronized_stages"):
            begin = time.perf_counter()
            out_keys, digi, nsteps = fastdigi.digitize_steps(steps, cond, rng, keys=keys, chunk=args.chunk,
                                                           stats=stats, backend=args.backend, event_timing=timing)
            elapsed = time.perf_counter() - begin
        if not np.array_equal(keys, out_keys):
            raise ValueError("digitiser changed event population")
        if first is None:
            first = digi
        elif any(not np.array_equal(first[key], digi[key], equal_nan=True) for key in first):
            raise ValueError("same-seed repeat changed output arrays")
        row = {"pass": index, "seconds": elapsed, "events_per_s": len(keys) / elapsed, "instrumentation": kind, **stats}
        if args.backend == "jax":
            row["device_memory"] = jax.devices()[0].memory_stats()
        passes.append(row)
        print(json.dumps(passes[-1]), flush=True)
    columns = digi_columns(keys, first, cond.nADCSamples)
    if args.fast_reference:
        compare_reference(keys, ak.Array(columns), args.fast_reference)
    begin = time.perf_counter()
    with uproot.recreate(args.out / "fast.root") as root:
        root.mktree("trkdigi/events", columns)
    report = {"events": len(keys), "steps": nsteps, "digis": len(first["sid"]), "backend": args.backend,
              "devices": devices, "seed": args.seed, "chunk": args.chunk, "field_bz_T": bz,
              "setup_s": setup_s, "root_write_s": time.perf_counter() - begin, "passes": passes,
              "matched_stock_events": len(keys), "extraction_has_digis": extraction_has_digis,
              "paired": comparison(stock, columns), "identical_step_prefix_events": prefix_events,
              "crv_identical_to_saved": crv_identical,
              "fast_reference_identical": True if args.fast_reference else None,
              "profiling": args.profile,
              "boundary": "merged steps plus shared timing to detector digis; excludes native mixing/CRV and ROOT serialization",
              "status": "development only; on-spill port not promoted; no full StrawDigiMC"}
    (args.out / "report.json").write_text(json.dumps(json_values(report), indent=2, allow_nan=False) + "\n")
    headline = {key: report[key] for key in ("events", "steps", "digis")}
    headline["total"] = report["paired"]["total"]
    headline["shape_chi2_ndf"] = {key: [value["chi2_diagonal"], value["ndf_diagonal"]]
                                   for key, value in report["paired"].items()}
    print(json.dumps(json_values(headline), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
