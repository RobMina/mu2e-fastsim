"""One predeclared radius-locality development candidate; reuse every saved baseline arm."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..artsteps import onspill_timing
from . import placement_diagnostic as pd
from . import production_compare as pc
from .compare import paired_mean
from .crystal_placement import place_crystals
from .library import load, sample
from .response import place

NEIGHBORS = 8
SEED = 20261105
RECO_SEED = 20261108
FIELDS = ("energy", "time", "x", "y", "z")
ARMS = ("geometric", "crystal")
CONDITIONS = ("Sim_crystal.txt", "CsI-waveform-2020-02-12.root", "ReadoutNoise.root", "CsI-propag-2020-07-29.root")


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def fingerprint(arrays):
    """Content digest includes sorted names, dtype, shape and complete array payloads."""
    result = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        a = np.ascontiguousarray(value)
        result.update(json.dumps([name, a.dtype.descr, a.shape], sort_keys=True).encode())
        result.update(a.tobytes())
    return result.hexdigest()


def verify_receipt(paths, receipt, previous, reco_inputs):
    """Pin current data to the completed diagnostic, without asserting historical code hashes."""
    for name, path in paths.items():
        if digest(path) != receipt["inputs"][name]["sha256"]:
            raise ValueError("saved diagnostic input hash mismatch: " + name)
    provenance = previous["provenance"]
    expected = dict(seed=str(SEED), reco_seed=str(RECO_SEED), stock_matched_phase0="True")
    if any(provenance[k] != v for k, v in expected.items()):
        raise ValueError("baseline seed or timing provenance differs")
    if Path(provenance["reco_inputs"]).resolve() != Path(reco_inputs).resolve():
        raise ValueError("baseline calibration input directory differs")
    if previous["audit"]["timing"] != pc.TIMING:
        raise ValueError("baseline timing settings differ")


def prepare(tables, library, metadata, saved):
    """Validate saved default draw first, then construct exactly one shared K=8 draw."""
    entries, original, baseline, baseline_placed, audit = pd.reproduce(tables, library, metadata, saved, SEED)
    timing = onspill_timing(saved["full_ids"]["event"], **pc.TIMING)
    for name, value in timing.items():
        pd.exact("timing_" + name, value, saved["timing_" + name])
    for arm in ("stockref", "control", *ARMS):
        positive = saved[arm + "_energy"] > 0
        for field in FIELDS:
            values = saved[arm + "_" + field]
            if values.shape != saved["event_row"].shape or not np.all(np.isfinite(values[positive])):
                raise ValueError("invalid saved cluster field: " + arm + "_" + field)
    draw = sample(library, entries, np.random.default_rng(SEED), orientation="disk", radius_neighbors=NEIGHBORS)
    for name in ("supported", "fallback"):
        pd.exact("candidate_" + name, baseline[name], draw[name])
    unchanged_pool = draw["supported"] & (draw["bin_pool_size"] <= NEIGHBORS)
    pd.exact("small_pool_donor", baseline["donor"][unchanged_pool], draw["donor"][unchanged_pool])
    placed = dict(geometric=place(library, draw), crystal=place_crystals(library, entries, draw))
    geometry = {k.removeprefix("geometry_"): v for k, v in saved.items() if k.startswith("geometry_")}
    audit.update(sampling=sampling_summary(library, entries, baseline, baseline_placed, draw, placed),
        fingerprints=dict(baseline_draw=fingerprint(baseline), candidate_draw=fingerprint(draw),
                          candidate_placements={k: fingerprint(v) for k, v in placed.items()}))
    return dict(entries=entries, original=original, baseline_draw=baseline, draw=draw, placed=placed,
                geometry=geometry, timing=timing, full_ids=saved["full_ids"], audit=audit)


def sampling_summary(library, entries, baseline, baseline_placed, draw, placed):
    """Report occupancy and donor reuse rather than pretending repeated donors are independent."""
    result = {}
    for name, candidate, mapping in (("baseline", baseline, baseline_placed), ("candidate", draw, placed)):
        features = pd.donor_features(library, entries, candidate)
        strata = {}
        for label, mask in pd.strata(entries).items():
            supported = mask & candidate["supported"]
            counts = np.bincount(candidate["donor"][supported], minlength=len(library["entry_energy"]))
            steps = mask[candidate["entry"]]
            total = float(candidate["energy_birks"][steps].sum())
            strata[label] = dict(entries=int(mask.sum()), supported=int(supported.sum()),
                fallback=int((mask & candidate["fallback"]).sum()),
                empty=int((supported & (np.diff(candidate["offsets"]) == 0)).sum()),
                unique_donors=int((counts > 0).sum()), max_donor_reuse=int(counts.max(initial=0)),
                donor_reuse_concentration=float(np.sum(counts ** 2) / counts.sum() ** 2) if counts.sum() else None,
                absolute_radius_mismatch_mm=pd.summarize(np.abs(features["radius_delta"][mask])),
                rotated_direction_mismatch_deg=pd.summarize(features["angle_deg"][mask]),
                lost_energy_percent={arm: 100 * float(v["energy_birks"][steps & ~v["mapped"]].sum()) / total
                                     if total else None for arm, v in mapping.items()})
        result[name] = strata
    result["changed_donors"] = int((draw["donor"] != baseline["donor"]).sum())
    result["radius_pool_size"] = pd.summarize(draw["radius_pool_size"])
    result["bin_pool_size"] = pd.summarize(draw["bin_pool_size"])
    return result


def cluster_mapping(saved):
    return {arm: {field: saved[arm + "_" + field] for field in FIELDS} for arm in ("stockref", "control", *ARMS)}


def comparison(prepared, baseline, candidate):
    """Reuse paired reports; add the same six-arm selected subset for spatial comparisons."""
    from .reconstruct import report

    rows = prepared["entries"]["event_row"]
    result = dict(candidate=pc.diagnostics(prepared, candidate), baseline=pc.diagnostics(prepared, baseline), change={})
    common = np.logical_and.reduce([v["energy"][rows] > 10 for v in (
        baseline["stockref"], baseline["control"], baseline["geometric"], baseline["crystal"],
        candidate["geometric"], candidate["crystal"])])
    result["six_arm_common_selected"] = int(common.sum())
    result["six_arm_distance_mm"] = {}
    stock = baseline["stockref"]
    for label, clusters in (("baseline", baseline), ("candidate", candidate)):
        for arm in ARMS:
            distance = np.hypot(clusters[arm]["x"][rows[common]] - stock["x"][rows[common]],
                                clusters[arm]["y"][rows[common]] - stock["y"][rows[common]])
            result["six_arm_distance_mm"][label + "_" + arm] = paired_mean(distance, np.zeros(len(distance)), rows[common] // 100)
    for arm in ARMS:
        result["change"][arm] = {}
        for name, mask in pd.strata(prepared["entries"]).items():
            f = {k: v[rows[mask]] for k, v in candidate[arm].items()}
            b = {k: v[rows[mask]] for k, v in baseline[arm].items()}
            result["change"][arm][name] = report(f, b, rows[mask])
    return result


def compact(result):
    summaries = {}
    for arm in ARMS:
        value = result["candidate"][arm]
        summaries[arm] = dict(selected=value["paired"]["fast_selected"], stock=value["paired"]["stock_selected"],
            ratio=value["paired"]["acceptance_ratio"], energy=value["paired"]["energy_mean"],
            time=value["paired"].get("paired_time_ns"),
            shapes={k: {f: v[f] for f in ("chi2", "ndf")} for k, v in value["observables"].items()},
            strata={k: dict(entries=v["events"], ratio=v["acceptance_ratio"], energy=v["energy_mean"])
                    for k, v in value["strata"].items()},
            change={k: {f: v[f] for f in ("events", "fast_selected", "stock_selected", "fast_only", "stock_only",
                                         "acceptance_difference", "energy_mean")} for k, v in result["change"][arm].items()})
    return dict(arms=summaries, six_arm_common_selected=result["six_arm_common_selected"],
                six_arm_distance_mm=result["six_arm_distance_mm"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("tables", "library", "baseline", "prior-diagnostic", "reco-inputs", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    if any((args.out / name).exists() for name in ("clusters.npz", "numbers.json", "timing.json")):
        raise ValueError("output already contains a run; reuse saved outputs, not overwrite")
    paths = dict(tables=args.tables, library=args.library, clusters=args.baseline / "clusters.npz")
    receipt = json.loads(args.prior_diagnostic.read_text())
    previous = json.loads((args.baseline / "numbers.json").read_text())
    verify_receipt(paths, receipt, previous, args.reco_inputs)
    with np.load(paths["clusters"], allow_pickle=False) as source:
        saved = {k: source[k] for k in source.files}
    library, metadata = load(args.library)
    prepared = prepare(args.tables, library, metadata, saved)
    source_root = Path(__file__).resolve().parents[1]
    inputs = {**paths, **{name: args.reco_inputs / name for name in CONDITIONS},
              "baseline_report": args.baseline / "numbers.json", "prior_diagnostic": args.prior_diagnostic}
    manifest = dict(neighbors=NEIGHBORS, seed=SEED, reco_seed=RECO_SEED,
        policy="Uniform among up to 8 nearest radii in the existing chosen bin/disk; stable ties by library row. "
               "Same occupied-cell fallback; unsupported distinct from empty; empty donors eligible; no rescale. "
               "Eligible rows sorted in original row order; exactly legacy RNG schedule, but changed donors alter reco RNG history.",
        audit=prepared["audit"],
        inputs={k: dict(path=str(v), sha256=digest(v)) for k, v in inputs.items()},
        calo_source_sha256={str(v.relative_to(source_root)): digest(v) for v in sorted(source_root.rglob("*.py"))},
        scope="One predeclared development candidate, no scan, new stock, transport or confirmation. "
              "All stored rows retained; sampled comparisons only direct-front queries; not inclusive acceptance. "
              "Same native 15-iteration backend; one CPU call per candidate arm, no throughput claim. "
              "Existing baseline runtime lacks code/condition hashes; current fingerprints are not retrospective proof. "
              "Paired 100-source-row errors omit library/draw uncertainty; legacy parent independence not certified.")
    args.out.mkdir(parents=True, exist_ok=True)
    pc._json(args.out / "preparation.json", manifest)
    np.savez_compressed(args.out / "draw.npz", **prepared["draw"])
    if args.prepare_only:
        print(json.dumps(prepared["audit"], indent=2, allow_nan=False))
        return
    from .. import chain, stockio
    from ..cluster import Geometry
    from ..templatefit import FitConfig
    from dataclasses import asdict
    import importlib.metadata
    from .reconstruct import leading, steps_for_reco
    from .compare import plot
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    if jax.default_backend() != "cpu":
        raise RuntimeError("CPU reconstruction required")
    if FitConfig().iterations != 15:
        raise ValueError("native fit iteration default differs from baseline")
    manifest["runtime"] = dict(backend=jax.default_backend(), x64=bool(jax.config.jax_enable_x64),
        versions={k: importlib.metadata.version(k) for k in ("numpy", "jax", "jaxlib", "scipy", "uproot")},
        fit_config=asdict(FitConfig()))
    pc._json(args.out / "preparation.json", manifest)
    sim = stockio.load_sim_crystal(args.reco_inputs / CONDITIONS[0])
    pulse = stockio.load_pulse(args.reco_inputs / CONDITIONS[1])
    noise = stockio.load_noise(args.reco_inputs / CONDITIONS[2])
    propagation = stockio.load_propagation(args.reco_inputs / CONDITIONS[3])
    geometry = Geometry.load(args.tables)
    baseline = cluster_mapping(saved)
    candidate = {k: v for k, v in baseline.items() if k in ("stockref", "control")}
    output = {k: v for k, v in saved.items() if not k.startswith(("geometric_", "crystal_", "sampling_"))}
    output.update({"baseline_" + k: v for k, v in saved.items() if k.startswith(("geometric_", "crystal_", "sampling_"))})
    output.update({"sampling_" + k: prepared["draw"][k] for k in
                   ("donor", "supported", "fallback", "offsets", "radius_pool_size", "bin_pool_size")})
    runs = []
    for arm in ARMS:
        steps = steps_for_reco(dict(entries=prepared["entries"], steps=prepared["placed"][arm]), sim)
        jax.effects_barrier()
        start = time.perf_counter()
        reco = chain.run_chain(steps, prepared["timing"], lambda sipm: stockio.adc_per_pe(sim, sipm),
            pulse, noise, geometry, np.random.default_rng(RECO_SEED), table=propagation, adc2mev=stockio.adc_to_mev)
        jax.effects_barrier()
        candidate[arm] = leading(reco["clusters"], len(saved["event_row"]))
        runs.append(dict(arm=arm, seconds=time.perf_counter() - start, stages=reco["seconds"],
            n_refit_skipped=reco["n_refit_skipped"], n_nonfinite_reco=reco["n_nonfinite_reco"],
            nonfinite_examples=reco.get("nonfinite_examples"),
            leading_nonfinite={k: int((~np.isfinite(v)).sum()) for k, v in candidate[arm].items()}))
        output.update({arm + "_" + k: v for k, v in candidate[arm].items()})
        np.savez_compressed(args.out / "clusters.npz", **output)
        pc._json(args.out / "timing.json", runs)
        print(json.dumps({k: runs[-1][k] for k in ("arm", "seconds", "n_refit_skipped", "n_nonfinite_reco")},
                         allow_nan=False), flush=True)
    result = comparison(prepared, baseline, candidate)
    css, arrays = pd.diagnose(library, prepared["entries"], prepared["original"], prepared["draw"], prepared["placed"], output)
    result.update(manifest=manifest, runs=runs)
    pc._json(args.out / "numbers.json", result)
    pc._json(args.out / "css_accounting.json", css)
    np.savez_compressed(args.out / "css_arrays.npz", **arrays)
    summary = compact(result)
    pc._json(args.out / "summary.json", summary)
    for arm in ARMS:
        plot(result["candidate"][arm]["observables"], args.out / (arm + ".png"),
             "Radius K=8 " + arm + ": development only", labels=("Saved stock reconstruction", "Radius-local candidate"))
    pd.plot_report(css, args.out / "css_energy_balance.png")
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
