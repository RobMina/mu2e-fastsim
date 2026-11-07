"""One authenticated production-candidate draw and two CPU calls, with no reference replay."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import uproot

from ..artsteps import onspill_timing
from . import candidate_support as auth, export, library, placement_diagnostic as pd
from . import production_compare as pc, source_overlap_report
from .crystal_placement import place_crystals
from .radius_trial import ARMS, CONDITIONS, FIELDS, NEIGHBORS, RECO_SEED, SEED, digest, fingerprint
from .response import place

PINS = {
    "candidate": "ed177ed1018e3bf515160c965d435b2bd4bd22b35b8d7bf6ed0db02c39a5d1f7",
    "support": "e9e9206104804c46f211e12093001756c359a01438ccdbc27fab872645a4936e",
    "k8_manifest": "d3fa5ee86a39f498ab237c5350ae65323806e8ec65e1b82e60fba5a1e86be97f",
    "k8_clusters": "6e2f6a4645887dfd2d4c10ffe9939e29c2905972a91300d28edade12456030cd",
    "k8_numbers": "4dc7c22eb0b57634f0071b74fa79ff9ee1d97eeb3dd3cee38f8f784b90f16d3a",
}
SCOPE = ("One fixed K8 draw, two existing placements, exactly two CPU calls; saved references only. "
         "Selected-production development, not inclusive acceptance, confirmation, current-head closure or throughput. "
         "Paired stored-row blocks omit library/model uncertainty; selected-production versus legacy is confounded. "
         "No original-proton independence; development/pilot 165 shared parents retained, file0/pilot zero.")


def checked_file(inputs, path, sha):
    path = Path(path).absolute()
    if digest(path) != sha:
        raise ValueError("input hash mismatch: " + path.name)
    if str(path) in inputs.sha256 and inputs.sha256[str(path)] != sha:
        raise ValueError("conflicting input anchors")
    inputs.sha256[str(path)] = sha


def validate_clusters(saved, total, arms=("stockref", "control", *ARMS)):
    pd.exact("stored event rows", saved["event_row"], np.arange(total))
    for arm in arms:
        energy = saved[arm + "_energy"]
        if energy.shape != (total,) or np.any(~np.isfinite(energy) | (energy < 0)):
            raise ValueError("invalid cluster energy: " + arm)
        for field in FIELDS[1:]:
            value = saved[arm + "_" + field]
            if value.shape != (total,) or not np.all(np.isfinite(value[energy > 0])):
                raise ValueError("invalid positive-cluster coordinate: " + arm + "_" + field)


def validate_references(prepared, baseline, k8):
    total = len(prepared["full_ids"])
    for saved in (baseline, k8):
        validate_clusters(saved, total)
        pd.exact("full_ids", prepared["full_ids"], saved["full_ids"])
        pd.exact("query_event_rows", prepared["entries"]["event_row"], saved["query_event_rows"])
        for prefix, values in (("entry_", prepared["entries"]), ("timing_", prepared["timing"])):
            for name, value in values.items():
                pd.exact(prefix + name, value, saved[prefix + name])
        for name, value in prepared["geometry"].items():
            if "geometry_" + name in saved:
                pd.exact("geometry_" + name, value, saved["geometry_" + name])
    for name, value in baseline.items():
        target = "baseline_" + name if name.startswith(("geometric_", "crystal_", "sampling_")) else name
        other = k8[target]
        if value.dtype != other.dtype or not np.array_equal(value, other, equal_nan=value.dtype.kind in "fc"):
            raise ValueError("reused baseline mismatch: " + name)


def validate_sources(inputs, prior, support):
    root = Path(__file__).resolve().parents[1]
    critical = [name for name in prior["calo_source_sha256"] if "/" not in name]
    critical += ["showers/" + name + ".py" for name in (
        "library", "response", "crystal_placement", "reconstruct", "production_compare",
        "placement_diagnostic", "compare", "validity_report")]
    for name in critical:
        checked_file(inputs, root / name, prior["calo_source_sha256"][name])
    for name, sha in support["source_sha256"].items():
        relative = "showers/candidate_support.py" if name == "__main__" else name.removeprefix("calofastsim.").replace(".", "/") + ".py"
        checked_file(inputs, root / relative, sha)
    sources = {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob("*.py"))}
    for name, sha in sources.items():
        checked_file(inputs, root / name, sha)
    return sources


def authenticate(args):
    inputs = source_overlap_report.Inputs()
    candidate = auth.pinned_json(inputs, args.candidate / "summary.json", PINS["candidate"])
    support = auth.pinned_json(inputs, args.support / "summary.json", PINS["support"])
    prior = auth.pinned_json(inputs, args.k8 / "preparation.json", PINS["k8_manifest"])
    old_report = auth.pinned_json(inputs, args.k8 / "numbers.json", PINS["k8_numbers"])
    if old_report["manifest"] != prior or (prior["neighbors"], prior["seed"], prior["reco_seed"]) != (NEIGHBORS, SEED, RECO_SEED):
        raise ValueError("historical fixed trial manifest mismatch")
    if (support["sampling_performed"] is not False or support["reconstruction_performed"] is not False or
            support["candidate_query_overlap"]["shared_parent_events"] != 0 or
            support["candidate_query_overlap"]["shared_output_events"] != 0):
        raise ValueError("support receipt lacks disjoint unsampled candidate")
    for name, sha in candidate["output_sha256"].items():
        if Path(name).name != name:
            raise ValueError("unsafe candidate artifact name")
        checked_file(inputs, args.candidate / name, sha)
    pools = auth.pinned_npz(inputs, args.support / "candidate_pools.npz", support["output_sha256"]["candidate_pools.npz"])
    checked_file(inputs, args.tables, support["input_sha256"]["query_tables"])
    packed, metadata = library.load(args.candidate / "library.npz")
    if metadata != inputs.json(args.candidate / "metadata.json"):
        raise ValueError("candidate metadata mismatch")
    baseline = auth.pinned_npz(inputs, args.baseline / "clusters.npz", prior["inputs"]["clusters"]["sha256"])
    k8 = auth.pinned_npz(inputs, args.k8 / "clusters.npz", PINS["k8_clusters"])
    previous = auth.pinned_json(inputs, args.baseline / "numbers.json", prior["inputs"]["baseline_report"]["sha256"])
    if previous["audit"]["timing"] != pc.TIMING:
        raise ValueError("historical timing differs")
    for name in CONDITIONS:
        checked_file(inputs, args.reco_inputs / name, prior["inputs"][name]["sha256"])
    sources = validate_sources(inputs, prior, support)
    entries, original, geometry, census, total = export.read(args.tables, input_mode="primary")
    with uproot.open(args.tables) as source:
        full_ids = export.keys(export.columns(source, "events", export.EVENT), export.EVENT)
    auth.geometry_parity(args.tables, geometry, packed)
    prepared = dict(entries=entries, original=original, geometry=geometry, full_ids=full_ids,
                    timing=onspill_timing(full_ids["event"], **pc.TIMING), census=census)
    validate_references(prepared, baseline, k8)
    pd.exact("pool event rows", entries["event_row"], pools["event_row"])
    if len(total) != 2813 or len(entries["event_row"]) != 2396 or not np.all(pools["supported"]) or np.any(pools["outside"]):
        raise ValueError("not the predeclared complete file0 query population")
    original_placed = place(packed, original)
    if not np.all(original_placed["mapped"]):
        raise ValueError("original CSS does not map completely")
    pd.exact("original CSS crystals", original["crystal"], original_placed["crystal"])
    inputs.unchanged()
    return dict(inputs=inputs, library=packed, prepared=prepared, baseline=baseline, k8=k8,
                pools=pools, prior=prior, sources=sources)


def validate_draw(packed, entries, draw, pools):
    n = len(entries["event_row"])
    for name, key in (("supported", "supported"), ("fallback", "fallback"),
                      ("radius_pool_size", "size"), ("bin_pool_size", "full_size")):
        pd.exact("draw " + name, draw[name], pools[key])
    donors = np.asarray(draw["donor"])
    if donors.dtype.kind not in "iu" or donors.shape != (n,) or np.any(donors < 0):
        raise ValueError("invalid supported donor indices")
    if not np.all(np.any(pools["eligible"] == donors[:, None], axis=1)):
        raise ValueError("donor outside frozen eligible pool")
    lengths = np.diff(packed["offsets"])[donors]
    offsets = np.r_[0, np.cumsum(lengths)]
    pd.exact("draw offsets", draw["offsets"], offsets)
    row = np.repeat(np.arange(n), lengths)
    pd.exact("draw step entries", draw["entry"], row)
    source = packed["offsets"][donors[row]] + np.arange(len(row)) - offsets[row]
    for name, value in packed.items():
        if name.startswith("step_") and name not in ("step_relative_position", "step_relative_time"):
            pd.exact("whole donor payload " + name, draw[name.removeprefix("step_")], value[source])
    pd.exact("donor time", draw["time"], entries["time"][row] + packed["step_relative_time"][source])
    if draw["position"].shape != (len(row), 3) or not np.all(np.isfinite(draw["position"])):
        raise ValueError("invalid draw position")
    pd.exact("draw empty", lengths == 0, pools["outcome_empty"][donors])
    actual = pd.energy_sum(draw, n)
    if not np.allclose(actual, pools["outcome_energy"][donors], rtol=0, atol=1e-10):
        raise ValueError("draw whole-event energy mismatch")


def runtime(prior):
    import jax
    from ..templatefit import FitConfig

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    value = dict(backend=jax.default_backend(), x64=bool(jax.config.jax_enable_x64),
        versions={k: importlib.metadata.version(k) for k in ("numpy", "jax", "jaxlib", "scipy", "uproot")},
        fit_config=asdict(FitConfig()))
    if value != prior["runtime"] or value["backend"] != "cpu" or FitConfig().iterations != 15:
        raise ValueError("runtime differs from fixed CPU/x64/native fitter reference")
    return value


def save_npz(path, values):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **values)


def run_calls(out, prepared, baseline, resources):
    """Persist each arm before advancing; a started marker prohibits implicit retry."""
    import jax
    from .. import chain, stockio
    from .reconstruct import leading, steps_for_reco

    sim, pulse, noise, propagation, geometry = resources
    reserved = ["clusters.npz", *[arm + suffix for arm in ARMS for suffix in ("_started.json", "_completed.json", ".npz")]]
    if any((out / name).exists() for name in reserved):
        raise ValueError("an arm has already started; no reconstruction replay permitted")
    output = {name: value for name, value in baseline.items() if not name.startswith(("geometric_", "crystal_", "sampling_"))}
    output.update({"sampling_" + k: prepared["draw"][k] for k in
                   ("donor", "supported", "fallback", "offsets", "radius_pool_size", "bin_pool_size")})
    runs = []
    for arm in ARMS:
        steps = steps_for_reco(dict(entries=prepared["entries"], steps=prepared["placed"][arm]), sim)
        with (out / (arm + "_started.json")).open("x") as stream:
            json.dump(dict(arm=arm, seed=RECO_SEED, retry_authorized=False), stream)
        jax.effects_barrier()
        start = time.perf_counter()
        reco = chain.run_chain(steps, prepared["timing"], lambda sipm: stockio.adc_per_pe(sim, sipm),
            pulse, noise, geometry, np.random.default_rng(RECO_SEED), table=propagation, adc2mev=stockio.adc_to_mev)
        jax.effects_barrier()
        clusters = leading(reco["clusters"], len(prepared["full_ids"]))
        save_npz(out / (arm + ".npz"), clusters)
        runs.append(dict(arm=arm, seconds=time.perf_counter() - start, stages=reco["seconds"],
            n_refit_skipped=reco["n_refit_skipped"], n_nonfinite_reco=reco["n_nonfinite_reco"],
            nonfinite_examples=reco.get("nonfinite_examples")))
        pc._json(out / (arm + "_completed.json"), runs[-1])
        output.update({arm + "_" + k: v for k, v in clusters.items()})
        validate_clusters(output, len(prepared["full_ids"]), arms=(arm,))
        print(json.dumps({k: runs[-1][k] for k in ("arm", "seconds", "n_refit_skipped", "n_nonfinite_reco")}), flush=True)
    validate_clusters(output, len(prepared["full_ids"]))
    save_npz(out / "clusters.npz", output)
    return output, runs


def publish(out, state, output, report_out=None):
    from .candidate_report import report

    target = out if report_out is None else report_out
    if target != out and target.exists():
        raise ValueError("recovery report destination must be new")
    result, arrays = report(state["library"], state["prepared"], state["baseline"], state["k8"], output, state["pools"])
    state["inputs"].unchanged()
    if target != out:
        target.mkdir(parents=True, exist_ok=False)
    pc._json(target / "numbers.json", result)
    save_npz(target / "diagnostics.npz", arrays)
    artifacts = {p.name: digest(p) for p in sorted(target.iterdir()) if p.is_file() and p.name != "summary.json"}
    pc._json(target / "summary.json", dict(completed=True, sampler_calls=1, reconstruction_calls=2,
        pins=PINS, input_sha256=state["inputs"].sha256, output_sha256=artifacts, scope=SCOPE))
    return result


def saved_report(args):
    """Finish reporting from a caller-pinned execution receipt without sampling or reconstruction."""
    if not args.execution_sha256:
        raise ValueError("saved reporting requires an external execution SHA256")
    report_out = getattr(args, "report_out", None)
    if report_out is None and any((args.out / name).exists() for name in ("numbers.json", "diagnostics.npz", "summary.json")):
        raise ValueError("report output already exists; use a new --report-out for saved-only recovery")
    if report_out is not None and report_out.exists():
        raise ValueError("recovery report destination must be new")
    state = authenticate(args)
    inputs = state["inputs"]
    execution = auth.pinned_json(inputs, args.out / "execution.json", args.execution_sha256)
    if execution["draws"] != 1 or execution["calls"] != 2 or [r["arm"] for r in execution["runs"]] != list(ARMS):
        raise ValueError("not a completed two-call execution")
    for name, sha in execution["output_sha256"].items():
        if Path(name).name != name:
            raise ValueError("unsafe execution artifact name")
        checked_file(inputs, args.out / name, sha)
    manifest = inputs.json(args.out / "preparation.json")
    if manifest["pins"] != PINS or (manifest["neighbors"], manifest["seed"], manifest["reco_seed"]) != (NEIGHBORS, SEED, RECO_SEED):
        raise ValueError("saved trial settings mismatch")
    draw = inputs.npz(args.out / "draw.npz")
    placed = {arm: inputs.npz(args.out / (arm + "_placement.npz")) for arm in ARMS}
    validate_draw(state["library"], state["prepared"]["entries"], draw, state["pools"])
    if fingerprint(draw) != manifest["fingerprints"]["draw"] or any(
            fingerprint(value) != manifest["fingerprints"]["placements"][arm] for arm, value in placed.items()):
        raise ValueError("saved trial payload fingerprint mismatch")
    state["prepared"].update(draw=draw, placed=placed)
    output = inputs.npz(args.out / "clusters.npz")
    validate_clusters(output, len(state["prepared"]["full_ids"]))
    for name, value in state["baseline"].items():
        if not name.startswith(("geometric_", "crystal_", "sampling_")) and not np.array_equal(
                value, output[name], equal_nan=value.dtype.kind in "fc"):
            raise ValueError("saved report reference mismatch: " + name)
    return publish(args.out, state, output, report_out=report_out)


def run(args):
    if getattr(args, "report_only", False):
        return saved_report(args)
    if not args.validate_only and args.out.exists():
        raise ValueError("trial output exists; no draw or reconstruction replay permitted")
    state = authenticate(args)
    value = runtime(state["prior"])
    if args.validate_only:
        print(json.dumps(dict(validated=True, sampled=False, reconstructed=False, runtime=value)), flush=True)
        return state
    from .. import stockio
    from ..cluster import Geometry
    from . import candidate_report, reconstruct

    if not callable(candidate_report.report) or not callable(reconstruct.steps_for_reco):
        raise RuntimeError("report/reconstruction dependencies unavailable")
    resources = (stockio.load_sim_crystal(args.reco_inputs / CONDITIONS[0]),
                 stockio.load_pulse(args.reco_inputs / CONDITIONS[1]),
                 stockio.load_noise(args.reco_inputs / CONDITIONS[2]),
                 stockio.load_propagation(args.reco_inputs / CONDITIONS[3]), Geometry.load(args.tables))
    state["inputs"].unchanged()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = dict(pins=PINS, neighbors=NEIGHBORS, seed=SEED, reco_seed=RECO_SEED, runtime=value, scope=SCOPE,
                    input_sha256=state["inputs"].sha256, calo_source_sha256=state["sources"], timing=pc.TIMING)
    pc._json(args.out / "preparation.json", manifest)
    prepared = state["prepared"]
    rng = np.random.default_rng(SEED)
    manifest["draw_rng_before"] = rng.bit_generator.state
    draw = library.sample(state["library"], prepared["entries"], rng, orientation="disk", radius_neighbors=NEIGHBORS)
    save_npz(args.out / "draw.npz", draw)
    manifest["draw_rng_after"] = rng.bit_generator.state
    validate_draw(state["library"], prepared["entries"], draw, state["pools"])
    prepared.update(draw=draw, placed=dict(geometric=place(state["library"], draw),
                    crystal=place_crystals(state["library"], prepared["entries"], draw)))
    for arm, steps in prepared["placed"].items():
        save_npz(args.out / (arm + "_placement.npz"), steps)
    manifest["fingerprints"] = dict(draw=fingerprint(draw), placements={k: fingerprint(v) for k, v in prepared["placed"].items()})
    pc._json(args.out / "preparation.json", manifest)
    output, runs = run_calls(args.out, prepared, state["baseline"], resources)
    pc._json(args.out / "execution.json", dict(draws=1, calls=2, runs=runs,
        output_sha256={p.name: digest(p) for p in sorted(args.out.iterdir()) if p.is_file()}))
    return publish(args.out, state, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "support", "tables", "baseline", "k8", "reco-inputs", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true", help="authenticate without creating output, drawing or reconstructing")
    mode.add_argument("--report-only", action="store_true", help="read completed execution only; never draw or reconstruct")
    parser.add_argument("--execution-sha256", help="external receipt pin required by --report-only")
    parser.add_argument("--report-out", type=Path, help="new saved-only recovery directory; never overwrites partial reports")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
