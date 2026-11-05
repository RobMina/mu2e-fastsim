"""Check deterministic stock response by persistent saved-step pointers, never by fit values."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from calofastsim import response
from calofastsim.compression import lookup
from calofastsim.identities import EVENT_FIELDS, keys


STEP_FIELDS = EVENT_FIELDS + ("product_id", "index")
RO_FIELDS = STEP_FIELDS + ("sipm",)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configuration(path):
    text = Path(path).read_text()
    unquoted = re.sub(r'"[^"\n]*"|#[^\n]*', '""', text)
    if re.search(r"\b\w+(?:\.\w+)+\s*:", unquoted) or "@" in unquoted:
        raise ValueError("canonical expanded FHiCL without dotted overrides or references required")
    blocks = re.findall(r"\bdeterministicRO\s*:\s*\{([^{}]*)\}", text)
    if len(blocks) != 1:
        raise ValueError("one fully expanded deterministicRO block required")
    block = blocks[0]
    timing_blocks = re.findall(r"\bcontrolEWM\s*:\s*\{([^{}]*)\}", text)
    if len(timing_blocks) != 1:
        raise ValueError("one explicit fixed-phase controlEWM block required")
    for name, value in (("module_type", '"EventWindowMarkerProducer"'), ("SpillType", "1"),
                        ("InitialPhaseShift", "0"), ("RecoFromMCTruth", "true"), ("RecoFromMCTruthErr", "0")):
        if re.findall(r"\b" + name + r"\s*:\s*([^\s]+)", timing_blocks[0]) != [value]:
            raise ValueError(f"fixed-phase unsmeared control timing requires {name}={value}")
    for name, value in (("module_type", '"CaloShowerROMaker"'),
                        ("PEStatCorrection", "false"), ("addTravelTime", "false")):
        found = re.findall(r"\b" + name + r"\s*:\s*([^\s]+)", block)
        if found != [value]:
            raise ValueError(f"deterministic stock control requires {name}={value}")
    result = {}
    for name, target in (("BirksCorrection", "birks"), ("LRUCorrection", "longitudinal")):
        found = re.findall(r"\b" + name + r"\s*:\s*(true|false)\b", block)
        if len(found) != 1:
            raise ValueError(f"one resolved {name} required")
        result[target] = found[0] == "true"
    for name, target in (("digitizationStart", "digitization_start"),
                         ("digitizationEnd", "digitization_end"),
                         ("digitizationBuffer", "digitization_buffer")):
        found = re.findall(r"\b" + name + r"\s*:\s*([^\s]+)", block)
        if len(found) != 1 or not np.isfinite(float(found[0])):
            raise ValueError(f"one finite resolved {name} required")
        result[target] = float(found[0])
    tags = re.findall(r'\bcaloShowerStepCollection\s*:\s*\[\s*"([^"\n]+)"\s*\]', block)
    if tags != ["CaloShowerStepMaker::CaloLibraryCE"]:
        raise ValueError("single explicit original saved-step input required")
    for name in ("eventWindowMarker", "protonBunchTimeMC"):
        if re.findall(r"\b" + name + r'\s*:\s*"([^"\n]+)"', block) != ["controlEWM::CaloResponseControl"]:
            raise ValueError("explicit control event timing inputs required")
    return result


def exact(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        return {"exact": False, "shape_equal": False}
    diff = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {"exact": bool(np.array_equal(left, right)), "shape_equal": True,
            "mismatches": int(np.count_nonzero(left != right)),
            "max_abs_difference": float(np.max(diff, initial=0))}


def validate_rows(events, table, count_name, product_name):
    event_index = lookup(keys(events, EVENT_FIELDS), keys(table, EVENT_FIELDS))
    counts = np.bincount(event_index, minlength=len(events["event"]))
    if not np.array_equal(counts, events[count_name]):
        raise ValueError(f"{count_name} does not reconcile with all exported event rows")
    if (np.any(table["product_id"] == 0) or
            not np.array_equal(table["product_id"], events[product_name][event_index])):
        raise ValueError("product registry mismatch or zero ProductID")
    row_keys = keys(table, STEP_FIELDS)
    if len(np.unique(row_keys)) != len(row_keys):
        raise ValueError("duplicate output row identity")
    expected = np.arange(len(event_index)) - np.repeat(np.cumsum(counts) - counts, counts)
    if not np.array_equal(table["index"], expected) or np.any(np.diff(event_index) < 0):
        raise ValueError("rows must be contiguous and ordered within exported events")
    return event_index


def compare(events, steps, readouts, summaries, *, options, expected_events, backend=response):
    """Compare observed RO/summaries; arithmetic intermediates below are predictions, not stock emissions."""
    event_keys = keys(events, EVENT_FIELDS)
    if len(event_keys) != expected_events or len(np.unique(event_keys)) != expected_events:
        raise ValueError("independent expected event count or event uniqueness failed")
    for name in ("step_product_id", "ro_product_id", "summary_product_id", "ewm_product_id", "pb_product_id"):
        if np.any(np.asarray(events[name]) <= 0):
            raise ValueError("zero event ProductID")
    for name in ("sim_iov", "timing_iov"):
        iov = np.asarray(events[name])
        if iov.shape != (expected_events, 4) or not np.issubdtype(iov.dtype, np.integer):
            raise ValueError("missing conditions interval")
        start_ok = (events["run"] > iov[:, 0]) | ((events["run"] == iov[:, 0]) & (events["subrun"] >= iov[:, 1]))
        end_ok = (events["run"] < iov[:, 2]) | ((events["run"] == iov[:, 2]) & (events["subrun"] <= iov[:, 3]))
        if not np.all(start_ok & end_ok):
            raise ValueError("event outside fetched conditions interval")
    si = validate_rows(events, steps, "n_steps", "step_product_id")
    validate_rows(events, readouts, "n_readouts", "ro_product_id")
    validate_rows(events, summaries, "n_summaries", "summary_product_id")
    keys(steps, ("sim_product_id", "sim_key", "crystal", "sipm0"))
    if np.any(steps["sim_product_id"] == 0) or np.any((events["spill"] != 0) & (events["spill"] != 1)):
        raise ValueError("invalid SimParticle identity or spill type")
    for name in ("energy_g4", "energy_birks", "z", "lru", "yield0", "yield1"):
        if not np.all(np.isfinite(steps[name])):
            raise ValueError("nonfinite input state or conditions")
    timing = {name: events[field][si] for name, field in (
        ("pbtime", "pbtime"), ("mbtime", "mbtime"), ("event_length", "event_length"),
        ("time_from_protons_to_dr_marker", "marker"))}
    timing.update(onspill=events["spill"][si] == 1,
                  **{name: options[name] for name in ("digitization_start", "digitization_end", "digitization_buffer")})
    response.fold_step_time(steps["time"], **timing)
    folded, accepted = backend.fold_step_time(steps["time"], **timing)
    folded, accepted = np.asarray(folded), np.asarray(accepted)
    length = events["crystal_length"][si]
    if np.any(~np.isfinite(length) | (length <= 0)):
        raise ValueError("invalid fetched crystal length")
    corrected = np.zeros(len(si), dtype=np.float32)
    corrected[accepted] = np.asarray(backend.corrected_energy(
        steps["energy_g4"][accepted], steps["energy_birks"][accepted], steps["z"][accepted],
        length[accepted], steps["lru"][accepted], birks=options["birks"], longitudinal=options["longitudinal"]))
    yields = np.column_stack([steps["yield0"], steps["yield1"]])
    means = np.asarray(backend.photoelectron_mean(corrected[accepted], yields[accepted]))
    response.nonnegative_lround(means)
    npe = np.zeros((len(si), 2), dtype=np.int32)
    npe[accepted] = np.asarray(backend.nonnegative_lround(means))
    present = accepted[:, None] & (npe > 0)
    predicted = {name: np.repeat(steps[name], 2)[present.ravel()] for name in STEP_FIELDS}
    predicted["sipm"] = (steps["sipm0"][:, None] + np.arange(2))[present]
    observed = {name: readouts[name] for name in EVENT_FIELDS}
    observed.update(product_id=readouts["step_product_id"], index=readouts["step_index"], sipm=readouts["sipm"])
    predicted_keys, observed_keys = keys(predicted, RO_FIELDS), keys(observed, RO_FIELDS)
    if len(np.unique(observed_keys)) != len(observed_keys):
        raise ValueError("duplicate stock readout for a saved step and SiPM")
    layout = np.array_equal(np.sort(predicted_keys), np.sort(observed_keys))
    report = {"events": expected_events, "saved_steps": len(si), "accepted_steps": int(np.count_nonzero(accepted)),
              "time_rejected_steps": int(np.count_nonzero(~accepted)),
              "accepted_zero_pe_steps": int(np.count_nonzero(accepted & ~np.any(npe > 0, axis=1))),
              "empty_saved_step_events": int(np.count_nonzero(events["n_steps"] == 0)),
              "empty_readout_events": int(np.count_nonzero(events["n_readouts"] == 0)),
              "observed_summaries": len(summaries["event"]),
              "predicted_readouts": len(predicted_keys), "observed_readouts": len(observed_keys),
              "exact_step_sipm_set": bool(layout)}
    counts = ak.to_numpy(ak.num(readouts["pe_times"]))
    times = ak.to_numpy(ak.flatten(readouts["pe_times"]))
    report["observed_photoelectrons"] = int(np.sum(counts, dtype=np.int64))
    if np.any(counts <= 0) or not np.all(np.isfinite(times)):
        raise ValueError("empty or nonfinite observed readout")
    if layout:
        order = lookup(predicted_keys, observed_keys)
        report["npe"] = exact(npe[present][order], counts)
        predicted_time = np.repeat(folded, 2)[present.ravel()][order].astype(np.float32)
        report["pe_times"] = exact(np.repeat(predicted_time, counts), times)
    member_counts = ak.to_numpy(ak.num(summaries["member_indices"]))
    if np.any(member_counts <= 0) or not np.array_equal(member_counts, ak.to_numpy(ak.num(summaries["member_products"]))):
        raise ValueError("invalid summary member offsets")
    group = np.repeat(np.arange(len(member_counts)), member_counts)
    members = {name: np.repeat(summaries[name], member_counts) for name in EVENT_FIELDS}
    members.update(product_id=ak.to_numpy(ak.flatten(summaries["member_products"])),
                   index=ak.to_numpy(ak.flatten(summaries["member_indices"])))
    mi = lookup(keys(steps, STEP_FIELDS), keys(members, STEP_FIELDS))
    member_set = np.array_equal(np.sort(mi), np.flatnonzero(accepted))
    if not member_set:
        raise ValueError("summary members are not exactly all accepted saved steps once")
    if np.any((np.diff(group) == 0) & (np.diff(mi) <= 0)):
        raise ValueError("stock summary member order differs from saved input order")
    group_key = keys(steps, EVENT_FIELDS + ("sim_product_id", "sim_key", "crystal"))
    offsets = np.cumsum(member_counts) - member_counts
    if (not np.array_equal(group_key[mi], np.repeat(group_key[mi[offsets]], member_counts)) or
            len(np.unique(group_key[mi[offsets]])) != len(member_counts)):
        raise ValueError("summary splits or mixes exact particle/crystal groups")
    energy = np.zeros(len(member_counts), dtype=np.float32)
    np.add.at(energy, group, corrected[mi])
    time = np.minimum.reduceat(folded[mi].astype(np.float32), offsets) if len(offsets) else np.empty(0, np.float32)
    order_table = {name: steps[name][mi[offsets]] for name in EVENT_FIELDS + ("crystal", "sim_product_id", "sim_key")}
    order_keys = keys(order_table, EVENT_FIELDS + ("crystal", "sim_product_id", "sim_key"))
    report["summary_group_order_exact"] = bool(np.array_equal(order_keys, np.sort(order_keys)))
    report["summary_energy"] = exact(energy, summaries["energy"])
    report["summary_time"] = exact(time, summaries["time"])
    report["all_observed_deterministic_fields_exact"] = bool(layout and report["summary_group_order_exact"] and all(
        report[name]["exact"] for name in ("npe", "pe_times", "summary_energy", "summary_time")))
    report["conditions"] = {name: np.unique(events[name]).tolist() for name in
        ("spill", "event_length", "pbtime", "mbtime", "marker", "crystal_length")}
    report["conditions"].update({name: np.unique(steps[name]).tolist() for name in ("lru", "yield0", "yield1")})
    report["conditions"].update({name: np.unique(events[name], axis=0).tolist() for name in ("sim_iov", "timing_iov")})
    report["conditions"].update({name: np.unique(ak.to_numpy(ak.flatten(events[name]))).tolist()
                                 for name in ("sim_cids", "timing_cids") if name in events})
    report.update(stock_per_step_intermediates_observed=False, stochastic_response_certified=False,
                  summary_accumulation_backend="numpy_sequential_float32_using_stock_member_order",
                  digitization_reconstruction_certified=False, unbiased_library_certified=False)
    return report


def read_tables(path):
    with uproot.open(path) as root:
        result = {}
        for name in ("events", "steps", "readouts", "summaries"):
            tree = root[f"response/{name}"]
            arrays = tree.arrays(library="ak")
            result[name] = {field: arrays[field] if field in ("pe_times", "member_products", "member_indices", "sim_cids", "timing_cids")
                            else ak.to_numpy(arrays[field]) for field in arrays.fields}
        return result


def reconcile(tables, reference, expected_events):
    with uproot.open(reference) as root:
        events = root["dump/events"].arrays(library="np")
        saved = root["dump/showers"].arrays(library="np")
    if (len(events["event"]) != expected_events or np.any(events["g4_status"] != 0) or
            not np.array_equal(keys(tables["events"], EVENT_FIELDS), keys(events, EVENT_FIELDS))):
        raise ValueError("control must retain all independently audited status-zero events in order")
    steps = tables["steps"]
    for name in STEP_FIELDS + ("sim_product_id", "sim_key", "crystal", "energy_g4", "energy_birks", "time"):
        if not np.array_equal(steps[name], saved[name]):
            raise ValueError(f"control saved steps differ from independent export in {name}")
    if not np.array_equal(steps["z"], saved["position_z"].astype(np.float32)):
        raise ValueError("source float local-z narrowing differs from independently saved position")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--expanded-fcl", type=Path, required=True)
    parser.add_argument("--expected-events", type=int, required=True)
    parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    parser.add_argument("--jax-jit", action="store_true", help="compile the four JAX response primitives; retain exact failures")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.expected_events <= 0:
        raise ValueError("positive independent expected count required")
    if args.jax_jit and args.backend != "jax":
        raise ValueError("--jax-jit requires --backend jax")
    options = configuration(args.expanded_fcl)
    tables = read_tables(args.input)
    reconcile(tables, args.reference, args.expected_events)
    environment = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                   "awkward": ak.__version__, "uproot": uproot.__version__}
    if args.backend == "jax":
        import jax
        from calofastsim import response_jax
        with jax.enable_x64():
            from types import SimpleNamespace
            backend = response_jax
            if args.jax_jit:
                backend = SimpleNamespace(**{name: jax.jit(getattr(response_jax, name)) for name in
                    ("fold_step_time", "corrected_energy", "photoelectron_mean", "nonnegative_lround")})
            environment.update(jax=jax.__version__, jax_x64=bool(jax.config.x64_enabled),
                               jax_devices=[str(device) for device in jax.devices()],
                               execution="jit_primitives" if args.jax_jit else "eager_primitives_not_jit")
            report = compare(**tables, options=options, expected_events=args.expected_events, backend=backend)
    else:
        report = compare(**tables, options=options, expected_events=args.expected_events)
    report["backend"] = args.backend
    report["environment"] = environment
    report["checker_sha256"] = sha256(Path(__file__))
    report["backend_sha256"] = sha256(Path(response_jax.__file__ if args.backend == "jax" else response.__file__))
    report.update(input_sha256=sha256(args.input), reference_sha256=sha256(args.reference),
                  expanded_fcl_sha256=sha256(args.expanded_fcl), options=options,
                  independent_event_and_saved_step_reconciliation=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if not report["all_observed_deterministic_fields_exact"]:
        raise SystemExit("deterministic stock response comparison failed; report retained")


if __name__ == "__main__":
    main()
