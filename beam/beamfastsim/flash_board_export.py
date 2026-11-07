"""Check aggregated FlashBoardExporter tables and evaluate the board observable.

This reads exported accessor tables, not art or raw G4 steps. Normalization is
withheld unless the caller explicitly declares complete selected SubRun input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from . import flash_board_observable as observable
from . import flash_board_selection as selection
from .flash_inclusive_check import EVENT, PRODUCT, keys, join

PROVENANCE = {k: "str" for k in ("tag", "label", "instance", "process", "branch")}
EID = {k: "u4" for k in EVENT}
STEP = {**EID, **{k: "u8" for k in ("product", "index", "particle_product", "particle_key")},
        "momentum_mev_c": "f4"}
SCHEMA = {
    "schema": {"version": "u4", "expected_events": "u8", "expected_subruns": "u8"},
    "requirements": {"detector": "u4", "slot": "u4", "tag": "str"},
    "events": {**EID, **{k: "u8" for k in ("nproducts", "nstraw", "ncalo", "ncrv")}},
    "products": {**EID, "product": "u8", "detector": "u4", "slot": "u4", "size": "u8", **PROVENANCE},
    "straw_steps": {**STEP, "ionizing_mev": "f4"},
    "calo_steps": {**STEP, "birks_mev": "f4"},
    "crv_steps": {**STEP, "visible_mev": "f4"},
    "subruns": {"run": "u4", "subrun": "u4", "product": "u8", "generated_events": "u8", **PROVENANCE},
    "completion": {"complete": "u4", **{k: "u8" for k in
                   ("events", "subruns", "straw_steps", "calo_steps", "crv_steps")}},
}


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _table(table, spec, name):
    if missing := set(spec) - table.keys():
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    out = {}
    for field, dtype in spec.items():
        value = table[field]
        if np.ma.isMaskedArray(value):
            raise ValueError(f"{name}.{field}: masked column")
        value = np.asarray(value)
        if value.ndim != 1:
            raise ValueError(f"{name}.{field}: expected one-dimensional column")
        if dtype == "str":
            if value.dtype.kind != "U":
                raise ValueError(f"{name}.{field}: expected Unicode string column")
        else:
            expected = np.dtype(dtype)
            if value.dtype.kind != expected.kind or value.dtype.itemsize != expected.itemsize:
                raise ValueError(f"{name}.{field}: expected {dtype}, got {value.dtype}")
        out[field] = value
    if len({len(a) for a in out.values()}) != 1:
        raise ValueError(f"{name}: unequal column lengths")
    return out


def _provenance(table, name):
    for field in ("tag", "label", "process", "branch"):
        if np.any(np.char.str_len(table[field]) == 0):
            raise ValueError(f"{name}: empty {field}")
    if np.any(table["product"] == 0):
        raise ValueError(f"{name}: invalid zero ProductID")
    first = np.char.partition(table["tag"], ":")
    rest = np.char.partition(first[:, 2], ":")
    if np.any(np.char.count(table["tag"], ":") > 2):
        raise ValueError(f"{name}: malformed configured InputTag")
    if (np.any(first[:, 0] != table["label"]) or np.any(rest[:, 0] != table["instance"])
            or np.any((rest[:, 2] != "") & (rest[:, 2] != table["process"]))):
        raise ValueError(f"{name}: configured tag disagrees with resolved provenance")


def validate(tables, *, expected_events, expected_subruns):
    """Validate the v1 export's typed counts and identities, not physics completeness."""
    expected_events = _positive_int(expected_events, "expected_events")
    expected_subruns = _positive_int(expected_subruns, "expected_subruns")
    if missing := set(SCHEMA) - tables.keys():
        raise ValueError(f"Missing export trees: {sorted(missing)}")
    t = {name: _table(tables[name], spec, name) for name, spec in SCHEMA.items()}
    schema, complete, events = t["schema"], t["completion"], t["events"]
    if len(schema["version"]) != 1 or schema["version"][0] != 1:
        raise ValueError("Expected exactly one schema v1 row")
    if (int(schema["expected_events"][0]) != expected_events
            or int(schema["expected_subruns"][0]) != expected_subruns):
        raise ValueError("Export expectation differs from independently supplied counts")
    if len(complete["complete"]) != 1 or complete["complete"][0] != 1:
        raise ValueError("Missing or invalid endJob completion marker")
    for name in ("events", "subruns", "straw_steps", "calo_steps", "crv_steps"):
        if int(complete[name][0]) != len(next(iter(t[name].values()))):
            raise ValueError(f"Completion count differs for {name}")
    if len(events["event"]) != expected_events or len(t["subruns"]["run"]) != expected_subruns:
        raise ValueError("Export row count differs from expected events/SubRuns")
    ek = keys(events, EVENT)
    join(ek, ek)
    p, req = t["products"], t["requirements"]
    _provenance(p, "products")
    _provenance(t["subruns"], "subruns")
    branches = np.rec.fromarrays([p[k] for k in (*EVENT, "branch")], names=(*EVENT, "branch"))
    if len(np.unique(branches)) != len(branches):
        raise ValueError("Duplicate resolved branch within an event")
    if len(np.unique(t["subruns"]["tag"])) != 1:
        raise ValueError("Configured counter tag changes between SubRuns")
    rk = keys(req, ("detector", "slot"))
    join(rk, rk)
    if not np.isin(req["detector"], (selection.TRACKER, selection.CALO, selection.CRV)).all():
        raise ValueError("Unknown requirement detector")
    for role in (selection.TRACKER, selection.CALO, selection.CRV):
        rows = req["detector"] == role
        if not rows.any() or not np.array_equal(np.sort(req["slot"][rows]), np.arange(rows.sum())):
            raise ValueError("Requirement slots must be contiguous for each detector")
        if np.any(np.char.str_len(req["tag"][rows]) == 0) or len(np.unique(req["tag"][rows])) != rows.sum():
            raise ValueError("Empty or duplicate configured requirement tag")
    ri = join(keys(p, ("detector", "slot")), rk)
    if np.any(ri < 0) or np.any(p["tag"] != req["tag"][ri]):
        raise ValueError("Product does not match a configured requirement")
    pk = keys(p, (*EVENT, "detector", "slot"))
    join(pk, pk)
    pe = join(keys(p, EVENT), ek)
    if np.any(pe < 0):
        raise ValueError("Product outside event ledger")
    product_counts = np.bincount(pe, minlength=len(ek))
    if np.any(product_counts != len(rk)):
        raise ValueError("Missing configured detector collection in an event")
    if np.any(product_counts.astype(np.uint64) != events["nproducts"]):
        raise ValueError("Event nproducts mismatch")
    for name, count in (("straw_steps", "nstraw"), ("calo_steps", "ncalo"), ("crv_steps", "ncrv")):
        step = t[name]
        if np.any(step["particle_product"] == 0):
            raise ValueError("Invalid zero particle ProductID")
        se = join(keys(step, EVENT), ek)
        if np.any(se < 0):
            raise ValueError(f"{name}: step outside event ledger")
        if np.any(np.bincount(se, minlength=len(ek)).astype(np.uint64) != events[count]):
            raise ValueError(f"Event {count} mismatch")
    selected = selection.select(events, p, t["straw_steps"], t["calo_steps"], t["crv_steps"])
    empty = {k: v[:0] for k, v in t["straw_steps"].items()}
    observable.summarize(events, empty, t["subruns"])
    return t, selected


def summarize(tables, *, expected_events, expected_subruns, input_scope, complete_subruns=False):
    """Report extraction and selection; Q needs an explicit full-SubRun declaration.

    retained: supplied events are the saved selected stream, so every supplied
    tracker deposit contributes even if diagnostic reselection disagrees.
    filter-input: supplied events are all those presented to the detector filter;
    only events selected here contribute. Neither scope certifies upstream G4.
    """
    if input_scope not in ("retained", "filter-input"):
        raise ValueError("input_scope must be retained or filter-input")
    if type(complete_subruns) is not bool:
        raise ValueError("complete_subruns must be a boolean declaration")
    t, selected = validate(tables, expected_events=expected_events, expected_subruns=expected_subruns)
    retained = t["events"] if input_scope == "retained" else selected["retained_events"]
    straw = t["straw_steps"] if input_scope == "retained" else selected["retained_straw_steps"]
    result = observable.summarize(retained, straw, t["subruns"])
    result.pop("event_tracker_ionizing_mev")
    result.update(
        scope="Aggregated accessor export and historical detector selection; NOT a fast prediction or physics closure",
        input_scope=input_scope, complete_subruns_declared=complete_subruns,
        normalization_available=complete_subruns, exported_events=len(t["events"]["event"]),
        exported_tracker_steps=len(t["straw_steps"]["index"]),
        exported_tracker_ionizing_mev=float(t["straw_steps"]["ionizing_mev"].astype(np.float64).sum()),
        detector_selected_events=int(selected["retained"].sum()),
        detector_rejected_events=int((~selected["retained"]).sum()),
        selected_stream_reselection_agrees=(bool(selected["retained"].all()) if input_scope == "retained" else None),
        detector_accept_counts={name: int(selected[name].sum()) for name in ("tracker", "calo", "crv")},
        recording_checks_passed=True, upstream_transport_validated=False,
    )
    if not np.isfinite(result["exported_tracker_ionizing_mev"]):
        raise ValueError("Exported tracker total is nonfinite")
    if not complete_subruns:
        for name in ("mev_per_generated_event", "retained_fraction", "central_within_budget"):
            result[name] = None
    result["caveats"] = [
        "Caller must supply the complete selected input inventory for every counted SubRun, including zero-retained SubRuns.",
        "An endJob marker/count agreement does not prove scheduler success, closed-file integrity, or upstream simulation completion.",
        "Retained-stream reselection cannot validate events absent from the selected input.",
        "No raw aggregation, source/pulse equivalence, uncertainty, optimizer feasibility or fast-response validation is supplied.",
    ]
    return result


def read_tables(path, *, directory="board"):
    """Read native TTrees using their latest ROOT cycle and preserve scalar types."""
    import awkward as ak
    import uproot

    with uproot.open(path) as root:
        output = {}
        for name, spec in SCHEMA.items():
            tree = root[f"{directory}/{name}"]
            if tree.classname != "TTree":
                raise ValueError(f"{name}: expected TTree, not {tree.classname}")
            arrays = tree.arrays(list(spec), library="ak")
            output[name] = {field: ak.to_numpy(arrays[field]) for field in spec}
        return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--directory", default="board")
    parser.add_argument("--events", required=True, type=int)
    parser.add_argument("--subruns", required=True, type=int)
    parser.add_argument("--input-scope", required=True, choices=("retained", "filter-input"))
    parser.add_argument("--complete-subruns", action="store_true",
                        help="Declare complete selected input for every counted SubRun, not a partial event read")
    args = parser.parse_args(argv)
    start = perf_counter()
    result = summarize(read_tables(args.input, directory=args.directory), expected_events=args.events,
                       expected_subruns=args.subruns, input_scope=args.input_scope,
                       complete_subruns=args.complete_subruns)
    result.update(input=str(args.input), elapsed_seconds=perf_counter() - start)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{result['exported_events']} events, {result['retained_events']} retained; "
          f"Q={result['mev_per_generated_event']}; {result['elapsed_seconds']:.3f} s/file. NOT physics closure.")
    return int(result["selected_stream_reselection_agrees"] is False)


if __name__ == "__main__":
    raise SystemExit(main())
