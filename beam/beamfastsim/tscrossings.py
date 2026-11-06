"""Bounded exact-ID stock crossing extraction; never a material-step trajectory."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import uproot

from .tsreference import IDS, _tree, ids, lookup, read_table, subset

TARGETS = ((1430, 920075, 4321, 1155), (1430, 920071, 2260, 4478))
MAX_TABLE_ROWS = 100_000
LIMITS = [
    "Recorded VD and DS Beam pre-step samples only, not a full stock material trajectory.",
    "No interpolation, spatial matching, material assignment or new simulation.",
    "All exact-ID crossings retained, including repeats and backward crossings; ordered by time.",
    "ntpart is not read: existing exact-ancestry reference verifies source/endpoint identity.",
    "Compact upstream ancestry may be incomplete; no claim of full primary ancestry closure.",
]


def targets_checked(targets):
    """Restrict this diagnostic to the two authorized development mu+ identities."""
    a = np.asarray(targets)
    if a.ndim != 2 or a.shape[1] != 4 or not len(a):
        raise ValueError("nonempty N by 4 exact IDs required")
    a = ids(dict(zip(IDS, a.T)))
    if len(a) > 2 or np.any(lookup(np.asarray(TARGETS), a) < 0):
        raise ValueError("only the two authorized development IDs are allowed")
    lookup(a, a)
    return a


def reference_identity(path, targets):
    """Load only reference columns needed for exact identity and state checks."""
    with np.load(path, allow_pickle=False) as ref:
        key = ref["id"]
        if np.any((key[:, 1] < 920000) | (key[:, 1] >= 920100)):
            raise ValueError("reference must contain development subruns only")
        row = lookup(key, targets)
        if np.any(row < 0):
            raise ValueError("requested source absent from reference")
        fields = ("id", "pdg", "x", "p", "t", "stock_id", "stock_pdg", "stock_x",
                  "stock_p", "stock_t", "stock_reached", "source_parent_ancestry_incomplete")
        selected = {k: ref[k][row] for k in fields}
        if (not np.array_equal(selected["stock_id"], targets)
                or not np.all(selected["stock_reached"])
                or np.any(selected["pdg"] != -13) or np.any(selected["stock_pdg"] != -13)):
            raise ValueError("reference must verify same exact source/endpoint mu+ identity")
        erow = lookup(ref["endpoint_id"], targets)
        if np.any(erow < 0) or np.any(ref["endpoint_ancestry_incomplete"][erow]):
            raise ValueError("reference DS ancestry absent or incomplete")
        selected["endpoint_ancestry_incomplete"] = ref["endpoint_ancestry_incomplete"][erow]
    return selected, {"path": str(path), "source_rows": row.tolist(),
                      "endpoint_rows": erow.tolist(), "same_source_endpoint_id": True,
                      "selected": {k: v.tolist() for k, v in selected.items()}}


def select_exact(table, targets):
    """Vectorized exact-key selection, preserving every scalar column and entry."""
    target = lookup(targets, ids(table))
    keep = np.flatnonzero(target >= 0)
    out = subset(table, keep)
    ids(out, ("pdg", "sid", "code"))
    if np.any(out["pdg"] != -13):
        raise ValueError("exact ID has unexpected PDG (expected mu+, -13)")
    if not np.all(np.isfinite(out["time"])):
        raise ValueError("nonfinite crossing time")
    order = np.lexsort((keep, out["time"], target[keep]))
    out = subset(out, order)
    out["entry"] = keep[order]
    out["target"] = target[keep][order]
    out["momentum_mev_c"] = np.sqrt(sum(out[k].astype(float) ** 2 for k in ("px", "py", "pz")))
    return out


def census_inputs(paths, targets):
    """Validate the entire bounded input plan before opening any ROOT files."""
    plans = []
    for path in paths:
        c = json.loads(Path(path).read_text())
        i = c["index"]
        if (type(i) is not int or not 0 <= i < 100 or c["cohort"] != "development"
                or c["diagnostic_mode"] != "compact" or not c["checks_passed"]):
            raise ValueError("checked compact development census required (shards 0..99)")
        ntuple = Path(c["ntuple"])
        if ntuple.parent.name != f"{i:03d}" or ntuple.parent.parent.name != "shards":
            raise ValueError("census ntuple path does not match its development shard")
        if not np.any(targets[:, 1] == 920000 + i):
            raise ValueError("census outside requested exact-ID shards")
        plans.append((ntuple, c, str(path)))
    if sorted(c["index"] for _, c, _ in plans) != sorted((np.unique(targets[:, 1]) - 920000).tolist()):
        raise ValueError("exactly one census per requested shard required")
    return plans


def read_small(path, tree, expected_rows):
    """Enforce the row budget and highest cycle before reading a scalar table."""
    with uproot.open(path) as f:
        t, _ = _tree(f, tree)
        n = t.num_entries
        if n > MAX_TABLE_ROWS or n != expected_rows:
            raise ValueError(f"{tree}: row budget/census mismatch: {n} vs {expected_rows}")
    a, meta = read_table(path, tree)
    return a, {**meta, "rows_read": n}


def check_states(vd, ds, reference, targets):
    """Check identity-joined samples against the saved reference, not by proximity."""
    first = subset(vd, (vd["sid"] == 1) & (vd["pz"] > 0))
    _, take = np.unique(first["target"], return_index=True)
    first = subset(first, take)
    target = lookup(targets, ids(ds))
    if np.any(target < 0):
        raise ValueError('DS state outside requested identities')
    endpoint = subset(ds, ds['time'] == reference['stock_t'][target])
    for table, prefix in ((first, ""), (endpoint, "stock_")):
        key = ids(table)
        rows = lookup(key, targets)
        if np.any(rows < 0):
            raise ValueError("missing recorded VD1 entrance or DS endpoint for exact ID")
        for fields, k in ((("x", "y", "z"), "x"), (("px", "py", "pz"), "p"), (("time",), "t")):
            actual = np.column_stack([table[f][rows] for f in fields])
            expected = reference[prefix + k].reshape(actual.shape)
            if not np.array_equal(actual, expected):
                raise ValueError(f"identity-joined {prefix + k} disagrees with reference")


def extract(plans, targets, reference):
    """Read only ntvd/nttvd for requested shards; return small crossing columns."""
    tables, provenance = {"vd": [], "ds": []}, []
    for path, c, census_path in plans:
        meta = {"ntuple": str(path), "census": census_path, "index": c["index"], "trees": {}}
        for label, tree, count in (("vd", "ntvd", c["vd_rows"]),
                                   ("ds", "nttvd", c["ds_beam_all_species_rows"])):
            a, schema = read_small(path, tree, count)
            if np.any(a["run"] != 1430) or np.any(a["subrun"] != 920000 + c["index"]):
                raise ValueError("ROOT table contains identities outside declared shard")
            tables[label].append(select_exact(a, targets))
            meta["trees"][label] = schema
        provenance.append(meta)
    merged = {}
    for label, chunks in tables.items():
        if any(set(a) != set(chunks[0]) for a in chunks):
            raise ValueError("incompatible scalar schemas across requested shards")
        a = {k: np.concatenate([chunk[k] for chunk in chunks]) for k in chunks[0]}
        order = np.lexsort((a["entry"], a["time"], a["target"]))
        merged[label] = subset(a, order)
    check_states(merged["vd"], merged["ds"], reference, targets)
    kind = np.concatenate([np.full(len(merged[k]["time"]), j, dtype=np.int8) for j, k in enumerate(("vd", "ds"))])
    row = np.concatenate([np.arange(len(merged[k]["time"])) for k in ("vd", "ds")])
    target = np.concatenate([merged[k]["target"] for k in ("vd", "ds")])
    times = np.concatenate([merged[k]["time"] for k in ("vd", "ds")])
    order = np.lexsort((row, kind, times, target))
    timeline = {"kind": kind[order], "row": row[order], "target": target[order], "time": times[order]}
    arrays = {f"{label}_{k}": v for label, a in merged.items() for k, v in a.items()}
    arrays.update({"target_id": targets, **{f"timeline_{k}": v for k, v in timeline.items()},
                   **{f"reference_{k}": v for k, v in reference.items()}})
    report = {"inputs": provenance, "tables": {label: {k: v.tolist() for k, v in a.items()} for label, a in merged.items()},
              "timeline": {k: v.tolist() for k, v in timeline.items()}, "timeline_kind": {"0": "VD", "1": "DS Beam"},
              "counts": {k: np.bincount(a["target"], minlength=len(targets)).tolist() for k, a in merged.items()},
              "reference_states_exactly_equal": True, "ntpart_rows_read": 0,
              "ordering": "target input order, ascending time; ties: VD before DS, then original entry",
              "units": {"position": "mm", "momentum": "MeV/c", "time": "ns", "ke": "MeV"}}
    return arrays, report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--census", nargs="+", type=Path)
    p.add_argument("--id", nargs=4, type=int, action="append", dest="target_ids", metavar=("RUN", "SUBRUN", "EVENT", "TRACK"))
    p.add_argument("--check-reference", action="store_true", help="verify existing exact identity only; no ROOT reads/output writes")
    p.add_argument("--out", type=Path, help="new NPZ path; paired JSON also must not exist")
    args = p.parse_args(argv)
    started = time.monotonic()
    targets = targets_checked(TARGETS if args.target_ids is None else args.target_ids)
    if not args.check_reference:
        if args.out is None or args.out.suffix != ".npz" or not args.census:
            p.error("extraction requires --census and --out NEW.npz")
        if args.out.exists() or args.out.with_suffix(".json").exists():
            raise FileExistsError("refusing to overwrite extraction outputs")
        plans = census_inputs(args.census, targets)
    reference, refmeta = reference_identity(args.reference, targets)
    report = {"cohort": "development", "target_id": targets.tolist(), "reference": refmeta, "limits": LIMITS}
    if args.check_reference:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    arrays, details = extract(plans, targets, reference)
    report.update(details, seconds=time.monotonic() - started)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with args.out.with_suffix(".json").open("x") as stream:
        stream.write(text)
    print(json.dumps({"out": str(args.out), "target_id": targets.tolist(), "counts": report["counts"],
                      "seconds": report["seconds"], "ntpart_rows_read": 0}))


if __name__ == "__main__":
    main()
