"""Exact-ID selection of recorded stock VD/DS crossings, not material trajectories."""
import argparse
from pathlib import Path

import numpy as np

from .tsreference import IDS, ids, lookup, read_table, subset


def targets_checked(targets):
    """Return nonempty, distinct exact (run, subrun, event, track) keys."""
    a = np.asarray(targets)
    if a.ndim != 2 or a.shape[1] != 4 or not len(a):
        raise ValueError("nonempty N by 4 exact IDs required")
    a = ids(dict(zip(IDS, a.T)))
    lookup(a, a)
    return a


def select_exact(table, targets):
    """Select all crossings by identity, retaining repeats, directions and scalar columns."""
    targets = targets_checked(targets)
    target = lookup(targets, ids(table))
    keep = np.flatnonzero(target >= 0)
    out = subset(table, keep)
    if not np.all(np.isfinite(out["time"])):
        raise ValueError("nonfinite crossing time")
    order = np.lexsort((keep, out["time"], target[keep]))
    out = subset(out, order)
    out["entry"] = keep[order]
    out["target"] = target[keep][order]
    out["momentum_mev_c"] = np.sqrt(sum(out[k].astype(float) ** 2 for k in ("px", "py", "pz")))
    return out


def extract(paths, targets):
    """Read each supplied file's highest-cycle ntvd/nttvd tables; never read ntpart."""
    targets = targets_checked(targets)
    paths = list(paths)
    if not paths:
        raise ValueError("at least one stock ntuple required")
    tables = {"vd": [], "ds": []}
    for file_index, path in enumerate(paths):
        for label, tree in (("vd", "ntvd"), ("ds", "nttvd")):
            table, _ = read_table(path, tree)
            selected = select_exact(table, targets)
            selected["file_index"] = np.full(len(selected["time"]), file_index, dtype=np.int64)
            tables[label].append(selected)
    arrays = {"target_id": targets}
    for label, chunks in tables.items():
        if any(set(a) != set(chunks[0]) for a in chunks):
            raise ValueError("incompatible scalar schemas across stock ntuples")
        a = {k: np.concatenate([chunk[k] for chunk in chunks]) for k in chunks[0]}
        order = np.lexsort((a["entry"], a["file_index"], a["time"], a["target"]))
        arrays.update({f"{label}_{k}": v[order] for k, v in a.items()})
    return arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ntuple", nargs="+", required=True, type=Path)
    parser.add_argument("--id", nargs=4, type=int, action="append", required=True,
                        dest="target_ids", metavar=("RUN", "SUBRUN", "EVENT", "TRACK"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("refusing to overwrite crossing output")
    arrays = extract(args.ntuple, args.target_ids)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


if __name__ == "__main__":
    main()
