"""Audit persisted beam reference rows and cached-table identity without rerunning transport.

A particle tree certifies represented events, not events with no saved muons. Job completion and the source/config
provenance must be reviewed separately; hashes identify saved artifacts, not the binaries that produced them.
"""
import hashlib
from pathlib import Path

import numpy as np
import uproot

from .g4ref import BRANCHES, MU2E_PRIMARY, load


def _identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _ids(arrays, fields):
    values = np.column_stack([arrays[k] for k in fields])
    if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
        raise ValueError(f"Nonfinite or nonintegral identity fields: {fields}")
    return values.astype(np.int64)


def audit(paths, table=None, artifacts=()):
    """Read highest ntpart cycles, count unique IDs, and compare raw primary columns to a saved NPZ."""
    cols = {b: [] for b in BRANCHES}
    files = []
    for path in paths:
        with uproot.open(path) as root:
            directory = root["mutree"]
            cycles = sorted(int(k.rsplit(";", 1)[1]) for k in directory.keys(cycle=True, recursive=False)
                            if k.rsplit(";", 1)[0] == "ntpart")
            if not cycles:
                raise ValueError(f"No mutree/ntpart in {path}")
            key = f"mutree/ntpart;{cycles[-1]}"
            tree = root[key]
            arrays = tree.arrays(BRANCHES, library="np")
            files.append(dict(_identity(path), selected_key=key, entries=int(tree.num_entries),
                              cycles={str(c): int(root[f"mutree/ntpart;{c}"].num_entries) for c in cycles}))
            for b in BRANCHES:
                cols[b].append(arrays[b])
    if not files:
        raise ValueError("At least one ROOT reference is required")
    arrays = {b: np.concatenate(v) for b, v in cols.items()}
    event_ids = np.unique(_ids(arrays, ("run", "subrun", "evt")), axis=0)
    primary = arrays["code"] == MU2E_PRIMARY
    if np.any(primary & ~np.isin(arrays["pdg"], [13, -13])):
        raise ValueError("Non-muon primary rows in the muon reference")
    selected = {b: v[primary] for b, v in arrays.items()}
    primary_ids = _ids(selected, ("run", "subrun", "evt", "trk"))
    unique_primaries = np.unique(primary_ids, axis=0)
    report = dict(schema="beam-reference-audit-v1", files=files, particle_rows=len(arrays["evt"]),
                  represented_events=len(event_ids), primary_muons=int(primary.sum()),
                  primary_events=len(np.unique(primary_ids[:, :3], axis=0)),
                  duplicate_primary_ids=len(primary_ids) - len(unique_primaries),
                  event_id_first=event_ids[0].tolist() if len(event_ids) else None,
                  event_id_last=event_ids[-1].tolist() if len(event_ids) else None,
                  event_count_scope="Distinct IDs in persisted particle rows; empty-muon events are not certified",
                  artifacts=[_identity(p) for p in artifacts])
    if table is not None:
        saved = load(table)
        match = {b: b in saved and np.array_equal(selected[b], saved[b], equal_nan=True) for b in BRANCHES}
        report["table"] = dict(_identity(table), rows=len(saved["evt"]), raw_columns_match=match,
                               all_raw_columns_match=all(match.values()),
                               derived_columns_checked=False)
    return report
