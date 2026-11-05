"""Audit persisted beam reference rows and cached-table identity without rerunning transport.

A particle tree certifies represented events, not events with no saved muons. Job completion and the source/config
provenance must be reviewed separately; hashes identify saved artifacts, not the binaries that produced them.
"""
import hashlib
from dataclasses import asdict
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


def audit(paths, table=None, artifacts=(), geom_file=None, offline_root=None, baseline_geom=None):
    """Read highest ntpart cycles, count unique IDs, and compare raw primary columns to a saved NPZ."""
    if baseline_geom is not None and geom_file is None:
        raise ValueError("geom_file is required with baseline_geom")
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
    derived = None
    if geom_file is not None:
        from . import geometry
        if offline_root is None:
            raise ValueError("offline_root is required for geometry provenance")
        sources = []
        geom = geometry.from_offline(geom_file, offline_root, source_files=sources)
        geometry.validate_foils(geom, np.finfo(float).tiny)
        at_rest = selected["pstop"] < 1e-6
        foil = np.where(at_rest, geometry.foil_index(geom, selected["xstop"], selected["ystop"],
                                                   selected["zstop"]), -1)
        derived = dict(q=np.where(selected["pdg"] == 13, -1.0, 1.0), at_rest=at_rest, foil=foil)
        report["geometry"] = dict(
            sources=[_identity(p) for p in sources],
            effective={k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in asdict(geom).items()},
            scope="Current beam reader include closure, not historical G4 binary/geometry certification",
            at_rest=int(at_rest.sum()), target_stops=int((foil >= 0).sum()),
            foil_profile=np.bincount(foil[foil >= 0], minlength=len(geom.foil_z)).tolist())
        if baseline_geom is not None:
            from . import simpleconfig
            candidate = Path(geom_file) if Path(geom_file).is_absolute() else Path(offline_root) / geom_file
            baseline = Path(baseline_geom) if Path(baseline_geom).is_absolute() else Path(offline_root) / baseline_geom
            report["geometry"]["allowed_overlay_changes"] = geometry.check_overlay(
                simpleconfig.read(candidate, offline_root), simpleconfig.read(baseline, offline_root))
    if table is not None:
        saved = load(table)
        match = {b: b in saved and np.array_equal(selected[b], saved[b], equal_nan=True) for b in BRANCHES}
        report["table"] = dict(_identity(table), rows=len(saved["evt"]), raw_columns_match=match,
                               all_raw_columns_match=all(match.values()),
                               derived_columns_checked=derived is not None)
        if derived is not None:
            match = {b: b in saved and np.array_equal(v, saved[b], equal_nan=True)
                     for b, v in derived.items()}
            report["table"].update(derived_columns_match=match, all_derived_columns_match=all(match.values()))
    return report
