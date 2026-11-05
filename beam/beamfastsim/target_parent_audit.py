"""Audit an ordered stopped-muon parent tree without equating events to DIO electrons."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import geometry as G


MU_MINUS_CAPTURE_AT_REST = 32


def reduce_parent(columns, geom, expected_events=None):
    """Match stoppedMuMinusList: PDG 13 and capture-at-rest code, preserving event order."""
    keys = ("run", "subrun", "evt", "trk", "pdg", "codestop", "pstop", "xstop", "ystop", "zstop", "tstop")
    if any(k not in columns for k in keys):
        raise ValueError("missing parent identity or stop columns")
    a = {k: np.asarray(columns[k]) for k in keys}
    n = a["pdg"].size
    if not n or any(v.shape != (n,) for v in a.values()):
        raise ValueError("parent columns must be nonempty equal-length vectors")
    if any(not np.isfinite(v).all() for v in a.values()):
        raise ValueError("nonfinite parent columns")
    identity = np.column_stack([a[k] for k in ("run", "subrun", "evt", "trk")])
    if np.unique(identity, axis=0).shape[0] != n:
        raise ValueError("duplicate parent event/track identity")
    event, first, inverse = np.unique(identity[:, :3], axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size)
    event_index = rank[inverse]
    if np.any(np.diff(event_index) < 0):
        raise ValueError("parent events are not contiguous in tree order")
    nevent = event.shape[0]
    if expected_events is not None and (expected_events < 1 or nevent != expected_events):
        raise ValueError("parent tree event count does not equal expected art entries; zero-row events may be missing")
    sel = (a["pdg"] == 13) & (a["codestop"] == MU_MINUS_CAPTURE_AT_REST)
    if not sel.any():
        raise ValueError("no captured negative muons in parent")
    counts = np.bincount(event_index[sel], minlength=nevent)
    if np.any(counts == 0):
        raise ValueError("parent contains events without eligible muons")
    out = {k: v[sel] for k, v in a.items()}
    out["event_index"] = event_index[sel]
    out["at_rest"] = np.ones(int(sel.sum()), bool)
    out["foil"] = G.foil_index(geom, out["xstop"], out["ystop"], out["zstop"])
    if np.any(out["foil"] < 0):
        raise ValueError("captured muons outside staged target foils; generator and source selections would differ")
    out["event_keys"] = event[order]
    out["event_multiplicity"] = counts
    unique, frequency = np.unique(counts, return_counts=True)
    report = dict(parent_events=nevent, eligible_muons=int(sel.sum()),
        electrons_per_parent_event=float(counts.mean()), multiplicity={str(int(k)): int(v) for k, v in zip(unique, frequency)},
        selection="PDG 13 and stopping ProcessCode 32, exactly stoppedMuMinusList; no extra momentum cut",
        max_end_momentum=float(out["pstop"].max()), event_order_preserved=True,
        expected_event_count_verified=expected_events is not None, exact_production_electron_exposure_known=False,
        caveat="full-pool mean is not a particular production job's sequential window; audit its mixer skip or parent IDs")
    return out, report


def window_exposure(counts, n):
    """All circular sequential-window electron counts, not independent multinomial draws."""
    counts = np.asarray(counts)
    if counts.ndim != 1 or not counts.size or np.any(counts < 1) or np.any(counts != np.floor(counts)):
        raise ValueError("positive integer event multiplicities required")
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise ValueError("positive integer generated event count required")
    cycles, rest = divmod(n, counts.size)
    base = cycles * counts.sum(dtype=np.int64)
    extended = np.r_[counts, counts[:rest]]
    prefix = np.r_[0, np.cumsum(extended, dtype=np.int64)]
    windows = base + prefix[np.arange(counts.size) + rest] - prefix[np.arange(counts.size)]
    return dict(generated_events=int(n), mean_electrons=float(windows.mean()), min_electrons=int(windows.min()),
                max_electrons=int(windows.max()), window_sd=float(windows.std()),
                exact_job_exposure=False, method="all possible circular sequential start offsets in ordered parent pool")


def production_exposure(parent, reference, generated_events, first_event=1):
    """Infer each sequential mixer window from exact birth-position matches and generated event IDs."""
    if not isinstance(generated_events, (int, np.integer)) or generated_events < 1:
        raise ValueError("positive generated event window required")
    xyz = np.column_stack([parent[k] for k in ("xstop", "ystop", "zstop")]).astype("<f4")
    starts = np.column_stack([reference[k] for k in ("x", "y", "z")]).astype("<f4")
    if not np.isfinite(xyz).all() or not np.isfinite(starts).all() or not starts.size:
        raise ValueError("finite nonempty stop and birth positions required")
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    keys = np.ascontiguousarray(xyz).view(dtype).ravel()
    query = np.ascontiguousarray(starts).view(dtype).ravel()
    order = np.argsort(keys)
    left = np.searchsorted(keys[order], query, side="left")
    right = np.searchsorted(keys[order], query, side="right")
    if np.any(right - left != 1):
        raise ValueError("birth positions must match exactly one parent stop each; no nearest-neighbor fallback")
    matched_event = np.asarray(parent["event_index"])[order[left]]
    counts = np.asarray(parent["event_multiplicity"])
    evt = np.asarray(reference["evt"])
    if evt.shape != matched_event.shape or np.any(evt != np.floor(evt)):
        raise ValueError("invalid generated event IDs")
    relative = evt.astype(np.int64) - first_event
    if np.any(relative < 0) or np.any(relative >= generated_events):
        raise ValueError("reference event ID outside declared generated window")
    offsets = (matched_event - relative) % counts.size
    prefix = np.r_[0, np.cumsum(np.tile(counts, generated_events // counts.size + 2), dtype=np.int64)]
    reports = []
    for group in np.unique(reference["ifile"]):
        sel = np.asarray(reference["ifile"]) == group
        candidates = np.unique(offsets[sel])
        if candidates.size != 1:
            values, frequencies = np.unique(offsets[sel], return_counts=True)
            raise ValueError("matched starts do not imply one sequential mixer offset per file: "
                             + json.dumps(dict(ifile=int(group), offsets=values.tolist(), rows=frequencies.tolist(),
                                               event_range=[int(evt[sel].min()), int(evt[sel].max())])))
        offset = int(candidates[0])
        denominator = int(prefix[offset + generated_events] - prefix[offset])
        reports.append(dict(ifile=int(group), inferred_first_parent_event_index=offset,
            matched_electrons=int(sel.sum()), matched_generated_events=int(np.unique(evt[sel]).size),
            generated_events=int(generated_events), generated_electrons=denominator,
            denominator_method="all persisted electron birth positions uniquely exact-float32 joined to parent stops; unanimous event-index minus generated-event offset; sum complete sequential window"))
    return dict(files=reports, total_generated_electrons=sum(r["generated_electrons"] for r in reports),
                exact_birth_matches=int(starts.shape[0]), exact_production_electron_exposure_known=True,
                filter_implication_certified=False, generated_yield_validated=False,
                assumptions="sequential one-parent-event mixer, unchanged positions, first generated event explicitly declared, all captured muons emit one DIO electron")


def physical_event_order(parent, art_path, byteswap=False):
    """Map extracted event identities to the parent art Events-entry order through FileIndex metadata."""
    import awkward as ak
    import uproot
    with uproot.open(art_path) as root:
        a = root["FileIndex"]["Element"].array(library="ak")
        event = a.eventID
        run = ak.to_numpy(event.subRun_.run_.run_)
        subrun = ak.to_numpy(event.subRun_.subRun_)
        event_number, entries = ak.to_numpy(event.event_), ak.to_numpy(a.entry)
        if byteswap:
            run, subrun, event_number, entries = (v.byteswap() for v in (run, subrun, event_number, entries))
        valid = (event_number > 0) & (event_number != np.iinfo(np.uint32).max)
        valid &= subrun != np.iinfo(np.uint32).max
        keys = np.column_stack([run[valid], subrun[valid], event_number[valid]])
        entries = entries[valid]
        n = root["Events"].num_entries
    if keys.shape[0] != n or not np.array_equal(np.sort(entries), np.arange(n)):
        raise ValueError("FileIndex does not identify every physical Events entry exactly once: " + json.dumps(dict(
            expected=int(n), index_rows=int(keys.shape[0]), unique_entries=int(np.unique(entries).size),
            entry_min=int(entries.min()), entry_max=int(entries.max()),
            subrun_range=[int(keys[:, 1].min()), int(keys[:, 1].max())],
            event_min=int(keys[:, 2].min()), event_max=int(keys[:, 2].max()))))
    keys = keys[np.argsort(entries)]
    old = parent["event_keys"]
    dtype = np.dtype([("run", "<i8"), ("subrun", "<i8"), ("evt", "<i8")])
    packed = np.ascontiguousarray(keys, dtype="<i8").view(dtype).ravel()
    query = np.ascontiguousarray(old, dtype="<i8").view(dtype).ravel()
    order = np.argsort(packed)
    left, right = np.searchsorted(packed[order], query, "left"), np.searchsorted(packed[order], query, "right")
    if np.any(right - left != 1):
        raise ValueError("parent event keys do not join uniquely to FileIndex")
    mapping = order[left]
    out = dict(parent)
    out["event_index"] = mapping[parent["event_index"]]
    out["event_keys"] = keys
    out["event_multiplicity"] = np.empty_like(parent["event_multiplicity"])
    out["event_multiplicity"][mapping] = parent["event_multiplicity"]
    return out, dict(method="exact event-key join to FileIndex, sorted by physical Events entry",
                     reordered_events=int((mapping != np.arange(n)).sum()), art_events=int(n),
                     explicit_uproot_metadata_byteswap=bool(byteswap))


def checksum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    import uproot
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("--tree", default="cetree/ntpart")
    ap.add_argument("--expected-events", type=int, required=True)
    ap.add_argument("--generated-events", type=int, default=5000)
    ap.add_argument("--offline-root", required=True)
    ap.add_argument("--geom", default="Offline/Mu2eG4/geom/geom_run1_a.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", help="optional reduced DIO starts to infer exact sequential production windows")
    ap.add_argument("--first-event", type=int, default=1)
    ap.add_argument("--parent-art", help="recover physical mixer order from art FileIndex metadata")
    ap.add_argument("--fileindex-byteswap", action="store_true", help="explicit workaround for byte-reversed custom-streamer metadata; exact event join still required")
    a = ap.parse_args(argv)
    names = ("run", "subrun", "evt", "trk", "pdg", "codestop", "pstop", "xstop", "ystop", "zstop", "tstop")
    with uproot.open(a.root) as root:
        tree = root[a.tree]
        columns = tree.arrays(list(names), library="np")
        cycles = {k: int(root[k].num_entries) for k in root.keys() if k.split(";")[0] == a.tree}
    out, report = reduce_parent(columns, G.from_offline(a.geom, a.offline_root), a.expected_events)
    if a.parent_art:
        out, report["physical_event_order"] = physical_event_order(out, a.parent_art, a.fileindex_byteswap)
        report["parent_art_sha256"] = checksum(a.parent_art)
    report.update(tree_cycles=cycles, input_sha256=checksum(a.root),
                  sequential_windows=window_exposure(out["event_multiplicity"], a.generated_events))
    if a.reference:
        with np.load(a.reference) as reference:
            report["production_exposure"] = production_exposure(out, reference, a.generated_events, a.first_event)
        report["production_exposure"]["reference_sha256"] = checksum(a.reference)
    np.savez_compressed(a.out, **out, metadata=json.dumps(report))
    Path(a.out).with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
