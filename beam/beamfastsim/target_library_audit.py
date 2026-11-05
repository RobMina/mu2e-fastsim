"""Independent saved-log and selected-parent audit for fresh Run1 muon libraries."""
import argparse
import json
from pathlib import Path

import numpy as np

from .target_inclusive_ledger import read_log, summary
from .target_parent_audit import G, checksum, physical_event_order, reduce_parent


PROCESSES = ("R1LibraryPOT", "R1LibraryStops")
COLUMNS = ("run", "subrun", "evt", "trk", "pdg", "codestop", "pstop", "xstop", "ystop", "zstop", "tstop")
TREE = "cetree/ntpart"


def positive(value, name, maximum=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < 1 or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} must be a positive integer" + (f" <= {maximum}" if maximum else ""))


def sorted_keys(keys):
    return np.asarray(keys)[np.lexsort(np.asarray(keys).T[::-1])]


def reconcile(generation_log, read_path, process, expected_events, minimum_stop_events=1):
    """Reconcile all events and counters before admitting any selected event."""
    positive(expected_events, "expected_events", 20000)
    positive(minimum_stop_events, "minimum_stop_events")
    if process not in PROCESSES:
        raise ValueError("unsupported library process")
    original = read_log(generation_log, expected_events, process, require_triggers=False)
    ledger = read_log(read_path, expected_events, process, require_triggers=True)
    go = np.lexsort(original["identities"].T[::-1])
    ro = np.lexsort(ledger["identities"].T[::-1])
    for key in ("identities", "status", "nG4Tracks", "overflowSimP", "nStepLimit"):
        if not np.array_equal(original[key][go], ledger[key][ro]):
            raise ValueError("saved read differs from independent generation census: " + key)
    ledger["generation_source_sha256"] = original["source_sha256"]
    for key in ("status", "overflowSimP", "nStepLimit"):
        if np.any(ledger[key] != 0):
            values, counts = np.unique(ledger[key], return_counts=True)
            raise ValueError(f"library blocked: all-event {key} must be zero; "
                             + str(dict(zip(values.tolist(), counts.tolist()))))
    accepted = sorted_keys(ledger["identities"][ledger["accepted"] == 1])
    minimum = minimum_stop_events if process == "R1LibraryStops" else 1
    if len(accepted) < minimum:
        raise ValueError("library blocked: insufficient accepted events; no automatic refill")
    report = summary(ledger)
    report.update(generation_source_sha256=str(original["source_sha256"].item()),
                  read_source_sha256=str(ledger["source_sha256"].item()),
                  accepted_event_keys=accepted.tolist(), all_event_status_zero=True,
                  filter_implication_certified=False, generated_yield_validated=False,
                  fresh_stage_events=int(expected_events),
                  fresh_POT_events=int(expected_events) if process == "R1LibraryPOT" else None,
                  beam_draw_events=int(expected_events) if process == "R1LibraryStops" else None,
                  independence_scope="distinct parent arms require fresh POT provenance; repeated beam draws are not independent parents",
                  POT_normalization_inferred=False, automatic_refill=False)
    return ledger, report


def selected_parent(columns, geom, accepted_keys):
    """Keep all eligible stops, checking the exact accepted event set before reduction."""
    columns = {key: np.asarray(columns[key]) for key in COLUMNS}
    for key in ("run", "subrun", "evt", "trk", "pdg", "codestop"):
        a = np.asarray(columns[key])
        if (not np.isfinite(a).all() or np.any(a != np.floor(a))
                or (key != "pdg" and np.any(a < 0))
                or (a.dtype.kind == "f" and np.any(np.abs(a) >= 2**24))):
            raise ValueError("unsafe fractional or float32-ambiguous parent identity/code: " + key)
    keys = np.column_stack([columns[k] for k in ("run", "subrun", "evt")])
    accepted = np.asarray(accepted_keys)
    if not len(accepted) or not np.array_equal(sorted_keys(np.unique(keys, axis=0)), sorted_keys(accepted)):
        raise ValueError("reader event keys must equal the exact accepted ledger subset")
    return reduce_parent(columns, geom, expected_events=len(accepted))


def art_entries(path):
    """uproot's unqualified Events lookup selects the highest persisted cycle."""
    import uproot
    with uproot.open(path) as root:
        return int(root["Events"].num_entries)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for option in ("generation-log", "read-log", "out"):
        ap.add_argument("--" + option, type=Path, required=True)
    ap.add_argument("--process", choices=PROCESSES, required=True)
    ap.add_argument("--expected-events", type=int, required=True)
    ap.add_argument("--minimum-stop-events", type=int, default=1)
    ap.add_argument("--beam-parent-events", type=int)
    for option in ("parent-root", "parent-art", "offline-root", "parent-out"):
        ap.add_argument("--" + option, type=Path)
    ap.add_argument("--fileindex-byteswap", action="store_true")
    a = ap.parse_args(argv)
    outputs = [a.out] + ([a.parent_out] if a.parent_out else [])
    if len(set(outputs)) != len(outputs):
        raise ValueError("JSON and NPZ output paths must differ")
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"refusing existing output: {path}")
    if a.parent_root and (a.process != "R1LibraryStops" or not a.parent_art or not a.offline_root):
        raise ValueError("selected stops extraction requires R1LibraryStops, parent-art and offline-root")
    if (a.parent_out or a.offline_root or a.fileindex_byteswap) and not a.parent_root:
        raise ValueError("parent extraction options require parent-root")
    if a.beam_parent_events is not None:
        positive(a.beam_parent_events, "beam_parent_events", 20000)
        if a.process != "R1LibraryStops":
            raise ValueError("beam-parent-events describes stop-stage provenance only")
    ledger, report = reconcile(a.generation_log, a.read_log, a.process, a.expected_events, a.minimum_stop_events)
    report["beam_parent_events"] = a.beam_parent_events
    report["beam_parent_events_scope"] = "declared provenance only; no repeated-draw POT normalization or independence inference"
    parent = None
    if a.parent_art:
        entries = art_entries(a.parent_art)
        if entries != report["accepted_events"]:
            raise ValueError("selected art Events count does not equal accepted PrimaryPath count")
        report.update(parent_art_sha256=checksum(a.parent_art), selected_art_events=entries)
    if a.parent_root:
        import uproot
        with uproot.open(a.parent_root) as root:
            columns = root[TREE].arrays(list(COLUMNS), library="np")
            cycles = {k: int(root[k].num_entries) for k in root.keys() if k.split(";")[0] == TREE}
        geom = G.from_offline("Offline/Mu2eG4/geom/geom_run1_a.txt", str(a.offline_root))
        parent, report["parent"] = selected_parent(columns, geom, ledger["identities"][ledger["accepted"] == 1])
        parent, report["physical_event_order"] = physical_event_order(parent, a.parent_art, a.fileindex_byteswap)
        report.update(parent_tree_sha256=checksum(a.parent_root), parent_tree=TREE, tree_cycles=cycles,
                      library_muons=int(parent["pdg"].size), selected_parent_verified=True)
    else:
        report["selected_parent_verified"] = False
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    created = []
    try:
        if a.parent_out:
            with a.parent_out.open("xb") as stream:
                created.append(a.parent_out)
                np.savez_compressed(stream, **parent, metadata=np.asarray(encoded))
        with a.out.open("x") as stream:
            created.append(a.out)
            stream.write(encoded)
    except BaseException:
        for path in created:
            path.unlink()
        raise
    print(encoded, end="")


if __name__ == "__main__":
    main()
